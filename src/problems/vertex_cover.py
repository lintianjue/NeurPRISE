import logging

import torch
from typing import Dict, Any, List, Tuple, Optional

from torch_geometric.data import Data

from src.core.problem import IProblem
from src.core.types import Instance, Labels
import numpy as np

from src.problems.scenario_generator import BaseScenarioGenerator, IidUniformScenarioGenerator
from src.utils.graph_batching import build_batched_scenario_graph, build_batched_det_graph
from src.utils.label_utils import _collate_forward_trace
from src.utils.seed import get_rng
import gurobipy as gp
from gurobipy import GRB

class VertexCoverProblem(IProblem):
    _PYG_CACHE_KEY = "vc_pyg_data_list_v1"

    def __init__(
        self,
        n_nodes: int,
        n_scenarios: int,
        n_min_cost: int = 1,
        n_max_cost: int = 100,
        p_edge: float = 10.0,
        scenario_name: str = "uniform",
        scenario_generator: Optional[BaseScenarioGenerator] = None,
        edge_attr_include_abs: bool = False, # simple
        **kwargs,
    ):
        self.n_nodes = n_nodes
        self.n_scenarios = n_scenarios
        self.n_min_cost = n_min_cost
        self.n_max_cost = n_max_cost
        self.p_edge = p_edge
        self.scenario_name = scenario_name
        self._rng = get_rng()
        self.edge_attr_include_abs = bool(edge_attr_include_abs)

        # if no generator is provided, fall back to iid-uniform
        if scenario_generator is None:
            self.scenario_generator = IidUniformScenarioGenerator(
                n_min_cost=n_min_cost,
                n_max_cost=n_max_cost,
                rng=self._rng,
            )
        else:
            self.scenario_generator = scenario_generator

        self.det_mat: np.ndarray = np.array([])
        self.sce_mat: np.ndarray = np.array([])
        self.edges: List[Tuple[int, int]] = []
        self.edge_adjacency: np.ndarray = np.zeros((self.n_nodes, self.n_nodes), dtype=int)

        # CCG solver defaults
        self.exact_backend = "bigmilp"
        self.eval_full_obj_source = "label"
        self.ccg_max_iterations = 20
        self.ccg_convergence_tolerance = 1e-4
        self.ccg_duplicate_tolerance = 1e-6
        self.ccg_recourse_time_limit: Optional[float] = None
        self.ccg_mip_gap: Optional[float] = None

    def configure_ccg_solver(
        self,
        *,
        max_iterations: int,
        convergence_tolerance: float,
        duplicate_tolerance: float,
        recourse_time_limit: Optional[float],
        mip_gap: Optional[float] = None,
    ) -> None:
        self.ccg_max_iterations = int(max_iterations)
        self.ccg_convergence_tolerance = float(convergence_tolerance)
        self.ccg_duplicate_tolerance = float(duplicate_tolerance)
        self.ccg_recourse_time_limit = (
            float(recourse_time_limit) if recourse_time_limit is not None else None
        )
        self.ccg_mip_gap = float(mip_gap) if mip_gap is not None else None


    def sample(self) -> Instance:
        # 1) Costs from strategy
        self.det_mat, self.sce_mat = self.scenario_generator.sample_costs(
            n_nodes=self.n_nodes,
            n_scenarios=self.n_scenarios,
        )

        # 2) Graph
        self._sample_edges()

        return Instance(
            {
                "n_nodes": np.array([self.n_nodes]),
                "n_scenarios": np.array([self.n_scenarios]),
                "n_min_cost": np.array([self.n_min_cost]),
                "n_max_cost": np.array([self.n_max_cost]),
                "p_edge": np.array([self.p_edge]),
                "det": self.det_mat,
                "sce": self.sce_mat,
                "adjacency": self.edge_adjacency,
            }
        )

    def _sample_edges(self) -> None:
        self.edges = []
        self.edge_adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=int)

        p = self.p_edge / max(1, self.n_nodes)

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self._rng.random() < p:
                    self.edges.append((i, j))
                    self.edge_adjacency[i, j] = 1
                    self.edge_adjacency[j, i] = 1


    def solve_robust_bigmilp(
            self,
            instance: Instance,
            time_limit: Optional[float] = None,
            mip_gap: Optional[float] = None,
            threads: int = 1,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        A = instance.data["adjacency"]  # (N, N) or (B, N, N)
        c_det = instance.data["det"]  # (N,)   or (B, N)
        sce = instance.data["sce"]  # (S, N) or (B, S, N)

        # Normalize to single-instance shapes
        if A.ndim == 3:
            A = A[0]
        if c_det.ndim > 1:
            c_det = c_det[0]
        if sce.ndim > 2:
            sce = sce[0]

        N = A.shape[0]
        S = sce.shape[0]

        # Build edge list from adjacency (u < v to avoid duplicates)
        edges: List[Tuple[int, int]] = []
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v] != 0:
                    edges.append((u, v))

        # --- Build and solve MILP ---
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1) if verbose else env.setParam("OutputFlag", 0)
        env.setParam("Threads", threads)
        if mip_gap is not None and mip_gap >= 0:
            env.setParam("MIPGap", float(mip_gap))

        if time_limit is not None and time_limit > 0:
            env.setParam("TimeLimit", float(time_limit))
        env.start()

        m = gp.Model(env=env)

        # Variables
        x = m.addVars(N, vtype=GRB.BINARY, name="x")
        y = m.addVars(S, N, vtype=GRB.BINARY, name="y")  # y[i,j]
        t = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t")

        # recourse cost constraints: sum_j sce[i,j]*y[i,j] <= t
        m.addConstrs(
            (
                gp.quicksum(float(sce[i, j]) * y[i, j] for j in range(N)) <= t
                for i in range(S)
            ),
            name="rec_cost",
        )

        # cover constraints: for each scenario and edge (u,v)
        m.addConstrs(
            (
                x[u] + y[i, u] + x[v] + y[i, v] >= 1.0
                for i in range(S)
                for (u, v) in edges
            ),
            name="cover",
        )

        # y <= 1 - x (equivalent to x[j] + y[i,j] <= 1)
        m.addConstrs(
            (
                x[j] + y[i, j] <= 1.0
                for i in range(S)
                for j in range(N)
            ),
            name="y_le_1_minus_x",
        )

        # Objective: sum_j det[j]*x_j + t
        m.setObjective(
            gp.quicksum(float(c_det[j]) * x[j] for j in range(N)) + t,
            GRB.MINIMIZE,
        )

        m.optimize()

        status = m.Status
        has_incumbent = (m.SolCount > 0)

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            if not has_incumbent:
                # Explicitly signal infeasibility / no solution found
                return {
                    "status": int(status),
                    "has_incumbent": False,
                    "obj": None,
                    "obj_bound": float(m.ObjBound) if hasattr(m, "ObjBound") else None,
                    "mip_gap": None,
                    "threads": int(m.Params.Threads),
                    "x": None,
                    "t": None,
                    "wc_idx": None,
                    "runtime_sec": float(m.Runtime),
                }

            # Feasible incumbent exists → SAFE to read values
            x_star = [1.0 if x[j].X > 0.5 else 0.0 for j in range(N)]
            t_star = float(t.X)

            return {
                "status": int(status),
                "has_incumbent": True,
                "obj": float(m.ObjVal),        # ← BEST OBJECTIVE (use this)
                "obj_bound": float(m.ObjBound),
                "mip_gap": float(m.MIPGap),
                "threads": int(m.Params.Threads),
                "x": x_star,
                "t": t_star,
                "wc_idx": None,                # optional: compute only if needed
                "runtime_sec": float(m.Runtime),
            }

        else:
            # INFEASIBLE / UNBOUNDED / ERROR
            return {
                "status": int(status),
                "has_incumbent": False,
                "obj": None,
                "obj_bound": None,
                "mip_gap": None,
                "threads": int(m.Params.Threads),
                "x": None,
                "t": None,
                "wc_idx": None,
                "runtime_sec": float(m.Runtime),
            }

    def solve_recourse_mip(
        self,
        instance: Instance,
        scenario_index: int,
        x_star: np.ndarray,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Given fixed first-stage x*, solve the recourse MIP for one scenario."""

        A = instance.data["adjacency"]
        sce = instance.data["sce"]
        if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if A.ndim == 3: A = A[0]
        if sce.ndim > 2: sce = sce[0]

        N = A.shape[0]
        s = int(scenario_index)
        row_cost = sce[s]  # [N]
        x_star = np.asarray(x_star, dtype=float).reshape(-1)

        env = gp.Env(empty=True)
        if not verbose: env.setParam("OutputFlag", 0)
        env.setParam("Threads", threads)
        if time_limit is not None and time_limit > 0:
            env.setParam("TimeLimit", float(time_limit))
        if mip_gap is not None and mip_gap >= 0:
            env.setParam("MIPGap", float(mip_gap))
        env.start()

        m = gp.Model(env=env)
        y = m.addVars(N, vtype=GRB.BINARY, name="y")

        # y_j can only be 1 if x_j = 0
        for j in range(N):
            if x_star[j] > 0.5:
                y[j].UB = 0.0

        # Cover constraints on residual edges
        rows, cols = np.nonzero(np.triu(A, 1))
        for u, v in zip(rows, cols):
            # x_u + y_u + x_v + y_v >= 1
            rhs = 1.0 - x_star[u] - x_star[v]
            if rhs > 0.01:  # edge not covered by x*
                m.addConstr(y[u] + y[v] >= rhs, name=f"cover_{u}_{v}")

        m.setObjective(
            gp.quicksum(float(row_cost[j]) * y[j] for j in range(N)),
            GRB.MINIMIZE,
        )
        m.optimize()

        status = m.Status
        has_incumbent = m.SolCount > 0
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and has_incumbent:
            return {"status": int(status), "has_incumbent": True, "obj": float(m.ObjVal)}
        return {"status": int(status), "has_incumbent": False, "obj": float("inf")}

    def solve_robust_ccg(
        self,
        instance: Instance,
        scenario_indices: Optional[List[int]] = None,
        time_limit: Optional[float] = None,
        recourse_time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
        max_iterations: int = 20,
        convergence_tolerance: float = 1e-4,
        duplicate_tolerance: float = 1e-6,
        initial_active_scenarios: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Solve robust vertex cover by column-and-constraint generation.

        ``scenario_indices`` restricts separation to a candidate subset;
        ``initial_active_scenarios`` only seeds the master, preserving that candidate set.
        Returns a solver result dict with the solution, objective and convergence details.
        """
        import time as _time

        A = instance.data["adjacency"]
        det = instance.data["det"]
        sce = instance.data["sce"]
        if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
        if isinstance(det, torch.Tensor): det = det.detach().cpu().numpy()
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if A.ndim == 3: A = A[0]
        if det.ndim > 1: det = det[0]
        if sce.ndim > 2: sce = sce[0]

        N = A.shape[0]
        S = sce.shape[0]

        if scenario_indices is None:
            candidate_indices = list(range(S))
        else:
            candidate_indices = sorted({int(s) for s in scenario_indices})

        if len(candidate_indices) == 0:
            return {
                "status": -1, "has_incumbent": False, "obj": None,
                "obj_bound": None, "mip_gap": None, "x": None, "t": None,
                "wc_idx": None, "runtime_sec": 0.0, "ccg_iterations": 0,
                "ccg_converged": False, "ccg_termination_reason": "empty_scenario_set",
            }

        recourse_limit = recourse_time_limit if recourse_time_limit is not None else time_limit

        # Warm start (GK-2025). Validate loudly: a silently dropped seed index
        # would show up as a smaller speedup, i.e. as a *result*, not an error.
        if initial_active_scenarios is None:
            active_scenarios: List[int] = [candidate_indices[0]]
        else:
            seeded = sorted({int(s) for s in initial_active_scenarios})
            out_of_range = [s for s in seeded if not (0 <= s < S)]
            if out_of_range:
                raise ValueError(
                    f"initial_active_scenarios out of range for S={S}: {out_of_range}"
                )
            not_candidates = [s for s in seeded if s not in set(candidate_indices)]
            if not_candidates:
                raise ValueError(
                    "initial_active_scenarios must be a subset of the candidate set; "
                    f"these are outside scenario_indices: {not_candidates}"
                )
            active_scenarios = seeded if seeded else [candidate_indices[0]]

        # Build incremental model seeded with the active set
        model = self.build_base_model(instance, time_limit=time_limit, threads=threads, verbose=verbose)
        if mip_gap is not None and mip_gap >= 0:
            model.Params.MIPGap = float(mip_gap)
        for _s0 in active_scenarios:
            self.add_scenario_to_model(model, instance, int(_s0))

        wall_start = _time.perf_counter()
        last_solution: Optional[Dict[str, Any]] = None
        converged = False
        termination_reason = "max_iterations_reached"

        for iteration in range(1, int(max_iterations) + 1):
            model.optimize()
            status = int(model.Status)
            has_incumbent = bool(model.SolCount > 0)
            if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or not has_incumbent:
                termination_reason = f"master_no_incumbent_status_{status}"
                break

            x_star = np.array([1.0 if model.getVarByName(f"x[{j}]").X > 0.5 else 0.0 for j in range(N)])
            theta = float(model.getVarByName("t").X)
            first_stage_cost = float(np.dot(det, x_star))

            # Evaluate recourse for all candidate scenarios
            worst_idx: Optional[int] = None
            worst_cost = float("-inf")
            rec_costs = np.full(S, np.nan, dtype=float)

            for s in candidate_indices:
                rec_out = self.solve_recourse_mip(
                    instance=instance, scenario_index=s, x_star=x_star,
                    time_limit=recourse_limit, mip_gap=mip_gap,
                    threads=threads, verbose=verbose,
                )
                rec_cost = float(rec_out["obj"]) if rec_out["has_incumbent"] else float("inf")
                rec_costs[s] = rec_cost
                if rec_cost > worst_cost + float(duplicate_tolerance):
                    worst_cost = rec_cost
                    worst_idx = int(s)

            if worst_idx is None:
                termination_reason = "no_worst_scenario_found"
                break

            upper_obj = first_stage_cost + worst_cost
            lower_bound = float(model.ObjVal)
            gap = abs(upper_obj - lower_bound) / abs(upper_obj) if np.isfinite(upper_obj) and upper_obj != 0.0 else None
            violation = worst_cost - theta

            last_solution = {
                "status": status, "has_incumbent": True,
                "obj": float(upper_obj), "obj_bound": float(lower_bound),
                "mip_gap": float(gap) if gap is not None else None,
                "x": x_star.tolist(), "t": theta,
                "wc_idx": int(worst_idx), "rec_costs": rec_costs,
                "ccg_iterations": int(iteration),
                "threads": int(model.Params.Threads),
                "ccg_active_scenarios": list(active_scenarios),
                "ccg_violation": float(violation),
            }

            if violation <= float(convergence_tolerance):
                converged = True
                termination_reason = "violation_within_tolerance"
                break
            if worst_idx in active_scenarios:
                termination_reason = "duplicate_worst_scenario"
                break

            active_scenarios.append(int(worst_idx))
            self.add_scenario_to_model(model, instance, int(worst_idx))

        runtime_sec = float(_time.perf_counter() - wall_start)
        if last_solution is None:
            return {
                "status": int(model.Status), "has_incumbent": False,
                "obj": None, "obj_bound": None, "mip_gap": None,
                "x": None, "t": None, "wc_idx": None,
                "runtime_sec": runtime_sec, "ccg_iterations": 0,
                "threads": int(model.Params.Threads),
                "ccg_converged": False, "ccg_termination_reason": termination_reason,
            }

        last_solution["runtime_sec"] = runtime_sec
        last_solution["ccg_converged"] = bool(converged)
        last_solution["ccg_termination_reason"] = str(termination_reason)
        return last_solution

    def solve_exact(self, instance: Instance, time_limit: float = None, threads: int = 1,
                    verbose: bool = False, backend: Optional[str] = None, mip_gap=None) -> Labels:
        backend_name = str(
            backend if backend is not None else getattr(self, "exact_backend", "bigmilp")
        ).lower()

        if backend_name == "bigmilp":
            out = self.solve_robust_bigmilp(
                instance,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                verbose=verbose,
            )
        elif backend_name == "ccg":
            out = self.solve_robust_ccg(
                instance, scenario_indices=None,
                time_limit=time_limit,
                recourse_time_limit=self.ccg_recourse_time_limit,
                mip_gap=self.ccg_mip_gap,
                threads=threads, verbose=verbose,
                max_iterations=self.ccg_max_iterations,
                convergence_tolerance=self.ccg_convergence_tolerance,
                duplicate_tolerance=self.ccg_duplicate_tolerance,
            )
        else:
            raise ValueError(f"Unsupported exact backend '{backend_name}'")

        x_star = np.asarray(out["x"], dtype=float) if out.get("x") is not None else None
        t_star = out.get("t", None)
        rec_costs = out.get("rec_costs", None)
        wc_idx = out.get("wc_idx", None)
        obj_val = float(out["obj"]) if out.get("obj") is not None else float("inf")
        status = out.get("status", None)

        sce_scores = np.asarray(rec_costs, dtype=float) if rec_costs is not None else None
        sce_target_idx = (
            np.asarray([wc_idx], dtype=int) if wc_idx is not None else None
        )

        labels = Labels(
            obj=obj_val,
            feasible=out.get("has_incumbent", False),
            x=x_star,
            sce_target_idx=sce_target_idx,
            sce_ranking=None,
            sce_scores=sce_scores,
            sce_top_idx=None,
            obj_bound=out.get("obj_bound"),
            mip_gap=out.get("mip_gap"),
            meta={
                "status": status,
                "t": t_star,
                "solver_backend": backend_name,
                "threads": out.get("threads"),
                "ccg_iterations": out.get("ccg_iterations"),
                "ccg_converged": out.get("ccg_converged"),
                "ccg_termination_reason": out.get("ccg_termination_reason"),
            },
        )
        return labels

    def decision_dim(self, instance: Instance) -> int:
        return instance.data["det"].shape[-1]

    def _to_tensor(self, v) -> torch.Tensor:
        """Convert numpy/array/scalar to torch.Tensor."""
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        return torch.tensor(v)


    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ----- Collate Labels (scenario fields) -----
        def _collate_optional_field(labels_list, field_name: str, expected_s: Optional[int] = None):
            vals = [getattr(lab, field_name, None) for lab in labels_list]
            all_none = all(v is None for v in vals)
            all_present = all(v is not None for v in vals)

            if all_none:
                return None
            if not all_present:
                raise ValueError(
                    f"Some labels have '{field_name}' while others don't."
                )

            tensors = []
            truncate_scenarios = field_name in {"sce_ranking", "sce_scores"}
            for idx, v in enumerate(vals):
                t = self._to_tensor(v)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                if truncate_scenarios and expected_s is not None and t.shape[0] != expected_s:
                    if t.shape[0] < expected_s:
                        logging.warning(
                            "label.%s shorter than expected S=%d at batch index %d: shape=%s, data=%s",
                            field_name,
                            expected_s,
                            idx,
                            tuple(t.shape),
                            t.detach().cpu().tolist(),
                        )
                        raise ValueError(
                            f"label.{field_name} shape {tuple(t.shape)} shorter than expected S={expected_s}."
                        )
                    # This truncation is expected for stable datasets; avoid noisy logs.
                    t = t[:expected_s]
                tensors.append(t)

            ref_shape = tensors[0].shape
            if not all(t.shape == ref_shape for t in tensors):
                raise ValueError(
                    f"Inconsistent shapes for label.{field_name}: "
                    f"{[tuple(t.shape) for t in tensors]}"
                )

            return torch.stack(tensors, dim=0)

        """
        Collate a list of {"instance": Instance, "label": Labels} into
        {"instance": Instance(batched_data), "label": Labels(batched_data)}.
        """
        # ----- Collate Instance -----
        first_inst = batch[0]["instance"]
        keys = first_inst.data.keys()
        batched_data: Dict[str, torch.Tensor] = {}

        for key in keys:
            tensors = []
            for item in batch:
                v = self._to_tensor(item["instance"].data[key])
                if v.ndim == 0:
                    v = v.unsqueeze(0)
                tensors.append(v)
            ref_shape = tensors[0].shape
            if not all(t.shape == ref_shape for t in tensors):
                raise ValueError(
                    f"Inconsistent shapes for instance key '{key}': "
                    f"{[tuple(t.shape) for t in tensors]}"
                )
            batched_data[key] = torch.stack(tensors, dim=0)  # [B, *ref_shape]

        batched_instance = Instance(data=batched_data)

        per_instances = [item["instance"] for item in batch]

        # infer S from the first per-instance "sce"
        sce0 = per_instances[0].data["sce"]
        sce0 = torch.as_tensor(sce0)
        S = int(sce0.shape[0]) if sce0.ndim == 2 else int(sce0.shape[0])  # typical unbatched: [S, ...]
        # (if your per-instance 'sce' is [S, N] this is correct)

        gpack = build_batched_scenario_graph(per_instances, problem=self, expected_S=S)

        # Attach tensors (CPU). unpack_batch() will move them to GPU once.
        batched_instance.data["scen_graph"] = {
            "x": gpack.x,
            "edge_index": gpack.edge_index,
            "edge_attr": gpack.edge_attr,
            "scen_batch": gpack.scen_batch,
            "inst_batch": gpack.inst_batch,
        }

        dpack = build_batched_det_graph(per_instances, problem=self)

        batched_instance.data["det_graph"] = {
            "x": dpack.x,
            "edge_index": dpack.edge_index,
            "edge_attr": dpack.edge_attr,
            "batch": dpack.batch,
        }

        labels_list = [item["label"] for item in batch]
        obj_b = _collate_optional_field(labels_list=labels_list, field_name="obj")  # can be None
        feas_b = _collate_optional_field(labels_list=labels_list, field_name="feasible")  # can be None
        x_b = _collate_optional_field(labels_list=labels_list, field_name="x")  # can be None
        sce_target_idx_b = _collate_optional_field(labels_list=labels_list, field_name="sce_target_idx")  # e.g. [B, 1] or [B, …]
        sce_ranking_b = _collate_optional_field(labels_list=labels_list, field_name="sce_ranking", expected_s=S)  # e.g. [B, S]
        sce_scores_b = _collate_optional_field(labels_list=labels_list, field_name="sce_scores", expected_s=S)  # e.g. [B, S]
        sce_top_idx_b = _collate_optional_field(labels_list=labels_list, field_name="sce_top_idx")  # e.g. [B, K]
        fwd_trace_b = _collate_forward_trace(labels_list, to_tensor_fn=self._to_tensor)

        # ----- Build batched Labels -----
        batched_labels = Labels(
            obj=obj_b,
            feasible=feas_b,
            x=x_b,
            sce_target_idx=sce_target_idx_b,
            sce_ranking=sce_ranking_b,
            sce_scores=sce_scores_b,
            sce_top_idx=sce_top_idx_b,
            # meta is left out here; you can add a list of metas if you want later
        )
        batched_labels.forward_trace = fwd_trace_b

        return {
            "instance": batched_instance,
            "label": batched_labels,
        }

    def get_scenarios(self, data_np: Dict[str, Any]) -> np.ndarray:
        if "sce" in data_np:
            return np.asarray(data_np["sce"])
        if "scenarios" in data_np:
            return np.asarray(data_np["scenarios"])
        raise KeyError("VertexCoverProblem expects scenario key 'sce' or 'scenarios'.")


    def get_data_keys(self) -> List[str]:
        """Return expected data keys for this problem."""
        return ['det', 'sce', 'adjacency', 'n_nodes', 'n_scenarios']


    def to_pyg_data(self, instance: Instance) -> List[Data]:
        """
        Cached wrapper around _build_pyg_data().

        This should be safe to call from hot paths: after the first call per Instance,
        it returns the cached graphs without rebuilding.
        """
        cached = instance.cache_get(self._PYG_CACHE_KEY, None)
        if cached is not None:
            return cached

        built = self._build_pyg_data(instance)
        instance.cache_set(self._PYG_CACHE_KEY, built)
        return built

    def _compute_cosine_sim(self, row_coeffs: dict, full_obj_vec: np.ndarray) -> float:
        """
        Computes cosine similarity between a sparse constraint row and the dense objective vector.
        row_coeffs: dict {var_idx: coeff_value}
        full_obj_vec: np.ndarray [n_vars]
        """
        # 1. Sparse Dot Product
        dot_prod = 0.0
        row_norm_sq = 0.0

        for idx, val in row_coeffs.items():
            dot_prod += val * full_obj_vec[idx]
            row_norm_sq += val ** 2

        row_norm = np.sqrt(row_norm_sq)
        obj_norm = np.linalg.norm(full_obj_vec)

        if row_norm < 1e-9 or obj_norm < 1e-9:
            return 0.0

        return float(dot_prod / (row_norm * obj_norm))

    def _edge_attr_dim(self) -> int:
        return 2 if self.edge_attr_include_abs else 1

    def _get_edge_features(self, coeff: float) -> List[float]:
        c = float(coeff)
        if self.edge_attr_include_abs:
            return [c, abs(c)]
        return [c]

    def _build_pyg_data(self, instance: Instance) -> List[Data]:
        A = instance.data["adjacency"]
        det = instance.data["det"]
        sce = instance.data["sce"]

        # Normalize inputs
        if isinstance(A, torch.Tensor):
            A_np = A.detach().cpu().numpy()
        else:
            A_np = np.asarray(A)
        if A_np.ndim == 3: A_np = A_np[0]

        if isinstance(det, torch.Tensor):
            det_np = det.detach().cpu().numpy()
        else:
            det_np = np.asarray(det)
        if det_np.ndim > 1: det_np = det_np[0]

        if isinstance(sce, torch.Tensor):
            sce_np = sce.detach().cpu().numpy()
        else:
            sce_np = np.asarray(sce)
        if sce_np.ndim > 2: sce_np = sce_np[0]

        N = A_np.shape[0]
        S = sce_np.shape[0]

        # Edges for cover constraints
        edges_uv: List[Tuple[int, int]] = []
        for u in range(N):
            for v in range(u + 1, N):
                if A_np[u, v] != 0:
                    edges_uv.append((u, v))
        M = len(edges_uv)

        # Pre-compute degrees (Structural Feature)
        # We use simple graph degree here, but bipartite degree is calculated implicitly by GNN usually
        deg_np = A_np.sum(axis=1)

        data_list: List[Data] = []

        edge_attr_dim = self._edge_attr_dim()

        for i in range(S):
            sce_i_np = sce_np[i]  # [N]

            # --- 1. Define Variable layout ---
            # x_j: 0..N-1
            # y_j: N..2N-1
            # t:   2N
            n_vars = 2 * N + 1
            n_constraints = M + N + 1
            num_nodes = n_vars + n_constraints

            # --- 2. Construct Objective Vector for Cosine Sim ---
            # Obj = [det_0..det_N, 0..0, 1] (y has 0 formal cost in objective, t has 1)
            # Note: budget constraint handles the scenario cost, but formal objective is c*x + t
            full_obj_vec = np.zeros(n_vars)
            full_obj_vec[0:N] = det_np
            full_obj_vec[2 * N] = 1.0

            # --- 3. Build Node Features ---
            # Dims: [kind, is_x, is_y, is_t, obj, hint, rhs, cos, deg]
            x_feat = torch.zeros(num_nodes, 9, dtype=torch.float32)

            # Variable Nodes
            # x_j
            x_feat[0:N, 0] = 0.0  # kind=var
            x_feat[0:N, 1] = 1.0  # is_x
            x_feat[0:N, 4] = torch.from_numpy(det_np).float()  # obj
            x_feat[0:N, 8] = torch.from_numpy(deg_np).float()  # degree

            # y_j
            y_start = N
            x_feat[y_start:2 * N, 0] = 0.0  # kind=var
            x_feat[y_start:2 * N, 2] = 1.0  # is_y
            x_feat[y_start:2 * N, 5] = torch.from_numpy(sce_i_np).float()  # HINT: Scenario Cost
            x_feat[y_start:2 * N, 8] = torch.from_numpy(deg_np).float()  # degree

            # t
            t_idx = 2 * N
            x_feat[t_idx, 0] = 0.0
            x_feat[t_idx, 3] = 1.0  # is_t
            x_feat[t_idx, 4] = 1.0  # obj coeff

            # Constraint Nodes
            constr_start = n_vars
            x_feat[constr_start:, 0] = 1.0  # kind=constr

            # RHS Values
            # Cover (M): -1
            if M > 0: x_feat[constr_start:constr_start + M, 6] = -1.0
            # Compat (N): 1
            if N > 0: x_feat[constr_start + M:constr_start + M + N, 6] = 1.0
            # Budget (1): 0
            x_feat[constr_start + M + N, 6] = 0.0

            # --- 4. Build Edges & Calculate Cosine Sim ---
            edge_indices_list = []
            edge_attrs_list = []

            # Helper to add bidirectional edge
            def add_edge(src, dst, val):
                # src -> dst
                edge_indices_list.append([src, dst])
                edge_attrs_list.append(self._get_edge_features(val))
                # dst -> src
                edge_indices_list.append([dst, src])
                edge_attrs_list.append(self._get_edge_features(val))

            # A. Cover Constraints: -x_u - y_u - x_v - y_v <= -1
            for k, (u, v) in enumerate(edges_uv):
                c_idx = constr_start + k
                # Row dictionary for cos sim
                row_dict = {u: -1.0, y_start + u: -1.0, v: -1.0, y_start + v: -1.0}

                # Add Edges
                for var_idx, coeff in row_dict.items():
                    add_edge(var_idx, c_idx, coeff)

                # Cosine Sim
                x_feat[c_idx, 7] = self._compute_cosine_sim(row_dict, full_obj_vec)

            # B. Compatibility: x_j + y_j <= 1
            for j in range(N):
                c_idx = constr_start + M + j
                row_dict = {j: 1.0, y_start + j: 1.0}

                for var_idx, coeff in row_dict.items():
                    add_edge(var_idx, c_idx, coeff)

                x_feat[c_idx, 7] = self._compute_cosine_sim(row_dict, full_obj_vec)

            # C. Budget: sum(c * y) - t <= 0
            c_budget = constr_start + M + N
            row_dict_budget = {t_idx: -1.0}
            for j in range(N):
                cost_val = float(sce_i_np[j])
                if abs(cost_val) > 1e-9:  # Only add non-zero edges
                    row_dict_budget[y_start + j] = cost_val

            for var_idx, coeff in row_dict_budget.items():
                add_edge(var_idx, c_budget, coeff)

            x_feat[c_budget, 7] = self._compute_cosine_sim(row_dict_budget, full_obj_vec)

            # --- 5. Finalize Data Object ---
            if len(edge_indices_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float32)

            data_list.append(Data(
                x=x_feat,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_vars=torch.tensor([n_vars]),
                n_constraints=torch.tensor([n_constraints]),
                scenario_idx=torch.tensor([i]),
                det=torch.tensor(det_np).float(),
                sce=torch.tensor(sce_i_np).float(),
                constr_rhs=x_feat[constr_start:, 6].clone()
            ))

        return data_list

    def to_pyg_data_det(self, instance: Instance) -> Data:
        """
        Builds the Deterministic Master Graph.
        Structure: Same as Scenario Graph (Variables x, y, t + Constraints Cover, Compat).
        Crucial Difference:
          - No Budget Constraint (since it depends on scenario costs).
          - 'scen_cost' hint is 0.
        """
        A = instance.data["adjacency"]
        det = instance.data["det"]

        # Normalize
        if isinstance(A, torch.Tensor):
            A_np = A.detach().cpu().numpy()
        else:
            A_np = np.asarray(A)
        if A_np.ndim == 3: A_np = A_np[0]

        if isinstance(det, torch.Tensor):
            det_np = det.detach().cpu().numpy()
        else:
            det_np = np.asarray(det)
        if det_np.ndim > 1: det_np = det_np[0]

        N = A_np.shape[0]
        deg_np = A_np.sum(axis=1)

        edges_uv: List[Tuple[int, int]] = []
        for u in range(N):
            for v in range(u + 1, N):
                if A_np[u, v] != 0:
                    edges_uv.append((u, v))
        M = len(edges_uv)

        # In Deterministic Graph, we KEEP y and t variables to maintain
        # identical node structure for Attention keys, but we remove the scenario-dependent budget constraint.
        n_vars = 2 * N + 1
        n_constraints = M + N  # No budget constraint
        num_nodes = n_vars + n_constraints

        # Obj vector for Cos Sim (Deterministic)
        full_obj_vec = np.zeros(n_vars)
        full_obj_vec[0:N] = det_np
        full_obj_vec[2 * N] = 1.0

        x_feat = torch.zeros(num_nodes, 9, dtype=torch.float32)

        # -- Vars --
        # x
        x_feat[0:N, 0] = 0.0
        x_feat[0:N, 1] = 1.0
        x_feat[0:N, 4] = torch.from_numpy(det_np).float()
        x_feat[0:N, 8] = torch.from_numpy(deg_np).float()

        # y (Pure structural placeholders now)
        y_start = N
        x_feat[y_start:2 * N, 0] = 0.0
        x_feat[y_start:2 * N, 2] = 1.0
        x_feat[y_start:2 * N, 5] = 0.0  # No hint in deterministic graph
        x_feat[y_start:2 * N, 8] = torch.from_numpy(deg_np).float()

        # t
        t_idx = 2 * N
        x_feat[t_idx, 0] = 0.0
        x_feat[t_idx, 3] = 1.0
        x_feat[t_idx, 4] = 1.0

        # -- Constraints --
        constr_start = n_vars
        x_feat[constr_start:, 0] = 1.0
        if M > 0: x_feat[constr_start:constr_start + M, 6] = -1.0
        if N > 0: x_feat[constr_start + M:constr_start + M + N, 6] = 1.0

        edge_indices_list = []
        edge_attrs_list = []

        edge_attr_dim = self._edge_attr_dim()

        def add_edge(src, dst, val):
            edge_indices_list.append([src, dst])
            edge_attrs_list.append(self._get_edge_features(val))
            edge_indices_list.append([dst, src])
            edge_attrs_list.append(self._get_edge_features(val))

        # Cover
        for k, (u, v) in enumerate(edges_uv):
            c_idx = constr_start + k
            row_dict = {u: -1.0, y_start + u: -1.0, v: -1.0, y_start + v: -1.0}
            for var_idx, coeff in row_dict.items():
                add_edge(var_idx, c_idx, coeff)
            x_feat[c_idx, 7] = self._compute_cosine_sim(row_dict, full_obj_vec)

        # Compat
        for j in range(N):
            c_idx = constr_start + M + j
            row_dict = {j: 1.0, y_start + j: 1.0}
            for var_idx, coeff in row_dict.items():
                add_edge(var_idx, c_idx, coeff)
            x_feat[c_idx, 7] = self._compute_cosine_sim(row_dict, full_obj_vec)

        # No Budget Constraint in Deterministic Graph

        if len(edge_indices_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float32)

        return Data(
            x=x_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_vars=torch.tensor([n_vars]),
            n_constraints=torch.tensor([n_constraints]),
            det=torch.tensor(det_np).float(),
            constr_rhs=x_feat[constr_start:, 6].clone()
        )

    def build_base_model(
            self,
            instance: Instance,
            time_limit: Optional[float] = None,
            threads: int = 1,
            verbose: bool = False
    ) -> gp.Model:
        """
        Initializes the base 2RO model with first-stage variables (x) and the epigraph variable (t).
        No scenario constraints are added yet.
        """
        # Data extraction (same as solve_robust_bigmilp)
        det = instance.data["det"]
        if isinstance(det, torch.Tensor): det = det.detach().cpu().numpy()
        if det.ndim > 1: det = det[0]
        N = det.shape[0]

        env = gp.Env(empty=True)
        if not verbose: env.setParam("OutputFlag", 0)
        env.setParam("Threads", threads)
        if time_limit is not None and time_limit > 0:
            env.setParam("TimeLimit", float(time_limit))
        env.start()

        m = gp.Model(env=env)

        # 1. First-stage variables x
        x = m.addVars(N, vtype=GRB.BINARY, name="x")

        # 2. Epigraph variable t
        t = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t")

        # 3. Objective: c*x + t
        m.setObjective(
            gp.quicksum(float(det[j]) * x[j] for j in range(N)) + t,
            GRB.MINIMIZE,
        )

        # Store N for easy access later
        m._n_nodes = N
        m.update()
        return m


    def add_scenario_to_model(
            self,
            model: gp.Model,
            instance: Instance,
            sce_idx: int
    ):
        """
        Adds variables (y) and constraints for a SINGLE scenario `sce_idx` to the model.
        """
        # --- 1. Robust Data Extraction (Derive N here instead of model._n_nodes) ---
        sce = instance.data["sce"]
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if sce.ndim == 3: sce = sce[0]

        A = instance.data["adjacency"]
        if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
        if A.ndim == 3: A = A[0]

        # Derive N from the adjacency matrix shape (safe and consistent)
        N = A.shape[0]

        row_cost = sce[sce_idx]  # [N]

        # --- 2. Retrieve Variables ---
        # Note: Gurobi access by name is efficient enough for this loop
        x_vars = [model.getVarByName(f"x[{j}]") for j in range(N)]
        t_var = model.getVarByName("t")

        # --- 3. Add Recourse Variables y_{sce_idx} ---
        # Naming convention: "y_s{sce_idx}[node]"
        y = model.addVars(N, vtype=GRB.BINARY, name=f"y_s{sce_idx}")

        # --- 4. Recourse Cost Constraint: c_s * y_s <= t ---
        model.addConstr(
            gp.quicksum(float(row_cost[j]) * y[j] for j in range(N)) <= t_var,
            name=f"rec_cost_s{sce_idx}"
        )

        # --- 5. Cover Constraints: x_u + y_u + x_v + y_v >= 1 ---
        # Iterate over upper triangle of A to find edges
        rows, cols = np.nonzero(np.triu(A, 1))
        edges = zip(rows, cols)

        for u, v in edges:
            model.addConstr(
                x_vars[u] + y[u] + x_vars[v] + y[v] >= 1.0,
                name=f"cover_s{sce_idx}_{u}_{v}"
            )

        # --- 6. Compatibility: x_j + y_j <= 1 ---
        model.addConstrs(
            (x_vars[j] + y[j] <= 1.0 for j in range(N)),
            name=f"compat_s{sce_idx}"
        )

        model.update()
