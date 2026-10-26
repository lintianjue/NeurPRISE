import logging
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any, List, Optional

from torch_geometric.data import Data

from src.core.problem import IProblem
from src.core.types import Instance, Labels
from src.problems.scenario_generator import BaseScenarioGenerator, IidUniformScenarioGenerator
from src.utils.graph_batching import build_batched_scenario_graph, build_batched_det_graph
from src.utils.label_utils import _collate_forward_trace
from src.utils.seed import get_rng


class SelectionProblem(IProblem):
    """
    Two-Stage Robust Selection Problem (Cardinality Constrained).

    Formulation:
        min c*x + t
        s.t.
          sum(x_j + y_j^i) == p      ForAll i (Total items selected must be p)
          x_j + y_j^i <= 1           ForAll i, j (Item cannot be selected in both stages)
          sum(sce_cost_j^i * y_j^i) <= t   ForAll i (Epigraph constraint)
          x, y binary
    """
    _PYG_CACHE_KEY = "sel_pyg_data_list_v1"

    def __init__(
            self,
            n_nodes: int,
            n_scenarios: int,
            p_select: int,
            n_min_cost: int = 1,
            n_max_cost: int = 100,
            scenario_name: str = "uniform",
            scenario_generator: Optional[BaseScenarioGenerator] = None,
    ):
        self.n_nodes = n_nodes
        self.n_scenarios = n_scenarios
        self.p_select = p_select
        self.n_min_cost = n_min_cost
        self.n_max_cost = n_max_cost
        self.scenario_name = scenario_name
        self._rng = get_rng()

        if scenario_generator is None:
            self.scenario_generator = IidUniformScenarioGenerator(
                n_min_cost=n_min_cost,
                n_max_cost=n_max_cost,
                rng=self._rng,
            )
        else:
            self.scenario_generator = scenario_generator
        logging.info(f"Using {self.scenario_name} scenario generator. n_node: {n_nodes}, n_scenarios: {n_scenarios}, p_select: {p_select}.")

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
        # 1) Costs
        det_mat, sce_mat = self.scenario_generator.sample_costs(
            n_nodes=self.n_nodes,
            n_scenarios=self.n_scenarios,
        )

        # 2) Return Instance
        # Note: Selection problem doesn't have an adjacency matrix,
        # but we might keep an empty one if shared batching code expects keys.
        return Instance(
            {
                "n_nodes": np.array([self.n_nodes]),
                "n_scenarios": np.array([self.n_scenarios]),
                "p_select": np.array([self.p_select]),
                "det": det_mat,
                "sce": sce_mat,
                # Placeholder to satisfy get_data_keys if generic batching checks it
                # or strictly unnecessary if code handles missing adj.
                "adjacency": np.zeros((self.n_nodes, self.n_nodes), dtype=int),
            }
        )

    def solve_robust_bigmilp(
            self,
            instance: Instance,
            time_limit: Optional[float] = None,
            mip_gap: Optional[float] = None,
            threads: int = 1,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        #logging.info(f"Solving Selection Problem with {self.scenario_name} scenarios with threads {threads}.")
        c_det = instance.data["det"]  # (N,)
        sce = instance.data["sce"]  # (S, N)
        p = int(instance.data["p_select"][0]) if instance.data["p_select"].ndim > 0 else int(instance.data["p_select"])

        # Normalize shapes
        if c_det.ndim > 1: c_det = c_det[0]
        if sce.ndim > 2: sce = sce[0]

        N = c_det.shape[0]
        S = sce.shape[0]

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
        y = m.addVars(S, N, vtype=GRB.BINARY, name="y")
        t = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t")

        # 1. Budget/Epigraph: sum(c_s * y) <= t
        m.addConstrs(
            (
                gp.quicksum(float(sce[i, j]) * y[i, j] for j in range(N)) <= t
                for i in range(S)
            ),
            name="rec_cost",
        )

        # 2. Total Selection: sum(x) + sum(y_s) == p
        m.addConstrs(
            (
                gp.quicksum(x[j] + y[i, j] for j in range(N)) == p
                for i in range(S)
            ),
            name="selection_count"
        )

        # 3. Disjoint: x_j + y_j_s <= 1
        m.addConstrs(
            (
                x[j] + y[i, j] <= 1.0
                for i in range(S)
                for j in range(N)
            ),
            name="disjoint"
        )

        # Objective
        m.setObjective(
            gp.quicksum(float(c_det[j]) * x[j] for j in range(N)) + t,
            GRB.MINIMIZE,
        )

        m.optimize()

        status = m.Status
        has_incumbent = (m.SolCount > 0)

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and has_incumbent:
            x_star = [1.0 if x[j].X > 0.5 else 0.0 for j in range(N)]
            t_star = float(t.X)
            return {
                "status": int(status),
                "has_incumbent": True,
                "obj": float(m.ObjVal),
                "obj_bound": float(m.ObjBound),
                "mip_gap": float(m.MIPGap),
                "threads": int(m.Params.Threads),
                "x": x_star,
                "t": t_star,
                "runtime_sec": float(m.Runtime),
            }
        else:
            return {
                "status": int(status),
                "has_incumbent": False,
                "obj": None,
                "obj_bound": None,
                "mip_gap": None,
                "threads": int(m.Params.Threads),
                "x": None,
                "t": None,
                "runtime_sec": float(m.Runtime),
            }

    def solve_recourse_greedy(
        self,
        instance: Instance,
        scenario_index: int,
        x_star: np.ndarray,
    ) -> Dict[str, Any]:
        """Given fixed first-stage x*, solve the recourse for one scenario.

        For selection, recourse is trivial: pick the cheapest (p - sum(x*))
        items from those not already selected.
        """
        sce = instance.data["sce"]
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if sce.ndim > 2: sce = sce[0]

        p = int(instance.data["p_select"][0]) if np.asarray(instance.data["p_select"]).ndim > 0 else int(instance.data["p_select"])
        x_star = np.asarray(x_star, dtype=float).reshape(-1)
        N = x_star.shape[0]
        s = int(scenario_index)
        row_cost = sce[s]  # [N]

        # Items not selected in first stage
        unselected = [j for j in range(N) if x_star[j] < 0.5]
        need = p - int(round(x_star.sum()))

        if need <= 0:
            return {"has_incumbent": True, "obj": 0.0}
        if need > len(unselected):
            return {"has_incumbent": False, "obj": float("inf")}

        # Sort by scenario cost, pick cheapest
        costs = [(float(row_cost[j]), j) for j in unselected]
        costs.sort()
        rec_cost = sum(c for c, _ in costs[:need])
        return {"has_incumbent": True, "obj": float(rec_cost)}

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
    ) -> Dict[str, Any]:
        import time as _time

        det = instance.data["det"]
        sce = instance.data["sce"]
        if isinstance(det, torch.Tensor): det = det.detach().cpu().numpy()
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if det.ndim > 1: det = det[0]
        if sce.ndim > 2: sce = sce[0]

        N = det.shape[0]
        S = sce.shape[0]
        p = int(instance.data["p_select"][0]) if np.asarray(instance.data["p_select"]).ndim > 0 else int(instance.data["p_select"])

        if scenario_indices is None:
            candidate_indices = list(range(S))
        else:
            candidate_indices = sorted({int(s) for s in scenario_indices})

        if len(candidate_indices) == 0:
            return {
                "status": -1, "has_incumbent": False, "obj": None,
                "obj_bound": None, "mip_gap": None, "x": None, "t": None,
                "runtime_sec": 0.0, "ccg_iterations": 0,
                "ccg_converged": False, "ccg_termination_reason": "empty_scenario_set",
            }

        active_scenarios: List[int] = [candidate_indices[0]]

        model = self.build_base_model(instance, time_limit=time_limit, threads=threads, verbose=verbose)
        if mip_gap is not None and mip_gap >= 0:
            model.Params.MIPGap = float(mip_gap)
        self.add_scenario_to_model(model, instance, active_scenarios[0])

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

            worst_idx: Optional[int] = None
            worst_cost = float("-inf")
            rec_costs = np.full(S, np.nan, dtype=float)

            for s in candidate_indices:
                rec_out = self.solve_recourse_greedy(
                    instance=instance, scenario_index=s, x_star=x_star,
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
                "x": None, "t": None,
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
                instance, time_limit=time_limit, threads=threads, verbose=verbose, mip_gap=mip_gap,
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

        if out.get("has_incumbent"):
            obj_val = float(out["obj"])
            x_star = np.asarray(out["x"], dtype=float)
            t_star = out.get("t")
        else:
            obj_val = float('inf')
            x_star = None
            t_star = None

        labels = Labels(
            obj=obj_val,
            feasible=out.get("has_incumbent", False),
            x=x_star,
            sce_target_idx=None,
            sce_ranking=None,
            sce_scores=None,
            sce_top_idx=None,
            obj_bound=out.get("obj_bound"),
            mip_gap=out.get("mip_gap"),
            meta={
                "status": out.get("status"),
                "t": t_star,
                "solver_backend": backend_name,
                "threads": out.get("threads"),
                "ccg_iterations": out.get("ccg_iterations"),
                "ccg_converged": out.get("ccg_converged"),
                "ccg_termination_reason": out.get("ccg_termination_reason"),
            },
        )
        return labels

    # --- Incremental Solver Support ---

    def build_base_model(
            self,
            instance: Instance,
            time_limit: Optional[float] = None,
            threads: int = 1,
            verbose: bool = False
    ) -> gp.Model:
        """
        Base model with x and t.
        """
        det = instance.data["det"]
        if isinstance(det, torch.Tensor): det = det.detach().cpu().numpy()
        if det.ndim > 1: det = det[0]

        N = det.shape[0]
        p = int(instance.data["p_select"][0]) if instance.data["p_select"].ndim > 0 else int(instance.data["p_select"])

        env = gp.Env(empty=True)
        if not verbose: env.setParam("OutputFlag", 0)
        env.setParam("Threads", threads)
        if time_limit is not None and time_limit > 0:
            env.setParam("TimeLimit", float(time_limit))
        env.start()

        m = gp.Model(env=env)

        x = m.addVars(N, vtype=GRB.BINARY, name="x")
        t = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t")

        # Enforce the first-stage cardinality bound before adding scenario constraints.
        m.addConstr(gp.quicksum(x[j] for j in range(N)) <= p, name="base_x_limit")

        m.setObjective(
            gp.quicksum(float(det[j]) * x[j] for j in range(N)) + t,
            GRB.MINIMIZE,
        )

        m._n_nodes = N
        m._p_select = p

        m._copy_payload = {
            "_n_nodes": N,
            "_p_select": p,
        }

        m.update()
        return m

    def add_scenario_to_model(self, model: gp.Model, instance: Instance, sce_idx: int):
        sce = instance.data["sce"]
        if isinstance(sce, torch.Tensor): sce = sce.detach().cpu().numpy()
        if sce.ndim == 3: sce = sce[0]

        N = model._n_nodes
        p = model._p_select
        row_cost = sce[sce_idx]

        x_vars = [model.getVarByName(f"x[{j}]") for j in range(N)]
        t_var = model.getVarByName("t")

        # Recourse vars
        y = model.addVars(N, vtype=GRB.BINARY, name=f"y_s{sce_idx}")

        # 1. Recourse Cost: c_s * y <= t
        model.addConstr(
            gp.quicksum(float(row_cost[j]) * y[j] for j in range(N)) <= t_var,
            name=f"rec_cost_s{sce_idx}"
        )

        # 2. Total Selection: sum(x) + sum(y) == p
        model.addConstr(
            gp.quicksum(x_vars[j] + y[j] for j in range(N)) == p,
            name=f"select_count_s{sce_idx}"
        )

        # 3. Disjoint: x[j] + y[j] <= 1
        model.addConstrs(
            (x_vars[j] + y[j] <= 1.0 for j in range(N)),
            name=f"disjoint_s{sce_idx}"
        )

        model.update()

    # --- PyG Graph Construction ---

    def to_pyg_data(self, instance: Instance) -> List[Data]:
        cached = instance.cache_get(self._PYG_CACHE_KEY, None)
        if cached is not None: return cached

        built = self._build_pyg_data(instance)
        instance.cache_set(self._PYG_CACHE_KEY, built)
        return built

    def _build_pyg_data(self, instance: Instance) -> List[Data]:
        det = instance.data["det"]
        sce = instance.data["sce"]
        p = int(instance.data["p_select"][0]) if instance.data["p_select"].ndim > 0 else int(instance.data["p_select"])

        # Normalize
        if isinstance(det, torch.Tensor):
            det = det.detach().cpu().numpy()
        else:
            det = np.asarray(det)
        if det.ndim > 1: det = det[0]

        if isinstance(sce, torch.Tensor):
            sce = sce.detach().cpu().numpy()
        else:
            sce = np.asarray(sce)
        if sce.ndim > 2: sce = sce[0]

        N = det.shape[0]
        S = sce.shape[0]

        data_list = []

        # Node Features Layout (8 dims):
        # [kind, is_x, is_y, is_t, obj_coeff, hint_cost, rhs, cos_sim]
        # Total Vars: 2N + 1
        # Total Constr: 1 + N + 1 = N + 2

        n_vars = 2 * N + 1
        n_constrs = N + 2
        num_nodes = n_vars + n_constrs

        # Obj Vector for Cosine Sim
        full_obj_vec = np.zeros(n_vars)
        full_obj_vec[0:N] = det
        full_obj_vec[2 * N] = 1.0

        for i in range(S):
            sce_i = sce[i]  # [N]

            x_feat = torch.zeros(num_nodes, 8, dtype=torch.float32)

            # --- Variables ---
            # x
            x_feat[0:N, 0] = 0.0  # Kind=Var
            x_feat[0:N, 1] = 1.0  # is_x
            x_feat[0:N, 4] = torch.from_numpy(det).float()

            # y
            y_start = N
            x_feat[y_start:2 * N, 0] = 0.0
            x_feat[y_start:2 * N, 2] = 1.0  # is_y
            x_feat[y_start:2 * N, 5] = torch.from_numpy(sce_i).float()  # Hint cost

            # t
            t_idx = 2 * N
            x_feat[t_idx, 0] = 0.0
            x_feat[t_idx, 3] = 1.0  # is_t
            x_feat[t_idx, 4] = 1.0

            # --- Constraints ---
            c_start = n_vars
            x_feat[c_start:, 0] = 1.0  # Kind=Constr

            # C1: Selection sum(x)+sum(y) == p
            idx_c_sel = c_start
            x_feat[idx_c_sel, 6] = float(p)

            # C2: Disjoint x + y <= 1 => -x -y >= -1. RHS = -1
            idx_c_dis_start = c_start + 1
            x_feat[idx_c_dis_start: idx_c_dis_start + N, 6] = -1.0

            # C3: Budget sum(c*y) - t <= 0 => -sum(c*y) + t >= 0. RHS = 0
            idx_c_bud = c_start + 1 + N
            x_feat[idx_c_bud, 6] = 0.0

            # --- Edges ---
            edge_indices = []
            edge_attrs = []

            def add_edge(u, v, w):
                feat = self._get_edge_features(w)
                edge_indices.append([u, v])
                edge_attrs.append(feat)
                edge_indices.append([v, u])
                edge_attrs.append(feat)

            # Selection constraint
            row_sel = {}
            for j in range(N):
                add_edge(j, idx_c_sel, 1.0)  # x_j
                add_edge(y_start + j, idx_c_sel, 1.0)  # y_j
                row_sel[j] = 1.0
                row_sel[y_start + j] = 1.0
            x_feat[idx_c_sel, 7] = self._compute_cosine_sim(row_sel, full_obj_vec)

            # Disjoint constraints
            for j in range(N):
                c_idx = idx_c_dis_start + j
                add_edge(j, c_idx, -1.0)
                add_edge(y_start + j, c_idx, -1.0)

                row_dis = {j: -1.0, y_start + j: -1.0}
                x_feat[c_idx, 7] = self._compute_cosine_sim(row_dis, full_obj_vec)

            # Budget constraint
            row_bud = {t_idx: 1.0}
            add_edge(t_idx, idx_c_bud, 1.0)
            for j in range(N):
                cost = -float(sce_i[j])
                if abs(cost) > 1e-9:
                    add_edge(y_start + j, idx_c_bud, cost)
                    row_bud[y_start + j] = cost
            x_feat[idx_c_bud, 7] = self._compute_cosine_sim(row_bud, full_obj_vec)

            # Tensorify (edge_attr is 1-dim)
            if len(edge_indices) > 0:
                ei = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                ea = torch.tensor(edge_attrs, dtype=torch.float32)
            else:
                ei = torch.empty((2, 0), dtype=torch.long)
                ea = torch.empty((0, 1), dtype=torch.float32)

            data_list.append(Data(
                x=x_feat,
                edge_index=ei,
                edge_attr=ea,
                n_vars=torch.tensor([n_vars]),
                n_constraints=torch.tensor([n_constrs]),
                scenario_idx=torch.tensor([i]),
                det=torch.tensor(det).float(),
                sce=torch.tensor(sce_i).float(),
            ))

        return data_list

    def to_pyg_data_det(self, instance: Instance) -> Data:
        """
        Deterministic master graph for Selection:
          - same variable layout as scenario graphs (x, y, t)
          - includes scenario-invariant constraints: selection_count + disjoint
          - excludes scenario-dependent budget constraint
          - y hint_cost is 0 (unknown scenario)
        """
        det = instance.data["det"]
        p = int(instance.data["p_select"][0]) if instance.data["p_select"].ndim > 0 else int(instance.data["p_select"])

        # Normalize
        if isinstance(det, torch.Tensor):
            det = det.detach().cpu().numpy()
        else:
            det = np.asarray(det)
        if det.ndim > 1:
            det = det[0]

        N = det.shape[0]

        n_vars = 2 * N + 1
        n_constrs = 1 + N  # selection_count + disjoint (no budget)
        num_nodes = n_vars + n_constrs

        # Obj vector for Cosine Sim
        full_obj_vec = np.zeros(n_vars)
        full_obj_vec[0:N] = det
        full_obj_vec[2 * N] = 1.0

        x_feat = torch.zeros(num_nodes, 8, dtype=torch.float32)

        # --- Variables ---
        # x
        x_feat[0:N, 0] = 0.0
        x_feat[0:N, 1] = 1.0
        x_feat[0:N, 4] = torch.from_numpy(det).float()

        # y (placeholders, no hint)
        y_start = N
        x_feat[y_start:2 * N, 0] = 0.0
        x_feat[y_start:2 * N, 2] = 1.0
        x_feat[y_start:2 * N, 5] = 0.0

        # t
        t_idx = 2 * N
        x_feat[t_idx, 0] = 0.0
        x_feat[t_idx, 3] = 1.0
        x_feat[t_idx, 4] = 1.0

        # --- Constraints ---
        c_start = n_vars
        x_feat[c_start:, 0] = 1.0

        # Selection: RHS = p
        idx_c_sel = c_start
        x_feat[idx_c_sel, 6] = float(p)

        # Disjoint: RHS = -1
        idx_c_dis_start = c_start + 1
        x_feat[idx_c_dis_start: idx_c_dis_start + N, 6] = -1.0

        # --- Edges (1-dim edge_attr) ---
        edge_indices: List[List[int]] = []
        edge_attrs: List[List[float]] = []

        def add_edge(u: int, v: int, w: float) -> None:
            feat = self._get_edge_features(w)
            edge_indices.append([u, v])
            edge_attrs.append(feat)
            edge_indices.append([v, u])
            edge_attrs.append(feat)

        # Selection constraint
        row_sel: Dict[int, float] = {}
        for j in range(N):
            add_edge(j, idx_c_sel, 1.0)
            add_edge(y_start + j, idx_c_sel, 1.0)
            row_sel[j] = 1.0
            row_sel[y_start + j] = 1.0
        x_feat[idx_c_sel, 7] = self._compute_cosine_sim(row_sel, full_obj_vec)

        # Disjoint constraints
        for j in range(N):
            c_idx = idx_c_dis_start + j
            add_edge(j, c_idx, -1.0)
            add_edge(y_start + j, c_idx, -1.0)
            row_dis = {j: -1.0, y_start + j: -1.0}
            x_feat[c_idx, 7] = self._compute_cosine_sim(row_dis, full_obj_vec)

        if len(edge_indices) > 0:
            ei = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            ea = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            ea = torch.empty((0, 1), dtype=torch.float32)

        return Data(
            x=x_feat,
            edge_index=ei,
            edge_attr=ea,
            n_vars=torch.tensor([n_vars]),
            n_constraints=torch.tensor([n_constrs]),
            det=torch.tensor(det).float(),
        )

    def decision_dim(self, instance: Instance) -> int:
        return int(instance.data["det"].shape[-1])

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ----- Collate Labels (scenario fields) -----


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
            batched_data[key] = torch.stack(tensors, dim=0)

        batched_instance = Instance(data=batched_data)

        per_instances = [item["instance"] for item in batch]

        # infer S from the first per-instance "sce" (unbatched: [S, N])
        sce0 = torch.as_tensor(per_instances[0].data["sce"])
        S = int(sce0.shape[0])
        def _collate_optional_field(labels_list, field_name: str):
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
            for v in vals:
                t = self._to_tensor(v)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                if field_name == "sce_ranking":
                    t = t[:S]
                tensors.append(t)

            ref_shape = tensors[0].shape
            if not all(t.shape == ref_shape for t in tensors):
                raise ValueError(
                    f"Inconsistent shapes for label.{field_name}: "
                    f"{[tuple(t.shape) for t in tensors]}"
                )

            return torch.stack(tensors, dim=0)
        gpack = build_batched_scenario_graph(per_instances, problem=self, expected_S=S)
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
        obj_b = _collate_optional_field(labels_list=labels_list, field_name="obj")
        feas_b = _collate_optional_field(labels_list=labels_list, field_name="feasible")
        x_b = _collate_optional_field(labels_list=labels_list, field_name="x")
        sce_target_idx_b = _collate_optional_field(labels_list=labels_list, field_name="sce_target_idx")
        sce_ranking_b = _collate_optional_field(labels_list=labels_list, field_name="sce_ranking")
        sce_scores_b = _collate_optional_field(labels_list=labels_list, field_name="sce_scores")
        sce_top_idx_b = _collate_optional_field(labels_list=labels_list, field_name="sce_top_idx")
        fwd_trace_b = _collate_forward_trace(labels_list, to_tensor_fn=self._to_tensor)

        batched_labels = Labels(
            obj=obj_b,
            feasible=feas_b,
            x=x_b,
            sce_target_idx=sce_target_idx_b,
            sce_ranking=sce_ranking_b,
            sce_scores=sce_scores_b,
            sce_top_idx=sce_top_idx_b,
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
        raise KeyError("SelectionProblem expects scenario key 'sce' or 'scenarios'.")

    def _to_tensor(self, v):
        if isinstance(v, torch.Tensor): return v
        if isinstance(v, np.ndarray): return torch.from_numpy(v)
        return torch.tensor(v)

    def _get_edge_features(self, coeff: float) -> List[float]:
        return [float(coeff)]

    def _compute_cosine_sim(self, row_coeffs: dict, full_obj_vec: np.ndarray) -> float:
        dot = 0.0
        norm_sq = 0.0
        for idx, val in row_coeffs.items():
            dot += val * full_obj_vec[idx]
            norm_sq += val ** 2
        if norm_sq < 1e-9: return 0.0
        obj_norm = np.linalg.norm(full_obj_vec)
        if obj_norm < 1e-9: return 0.0
        return float(dot / (np.sqrt(norm_sq) * obj_norm))

    def get_data_keys(self) -> List[str]:
        return ['det', 'sce', 'n_nodes', 'n_scenarios', 'p_select']
