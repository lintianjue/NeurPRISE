import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch_geometric.data import Data

from src.core.problem import IProblem
from src.core.types import Instance, Labels
from src.problems.scenario_generator import BaseScenarioGenerator
from src.utils.graph_batching import build_batched_det_graph, build_batched_scenario_graph
from src.utils.label_utils import _collate_forward_trace
from src.utils.seed import get_rng


class CFLPProblem(IProblem):
    """Model two-stage capacitated facility location with discrete demand scenarios.

    Choose binary openings y and capacity z (0 <= z_i <= K_i y_i), minimizing
    opening and capacity costs plus worst-case shipping cost. Each scenario
    ships nonnegative x_ij within capacity and meets every customer demand.
    Unlike the budgeted uncertainty in Zeng & Zhao (2013), scenarios are explicit.
    """

    _PYG_CACHE_KEY = "cflp_pyg_data_list_v2"

    def __init__(
        self,
        n_facilities: int,
        n_customers: int,
        n_scenarios: int,
        f_min: float = 100.0,
        f_max: float = 1000.0,
        a_min: float = 10.0,
        a_max: float = 100.0,
        K_min: float = 200.0,
        K_max: float = 700.0,
        c_min: float = 1.0,     #c_ij: transportation cost (deterministic across scenarios)
        c_max: float = 1000.0,
        d_min: float = 10.0,    # d_j^s: demand of customer j in scenario s
        d_max: float = 500.0,
        scenario_name: str = "uniform",
        scenario_generator: Optional[BaseScenarioGenerator] = None,
    ) -> None:
        self.n_facilities = int(n_facilities)
        self.n_customers = int(n_customers)
        self.n_scenarios = int(n_scenarios)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.K_min = float(K_min)
        self.K_max = float(K_max)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.scenario_name = scenario_name
        self.scenario_generator = scenario_generator
        self._rng = get_rng()
        # Default exact solver backend remains BigMILP for backward compatibility.
        self.exact_backend = "bigmilp"
        self.eval_full_obj_source = "label"
        # CCG defaults are configurable from eval scripts.
        self.ccg_max_iterations = 20
        self.ccg_convergence_tolerance = 1e-4
        self.ccg_duplicate_tolerance = 1e-6
        self.ccg_recourse_time_limit: Optional[float] = None
        self.ccg_mip_gap: Optional[float] = None

        logging.info(
            "Using %s location-transportation generator. I=%d, J=%d, S=%d. custom_generator=%s",
            self.scenario_name,
            self.n_facilities,
            self.n_customers,
            self.n_scenarios,
            "yes" if self.scenario_generator is not None else "no",
        )

    def sample(self) -> Instance:
        f = self._rng.uniform(self.f_min, self.f_max, size=(self.n_facilities,)).astype(np.float32)
        a = self._rng.uniform(self.a_min, self.a_max, size=(self.n_facilities,)).astype(np.float32)
        c = self._rng.uniform(self.c_min, self.c_max, size=(self.n_facilities, self.n_customers)).astype(np.float32)
        K, demand = self._sample_capacities_and_demands()

        return Instance(
            {
                "n_facilities": np.array([self.n_facilities], dtype=np.int64),
                "n_customers": np.array([self.n_customers], dtype=np.int64),
                "n_scenarios": np.array([self.n_scenarios], dtype=np.int64),
                "f": f,
                "a": a,
                "K": K,
                "c": c,
                "d": demand,
            }
        )

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

    def solve_robust_bigmilp(
        self,
        instance: Instance,
        scenario_indices: Optional[Sequence[int]] = None,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        data = self._extract_instance_data(instance)
        I, J, S = data["I"], data["J"], data["S"]

        if scenario_indices is None:
            scenario_indices = list(range(S))
        else:
            scenario_indices = [int(s) for s in scenario_indices]

        env = self._make_env(time_limit=time_limit, threads=threads, verbose=verbose, mip_gap=mip_gap)
        model = self._build_base_model_with_env(instance, env=env)
        for s in scenario_indices:
            self.add_scenario_to_model(model, instance, s)

        model.optimize()

        status = model.Status
        has_incumbent = model.SolCount > 0

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and has_incumbent:
            y = np.zeros(I, dtype=float)
            z = np.zeros(I, dtype=float)
            for i in range(I):
                y[i] = 1.0 if model.getVarByName(f"y[{i}]").X > 0.5 else 0.0
                z[i] = float(model.getVarByName(f"z[{i}]").X)

            theta = float(model.getVarByName("theta").X)
            rec_costs = np.full(S, np.nan, dtype=float)
            for s in scenario_indices:
                rec_costs[s] = self._scenario_recourse_from_model(
                    model=model,
                    instance=instance,
                    scenario_index=s,
                    I=I,
                    J=J,
                )

            wc_idx = int(np.nanargmax(rec_costs)) if np.any(~np.isnan(rec_costs)) else None
            return {
                "status": int(status),
                "has_incumbent": True,
                "obj": float(model.ObjVal),
                "obj_bound": float(model.ObjBound),
                "mip_gap": float(model.MIPGap),
                "threads": int(model.Params.Threads),
                "y": y,
                "z": z,
                "theta": theta,
                "recourse_costs": rec_costs,
                "wc_idx": wc_idx,
                "runtime_sec": float(model.Runtime),
            }

        return {
            "status": int(status),
            "has_incumbent": False,
            "obj": None,
            "obj_bound": None,
            "mip_gap": None,
            "threads": int(model.Params.Threads),
            "y": None,
            "z": None,
            "theta": None,
            "recourse_costs": None,
            "wc_idx": None,
            "runtime_sec": float(model.Runtime),
        }

    def solve_robust_ccg(
        self,
        instance: Instance,
        scenario_indices: Optional[Sequence[int]] = None,
        time_limit: Optional[float] = None,
        recourse_time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
        max_iterations: int = 20,
        convergence_tolerance: float = 1e-4,
        duplicate_tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        data = self._extract_instance_data(instance)
        f, a = data["f"], data["a"]
        I, J, S = data["I"], data["J"], data["S"]

        if scenario_indices is None:
            candidate_indices = list(range(S))
        else:
            candidate_indices = sorted({int(s) for s in scenario_indices})
        for s in candidate_indices:
            if s < 0 or s >= S:
                raise ValueError(f"scenario index out of range: {s} not in [0, {S}).")
        if len(candidate_indices) == 0:
            return {
                "status": -1,
                "has_incumbent": False,
                "obj": None,
                "obj_bound": None,
                "mip_gap": None,
                "y": None,
                "z": None,
                "theta": None,
                "recourse_costs": None,
                "wc_idx": None,
                "runtime_sec": 0.0,
                "ccg_iterations": 0,
                "ccg_converged": False,
                "ccg_termination_reason": "empty_scenario_set",
                "ccg_active_scenarios": [],
                "ccg_violation": None,
            }

        recourse_limit = recourse_time_limit if recourse_time_limit is not None else time_limit
        active_scenarios: List[int] = [candidate_indices[0]]

        env = self._make_env(time_limit=time_limit, threads=threads, verbose=verbose, mip_gap=mip_gap)
        model = self._build_base_model_with_env(instance, env=env)
        self.add_scenario_to_model(model, instance, active_scenarios[0])

        wall_start = time.perf_counter()
        last_solution: Dict[str, Any] | None = None
        converged = False
        termination_reason = "max_iterations_reached"
        for iteration in range(1, int(max_iterations) + 1):
            model.optimize()
            status = int(model.Status)
            has_incumbent = bool(model.SolCount > 0)
            if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or not has_incumbent:
                termination_reason = f"master_no_incumbent_status_{status}"
                break

            y = np.zeros(I, dtype=float)
            z = np.zeros(I, dtype=float)
            for i in range(I):
                y[i] = 1.0 if model.getVarByName(f"y[{i}]").X > 0.5 else 0.0
                z[i] = float(model.getVarByName(f"z[{i}]").X)
            theta = float(model.getVarByName("theta").X)

            rec_costs = np.full(S, np.nan, dtype=float)
            worst_idx: Optional[int] = None
            worst_cost = float("-inf")
            for s in candidate_indices:
                rec_out = self.solve_recourse_lp(
                    instance=instance,
                    scenario_index=s,
                    y=y,
                    z=z,
                    time_limit=recourse_limit,
                    threads=threads,
                    verbose=verbose,
                    return_solution=False,
                )
                if rec_out["has_incumbent"]:
                    rec_cost = float(rec_out["obj"])
                else:
                    rec_cost = float("inf")
                rec_costs[s] = rec_cost
                if rec_cost > worst_cost + float(duplicate_tolerance):
                    worst_cost = rec_cost
                    worst_idx = int(s)

            if worst_idx is None:
                termination_reason = "no_worst_scenario_found"
                break

            first_stage_cost = float(np.dot(f, y) + np.dot(a, z))
            upper_obj = first_stage_cost + worst_cost
            lower_bound = float(model.ObjVal)
            gap = None
            if np.isfinite(upper_obj) and upper_obj != 0.0:
                gap = abs(upper_obj - lower_bound) / abs(upper_obj)
            violation = worst_cost - theta

            last_solution = {
                "status": status,
                "has_incumbent": True,
                "obj": float(upper_obj),
                "obj_bound": float(lower_bound),
                "mip_gap": float(gap) if gap is not None else None,
                "y": y,
                "z": z,
                "theta": theta,
                "recourse_costs": rec_costs,
                "wc_idx": int(worst_idx),
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

        runtime_sec = float(time.perf_counter() - wall_start)
        if last_solution is None:
            return {
                "status": int(model.Status),
                "has_incumbent": False,
                "obj": None,
                "obj_bound": None,
                "mip_gap": None,
                "y": None,
                "z": None,
                "theta": None,
                "recourse_costs": None,
                "wc_idx": None,
                "runtime_sec": runtime_sec,
                "ccg_iterations": 0,
                "threads": int(model.Params.Threads),
                "ccg_converged": False,
                "ccg_termination_reason": termination_reason,
                "ccg_active_scenarios": list(active_scenarios),
                "ccg_violation": None,
            }

        last_solution["runtime_sec"] = runtime_sec
        last_solution["ccg_converged"] = bool(converged)
        last_solution["ccg_termination_reason"] = str(termination_reason)
        return last_solution

    def solve_exact(
        self,
        instance: Instance,
        time_limit: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
        backend: Optional[str] = None,
        mip_gap=None,
    ) -> Labels:
        backend_name = str(
            backend if backend is not None else getattr(self, "exact_backend", "bigmilp")
        ).lower()
        if backend_name == "bigmilp":
            out = self.solve_robust_bigmilp(
                instance,
                scenario_indices=None,
                time_limit=time_limit,
                threads=threads,
                verbose=verbose,
                mip_gap=mip_gap,
            )
        elif backend_name == "ccg":
            out = self.solve_robust_ccg(
                instance,
                scenario_indices=None,
                time_limit=time_limit,
                recourse_time_limit=self.ccg_recourse_time_limit,
                mip_gap=self.ccg_mip_gap,
                threads=threads,
                verbose=verbose,
                max_iterations=self.ccg_max_iterations,
                convergence_tolerance=self.ccg_convergence_tolerance,
                duplicate_tolerance=self.ccg_duplicate_tolerance,
            )
        else:
            raise ValueError(f"Unsupported exact backend '{backend_name}'")

        if out["has_incumbent"]:
            obj_val = float(out["obj"])
            y_star = np.asarray(out["y"], dtype=float)
            z_star = np.asarray(out["z"], dtype=float)
            rec_costs = out.get("recourse_costs")
            wc_idx = out.get("wc_idx")
        else:
            obj_val = float("inf")
            y_star = None
            z_star = None
            rec_costs = None
            wc_idx = None

        sce_target_idx = np.asarray([wc_idx], dtype=int) if wc_idx is not None else None

        labels = Labels(
            obj=obj_val,
            feasible=out["has_incumbent"],
            x=y_star,
            sce_target_idx=sce_target_idx,
            sce_ranking=None,
            sce_scores=rec_costs,
            sce_top_idx=None,
            obj_bound=out.get("obj_bound"),
            mip_gap=out.get("mip_gap"),
            meta={
                "status": out.get("status"),
                "z": z_star,
                "theta": out.get("theta"),
                "solver_backend": backend_name,
                "ccg_iterations": out.get("ccg_iterations"),
                "ccg_converged": out.get("ccg_converged"),
                "ccg_termination_reason": out.get("ccg_termination_reason"),
            },
        )
        return labels

    def evaluate_full_solution(
        self,
        instance: Instance,
        y: np.ndarray,
        z: np.ndarray,
        time_limit: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        data = self._extract_instance_data(instance)
        f, a = data["f"], data["a"]
        I, J, S = data["I"], data["J"], data["S"]

        y = np.asarray(y, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        if y.shape[0] != I or z.shape[0] != I:
            raise ValueError(f"y/z shape mismatch: got y={y.shape}, z={z.shape}, I={I}.")

        # Project z onto feasible region (handles solver tolerance issues)
        K = data["K"]
        z = np.minimum(z, K * y)

        rec_costs = np.full(S, np.inf, dtype=float)
        for s in range(S):
            rec_out = self.solve_recourse_lp(
                instance=instance,
                scenario_index=s,
                y=y,
                z=z,
                time_limit=time_limit,
                threads=threads,
                verbose=verbose,
                return_solution=False,
            )
            if rec_out["has_incumbent"]:
                rec_costs[s] = float(rec_out["obj"])

        worst_idx = int(np.argmax(rec_costs))
        worst_cost = float(rec_costs[worst_idx])
        total_obj = float(np.dot(f, y) + np.dot(a, z) + worst_cost)

        return {
            "obj": total_obj,
            "recourse_costs": rec_costs,
            "wc_idx": worst_idx,
            "wc_cost": worst_cost,
        }

    def solve_recourse_lp(
        self,
        instance: Instance,
        scenario_index: int,
        y: np.ndarray,
        z: np.ndarray,
        time_limit: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
        return_solution: bool = False,
    ) -> Dict[str, Any]:
        data = self._extract_instance_data(instance)
        I, J, S = data["I"], data["J"], data["S"]
        c, demand, K = data["c"], data["d"], data["K"]

        s = int(scenario_index)
        if s < 0 or s >= S:
            raise ValueError(f"scenario_index out of range: {s} not in [0, {S}).")

        y = np.asarray(y, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)
        if y.shape[0] != I or z.shape[0] != I:
            raise ValueError(f"y/z shape mismatch: got y={y.shape}, z={z.shape}, I={I}.")

        tol = 1e-6
        if np.any(z < -tol):
            raise ValueError("z must be nonnegative for recourse evaluation.")
        if np.any(z - K * y > tol):
            raise ValueError("z violates capacity-install constraint: z_i <= K_i * y_i.")

        env = self._make_env(time_limit=time_limit, threads=threads, verbose=verbose)
        model = gp.Model(env=env)

        x = model.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        model.addConstrs(
            (gp.quicksum(x[i, j] for j in range(J)) <= float(z[i]) for i in range(I)),
            name="capacity",
        )
        model.addConstrs(
            (gp.quicksum(x[i, j] for i in range(I)) >= float(demand[s, j]) for j in range(J)),
            name="demand",
        )

        model.setObjective(
            gp.quicksum(float(c[i, j]) * x[i, j] for i in range(I) for j in range(J)),
            GRB.MINIMIZE,
        )

        model.optimize()

        status = model.Status
        has_incumbent = model.SolCount > 0

        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and has_incumbent:
            rec_obj = float(model.ObjVal)
            if return_solution:
                x_vals = np.zeros((I, J), dtype=float)
                for i in range(I):
                    for j in range(J):
                        x_vals[i, j] = float(x[i, j].X)
            else:
                x_vals = None
            return {
                "status": int(status),
                "has_incumbent": True,
                "obj": rec_obj,
                "x": x_vals,
            }

        return {
            "status": int(status),
            "has_incumbent": False,
            "obj": None,
            "x": None,
        }

    def decision_dim(self, instance: Instance) -> int:
        data = self._extract_instance_data(instance)
        return int(data["I"])

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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

        demand_raw = per_instances[0].data.get("d", per_instances[0].data.get("demand"))
        if demand_raw is None:
            raise KeyError("CFLP instance missing required key 'd' (or legacy 'demand').")
        demand0 = torch.as_tensor(demand_raw)
        S = int(demand0.shape[0])

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
        obj_b = self._collate_optional_field(labels_list=labels_list, field_name="obj", expected_s=None)
        feas_b = self._collate_optional_field(labels_list=labels_list, field_name="feasible", expected_s=None)
        x_b = self._collate_optional_field(labels_list=labels_list, field_name="x", expected_s=None)
        sce_target_idx_b = self._collate_optional_field(
            labels_list=labels_list,
            field_name="sce_target_idx",
            expected_s=None,
        )
        sce_ranking_b = self._collate_optional_field(
            labels_list=labels_list,
            field_name="sce_ranking",
            expected_s=S,
        )
        sce_scores_b = self._collate_optional_field(
            labels_list=labels_list,
            field_name="sce_scores",
            expected_s=S,
        )
        sce_top_idx_b = self._collate_optional_field(
            labels_list=labels_list,
            field_name="sce_top_idx",
            expected_s=None,
        )
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

    def get_data_keys(self) -> List[str]:
        return ["f", "a", "K", "c", "d", "n_facilities", "n_customers", "n_scenarios"]

    def to_pyg_data(self, instance: Instance) -> List[Data]:
        cached = instance.cache_get(self._PYG_CACHE_KEY, None)
        if cached is not None:
            return cached

        # Features: [kind, is_yz, is_x, is_theta, obj, cost_hint, rhs, cos(row, obj)].
        # Variables: y[0:I], z[I:2I], x[2I:2I+IJ], theta; constraints: install, capacity, demand, epigraph.
        built = self._build_pyg_data(instance)
        instance.cache_set(self._PYG_CACHE_KEY, built)
        return built

    def to_pyg_data_det(self, instance: Instance) -> Data:
        data = self._extract_instance_data(instance)
        f, a, K, c = data["f"], data["a"], data["K"], data["c"]
        I, J = data["I"], data["J"]

        y_start = 0
        z_start = y_start + I
        x_start = z_start + I
        theta_idx = x_start + I * J

        n_vars = theta_idx + 1
        n_constraints = 2 * I + 1
        num_nodes = n_vars + n_constraints

        x_feat = torch.zeros((num_nodes, 8), dtype=torch.float32)

        x_feat[0:n_vars, 0] = 0.0
        x_feat[n_vars:, 0] = 1.0

        x_feat[y_start:y_start + I, 1] = 1.0
        x_feat[y_start:y_start + I, 4] = torch.from_numpy(f).float()

        x_feat[z_start:z_start + I, 1] = 1.0
        x_feat[z_start:z_start + I, 4] = torch.from_numpy(a).float()

        x_feat[x_start:theta_idx, 2] = 1.0
        x_feat[x_start:theta_idx, 5] = torch.from_numpy(c.reshape(-1)).float()

        x_feat[theta_idx, 3] = 1.0
        x_feat[theta_idx, 4] = 1.0

        full_obj_vec = np.zeros(n_vars, dtype=float)
        full_obj_vec[y_start:y_start + I] = f
        full_obj_vec[z_start:z_start + I] = a
        full_obj_vec[theta_idx] = 1.0

        constr_start = n_vars

        edge_indices: List[List[int]] = []
        edge_attrs: List[List[float]] = []

        def add_edge(u: int, v: int, w: float) -> None:
            edge_indices.append([u, v])
            edge_attrs.append(self._get_edge_features(w))
            edge_indices.append([v, u])
            edge_attrs.append(self._get_edge_features(w))

        # Install constraints: z_i - K_i y_i <= 0
        for i in range(I):
            c_idx = constr_start + i
            row = {
                z_start + i: 1.0,
                y_start + i: -float(K[i]),
            }
            for var_idx, coeff in row.items():
                add_edge(var_idx, c_idx, coeff)
            x_feat[c_idx, 6] = 0.0
            x_feat[c_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

        # Capacity constraints: sum_j x_ij - z_i <= 0
        cap_start = constr_start + I
        for i in range(I):
            c_idx = cap_start + i
            row = {z_start + i: -1.0}
            for j in range(J):
                var_idx = x_start + i * J + j
                row[var_idx] = 1.0
            for var_idx, coeff in row.items():
                add_edge(var_idx, c_idx, coeff)
            x_feat[c_idx, 6] = 0.0
            x_feat[c_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

        # Epigraph constraint: sum c_ij x_ij - theta <= 0
        epigraph_idx = cap_start + I
        row = {theta_idx: -1.0}
        for i in range(I):
            for j in range(J):
                var_idx = x_start + i * J + j
                row[var_idx] = float(c[i, j])
        for var_idx, coeff in row.items():
            add_edge(var_idx, epigraph_idx, coeff)
        x_feat[epigraph_idx, 6] = 0.0
        x_feat[epigraph_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        return Data(
            x=x_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_vars=torch.tensor([n_vars]),
            n_constraints=torch.tensor([n_constraints]),
        )

    def build_base_model(
        self,
        instance: Instance,
        time_limit: Optional[float] = None,
        threads: int = 1,
        verbose: bool = False,
    ) -> gp.Model:
        env = self._make_env(time_limit=time_limit, threads=threads, verbose=verbose)
        return self._build_base_model_with_env(instance, env=env)

    def add_scenario_to_model(self, model: gp.Model, instance: Instance, sce_idx: int) -> None:
        data = self._extract_instance_data(instance)
        I, J, S = data["I"], data["J"], data["S"]
        c, demand = data["c"], data["d"]

        s = int(sce_idx)
        if s < 0 or s >= S:
            raise ValueError(f"sce_idx out of range: {s} not in [0, {S}).")

        z_vars = [model.getVarByName(f"z[{i}]") for i in range(I)]
        theta_var = model.getVarByName("theta")

        x = model.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_s{s}")
        model.addConstrs(
            (gp.quicksum(x[i, j] for j in range(J)) <= z_vars[i] for i in range(I)),
            name=f"capacity_s{s}",
        )
        model.addConstrs(
            (gp.quicksum(x[i, j] for i in range(I)) >= float(demand[s, j]) for j in range(J)),
            name=f"demand_s{s}",
        )
        model.addConstr(
            gp.quicksum(float(c[i, j]) * x[i, j] for i in range(I) for j in range(J)) <= theta_var,
            name=f"rec_cost_s{s}",
        )

        model.update()

    def _build_pyg_data(self, instance: Instance) -> List[Data]:
        data = self._extract_instance_data(instance)
        f, a, K, c, demand = (
            data["f"],
            data["a"],
            data["K"],
            data["c"],
            data["d"],
        )
        I, J, S = data["I"], data["J"], data["S"]

        y_start = 0
        z_start = y_start + I
        x_start = z_start + I
        theta_idx = x_start + I * J

        n_vars = theta_idx + 1
        n_constraints = 2 * I + J + 1
        num_nodes = n_vars + n_constraints

        full_obj_vec = np.zeros(n_vars, dtype=float)
        full_obj_vec[y_start:y_start + I] = f
        full_obj_vec[z_start:z_start + I] = a
        full_obj_vec[theta_idx] = 1.0

        data_list: List[Data] = []

        for s in range(S):
            x_feat = torch.zeros((num_nodes, 8), dtype=torch.float32)
            x_feat[0:n_vars, 0] = 0.0
            x_feat[n_vars:, 0] = 1.0

            x_feat[y_start:y_start + I, 1] = 1.0
            x_feat[y_start:y_start + I, 4] = torch.from_numpy(f).float()

            x_feat[z_start:z_start + I, 1] = 1.0
            x_feat[z_start:z_start + I, 4] = torch.from_numpy(a).float()

            x_feat[x_start:theta_idx, 2] = 1.0
            x_feat[x_start:theta_idx, 5] = torch.from_numpy(c.reshape(-1)).float()

            x_feat[theta_idx, 3] = 1.0
            x_feat[theta_idx, 4] = 1.0

            constr_start = n_vars

            edge_indices: List[List[int]] = []
            edge_attrs: List[List[float]] = []

            def add_edge(u: int, v: int, w: float) -> None:
                edge_indices.append([u, v])
                edge_attrs.append(self._get_edge_features(w))
                edge_indices.append([v, u])
                edge_attrs.append(self._get_edge_features(w))

            # Install constraints: z_i - K_i y_i <= 0
            for i in range(I):
                c_idx = constr_start + i
                row = {z_start + i: 1.0, y_start + i: -float(K[i])}
                for var_idx, coeff in row.items():
                    add_edge(var_idx, c_idx, coeff)
                x_feat[c_idx, 6] = 0.0
                x_feat[c_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

            # Capacity constraints: sum_j x_ij - z_i <= 0
            cap_start = constr_start + I
            for i in range(I):
                c_idx = cap_start + i
                row = {z_start + i: -1.0}
                for j in range(J):
                    var_idx = x_start + i * J + j
                    row[var_idx] = 1.0
                for var_idx, coeff in row.items():
                    add_edge(var_idx, c_idx, coeff)
                x_feat[c_idx, 6] = 0.0
                x_feat[c_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

            # Demand constraints: -sum_i x_ij <= -d_j^s
            dem_start = cap_start + I
            for j in range(J):
                c_idx = dem_start + j
                row = {}
                for i in range(I):
                    var_idx = x_start + i * J + j
                    row[var_idx] = -1.0
                for var_idx, coeff in row.items():
                    add_edge(var_idx, c_idx, coeff)
                x_feat[c_idx, 6] = -float(demand[s, j])
                x_feat[c_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

            # Epigraph constraint: sum c_ij x_ij - theta <= 0
            epi_idx = dem_start + J
            row = {theta_idx: -1.0}
            for i in range(I):
                for j in range(J):
                    var_idx = x_start + i * J + j
                    row[var_idx] = float(c[i, j])
            for var_idx, coeff in row.items():
                add_edge(var_idx, epi_idx, coeff)
            x_feat[epi_idx, 6] = 0.0
            x_feat[epi_idx, 7] = self._compute_cosine_sim(row, full_obj_vec)

            if len(edge_indices) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

            data_list.append(
                Data(
                    x=x_feat,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    n_vars=torch.tensor([n_vars]),
                    n_constraints=torch.tensor([n_constraints]),
                    scenario_idx=torch.tensor([s]),
                )
            )

        return data_list

    def _build_base_model_with_env(self, instance: Instance, env: gp.Env) -> gp.Model:
        data = self._extract_instance_data(instance)
        f, a, K = data["f"], data["a"], data["K"]
        I = data["I"]

        model = gp.Model(env=env)

        y = model.addVars(I, vtype=GRB.BINARY, name="y")
        z = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="z")
        theta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="theta")

        model.addConstrs(
            (z[i] <= float(K[i]) * y[i] for i in range(I)),
            name="install",
        )

        model.setObjective(
            gp.quicksum(float(f[i]) * y[i] for i in range(I))
            + gp.quicksum(float(a[i]) * z[i] for i in range(I))
            + theta,
            GRB.MINIMIZE,
        )

        model._n_facilities = I
        model.update()
        return model

    def _extract_instance_data(self, instance: Instance) -> Dict[str, Any]:
        f = self._to_numpy(instance.data["f"])
        a_raw = instance.data.get("a", instance.data.get("g", None))
        K_raw = instance.data.get("K", instance.data.get("U", None))
        d_raw = instance.data.get("d", instance.data.get("demand", None))
        if a_raw is None or K_raw is None or d_raw is None:
            raise KeyError("CFLP instance missing required keys: expected a/K/d (or legacy g/U/demand).")

        a = self._to_numpy(a_raw)
        K = self._to_numpy(K_raw)
        c = self._to_numpy(instance.data["c"])
        demand = self._to_numpy(d_raw)

        if f.ndim > 1:
            f = f[0]
        if a.ndim > 1:
            a = a[0]
        if K.ndim > 1:
            K = K[0]
        if c.ndim > 2:
            c = c[0]
        if demand.ndim > 2:
            demand = demand[0]

        I = int(f.shape[0])
        if a.shape[0] != I or K.shape[0] != I:
            raise ValueError("f/a/K length mismatch for facilities.")
        if c.shape[0] != I:
            raise ValueError("c first dimension must match number of facilities.")

        J = int(c.shape[1])
        if demand.shape[1] != J:
            raise ValueError("d second dimension must match number of customers.")

        S = int(demand.shape[0])

        self._validate_nonnegative(f, "f")
        self._validate_nonnegative(a, "a")
        self._validate_nonnegative(K, "K")
        self._validate_nonnegative(c, "c")
        self._validate_nonnegative(demand, "d")

        return {
            "f": f.astype(float),
            "a": a.astype(float),
            "K": K.astype(float),
            "c": c.astype(float),
            "d": demand.astype(float),
            "I": I,
            "J": J,
            "S": S,
        }

    def _scenario_recourse_from_model(
        self,
        model: gp.Model,
        instance: Instance,
        scenario_index: int,
        I: int,
        J: int,
    ) -> float:
        data = self._extract_instance_data(instance)
        c = data["c"]
        s = int(scenario_index)

        rec_cost = 0.0
        for i in range(I):
            for j in range(J):
                var = model.getVarByName(f"x_s{s}[{i},{j}]")
                rec_cost += float(c[i, j]) * float(var.X)
        return float(rec_cost)

    def _make_env(
        self,
        time_limit: Optional[float],
        threads: int,
        verbose: bool,
        mip_gap: Optional[float] = None,
    ) -> gp.Env:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if verbose else 0)
        env.setParam("Threads", int(threads))
        if mip_gap is not None and mip_gap >= 0:
            env.setParam("MIPGap", float(mip_gap))
        if time_limit is not None and time_limit > 0:
            env.setParam("TimeLimit", float(time_limit))
        env.start()
        return env

    def _validate_nonnegative(self, arr: np.ndarray, name: str) -> None:
        if np.any(arr < -1e-9):
            raise ValueError(f"{name} must be nonnegative.")

    def _to_tensor(self, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        return torch.tensor(v)

    def _to_numpy(self, v: Any) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        if v is None:
            return np.asarray([])
        return np.asarray(v)

    def _sample_capacities_and_demands(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample capacities and scenario demands using the Zeng & Zhao ranges.
        Demands are generated from base d_j in [d_min, d_max] with
        deviations alpha in [0.1, 0.5]: d_j^s in [d_j*(1-alpha), d_j*(1+alpha)].
        Ensures total capacity >= worst-case scenario total demand.
        """
        max_attempts = 100
        K = self._rng.uniform(self.K_min, self.K_max, size=(self.n_facilities,)).astype(np.float32)
        total_capacity = float(np.sum(K))

        if self.scenario_generator is not None:
            demand: np.ndarray | None = None
            for _ in range(max_attempts):
                demand = self._sample_demands_from_generator()
                if float(np.max(np.sum(demand, axis=1))) <= total_capacity:
                    return K, demand
            if demand is None:
                raise RuntimeError("scenario_generator did not produce any demand sample.")
            return K, self._scale_demands_to_capacity(demand=demand, total_capacity=total_capacity)

        for _ in range(max_attempts):
            base_demand = self._rng.uniform(self.d_min, self.d_max, size=(self.n_customers,)).astype(np.float32)
            alpha = self._rng.uniform(0.1, 0.5, size=(self.n_customers,)).astype(np.float32)
            low = base_demand * (1.0 - alpha)
            high = base_demand * (1.0 + alpha)
            demand = self._rng.uniform(low, high, size=(self.n_scenarios, self.n_customers)).astype(np.float32)

            if float(np.max(np.sum(demand, axis=1))) <= total_capacity:
                return K, demand

        return K, self._scale_demands_to_capacity(demand=demand, total_capacity=total_capacity)

    def _sample_demands_from_generator(self) -> np.ndarray:
        if self.scenario_generator is None:
            raise RuntimeError("scenario_generator is required for custom demand sampling.")

        _, demand_raw = self.scenario_generator.sample_costs(
            n_nodes=self.n_customers,
            n_scenarios=self.n_scenarios,
        )
        demand = np.asarray(demand_raw, dtype=np.float32)
        expected_shape = (self.n_scenarios, self.n_customers)
        if demand.shape != expected_shape:
            raise ValueError(
                f"scenario_generator returned shape {demand.shape}, expected {expected_shape}."
            )
        if not np.all(np.isfinite(demand)):
            raise ValueError("scenario_generator returned non-finite demand values.")
        demand = np.clip(demand, self.d_min, self.d_max)
        return demand.astype(np.float32, copy=False)

    @staticmethod
    def _scale_demands_to_capacity(demand: np.ndarray, total_capacity: float) -> np.ndarray:
        scaled = np.asarray(demand, dtype=np.float32).copy()
        max_total = float(np.max(np.sum(scaled, axis=1)))
        if max_total > 0.0:
            scale = 0.99 * float(total_capacity) / max_total
            scaled = scaled * scale
        return scaled.astype(np.float32, copy=False)

    def _collate_optional_field(
        self,
        labels_list: List[Labels],
        field_name: str,
        expected_s: Optional[int],
    ) -> Optional[torch.Tensor]:
        vals = [getattr(lab, field_name, None) for lab in labels_list]
        all_none = all(v is None for v in vals)
        all_present = all(v is not None for v in vals)

        if all_none:
            return None
        if not all_present:
            raise ValueError(f"Some labels have '{field_name}' while others don't.")

        tensors = []
        truncate_scenarios = field_name in {"sce_ranking", "sce_scores"}
        for idx, v in enumerate(vals):
            t = self._to_tensor(v)
            if t.ndim == 0:
                t = t.unsqueeze(0)
            if truncate_scenarios and expected_s is not None and t.shape[0] != expected_s:
                if t.shape[0] < expected_s:
                    raise ValueError(
                        f"label.{field_name} shape {tuple(t.shape)} shorter than expected S={expected_s}."
                    )
                t = t[:expected_s]
            tensors.append(t)

        ref_shape = tensors[0].shape
        if not all(t.shape == ref_shape for t in tensors):
            raise ValueError(
                f"Inconsistent shapes for label.{field_name}: "
                f"{[tuple(t.shape) for t in tensors]}"
            )

        return torch.stack(tensors, dim=0)

    def _get_edge_features(self, coeff: float) -> List[float]:
        return [float(coeff)]

    def _compute_cosine_sim(self, row_coeffs: Dict[int, float], full_obj_vec: np.ndarray) -> float:
        dot = 0.0
        norm_sq = 0.0
        for idx, val in row_coeffs.items():
            dot += val * full_obj_vec[idx]
            norm_sq += val ** 2
        if norm_sq < 1e-9:
            return 0.0
        obj_norm = np.linalg.norm(full_obj_vec)
        if obj_norm < 1e-9:
            return 0.0
        return float(dot / (np.sqrt(norm_sq) * obj_norm))

    def get_scenarios(self, data_np: Dict[str, Any]) -> np.ndarray:
        if "d" in data_np:
            return np.asarray(data_np["d"])
        if "demand" in data_np:
            return np.asarray(data_np["demand"])
        if "sce" in data_np:
            return np.asarray(data_np["sce"])
        if "scenarios" in data_np:
            return np.asarray(data_np["scenarios"])
        raise KeyError("CFLPProblem expects scenario key 'd' or 'demand'.")


class CFLPEntityBipartiteProblem(CFLPProblem):
    """
    CFLP with entity-bipartite encoding: facility/customer nodes with edge costs.
    """

    _PYG_CACHE_KEY = "cflp_entity_bipartite_pyg_data_v1"

    def to_pyg_data(self, instance: Instance) -> List[Data]:
        cached = instance.cache_get(self._PYG_CACHE_KEY, None)
        if cached is not None:
            return cached

        data = self._extract_instance_data(instance)
        f, a, K, c, demand = data["f"], data["a"], data["K"], data["c"], data["d"]
        I, J, S = data["I"], data["J"], data["S"]

        data_list: List[Data] = []
        for s in range(S):
            graph = self._build_entity_bipartite_graph(
                f=f,
                a=a,
                K=K,
                c=c,
                demand=demand[s],
                scenario_idx=s,
                I=I,
                J=J,
            )
            data_list.append(graph)

        instance.cache_set(self._PYG_CACHE_KEY, data_list)
        return data_list

    def to_pyg_data_det(self, instance: Instance) -> Data:
        data = self._extract_instance_data(instance)
        f, a, K, c, demand = data["f"], data["a"], data["K"], data["c"], data["d"]
        I, J = data["I"], data["J"]

        demand_mean = demand.mean(axis=0)
        return self._build_entity_bipartite_graph(
            f=f,
            a=a,
            K=K,
            c=c,
            demand=demand_mean,
            scenario_idx=None,
            I=I,
            J=J,
        )

    def _build_entity_bipartite_graph(
        self,
        *,
        f: np.ndarray,
        a: np.ndarray,
        K: np.ndarray,
        c: np.ndarray,
        demand: np.ndarray,
        scenario_idx: Optional[int],
        I: int,
        J: int,
    ) -> Data:
        # Node features: [is_facility, is_customer, f, a, K, demand]
        x_feat = torch.zeros((I + J, 6), dtype=torch.float32)
        x_feat[0:I, 0] = 1.0
        x_feat[I:, 1] = 1.0
        x_feat[0:I, 2] = torch.from_numpy(f).float()
        x_feat[0:I, 3] = torch.from_numpy(a).float()
        x_feat[0:I, 4] = torch.from_numpy(K).float()
        x_feat[I:, 5] = torch.from_numpy(demand).float()

        # Edge list for complete bipartite graph: facilities <-> customers.
        fac_idx = torch.arange(I, dtype=torch.long).repeat_interleave(J)
        cust_idx = torch.arange(J, dtype=torch.long).repeat(I) + I
        edge_index = torch.stack(
            [torch.cat([fac_idx, cust_idx]), torch.cat([cust_idx, fac_idx])],
            dim=0,
        )
        c_flat = torch.from_numpy(c.reshape(-1)).float()
        edge_attr = torch.cat([c_flat, c_flat], dim=0).unsqueeze(1)

        data = Data(
            x=x_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_facilities=torch.tensor([I]),
            n_customers=torch.tensor([J]),
        )
        if scenario_idx is not None:
            data.scenario_idx = torch.tensor([int(scenario_idx)])
        return data
