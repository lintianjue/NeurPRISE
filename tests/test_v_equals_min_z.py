#!/usr/bin/env python3
"""Verify the fundamental identity: V(Xi) = min_x Z(x)
where Z(x) = c^T x + max_xi Q(x, xi).

V(Xi)  is computed by solve_robust_bigmilp()  (full-scenario MILP).
Z(x*)  is computed by evaluate_x_on_all_scenarios() (first-stage cost + worst-case recourse).

For the OPTIMAL x*, these must agree.
For any SUBOPTIMAL x (e.g. from a reduced k=1 problem), Z(x) >= V(Xi).

For CFLP: when first-stage capacity is insufficient for some scenario's demand,
the recourse LP is infeasible and evaluate_full_solution should return inf for
that scenario's recourse cost.
"""
import os
import sys
import pickle
import time
import traceback

import numpy as np


from src.problems.selection import SelectionProblem
from src.problems.vertex_cover import VertexCoverProblem
from src.problems.cflp import CFLPEntityBipartiteProblem
from src.core.types import Instance

# -- Settings --
BASE = os.environ.get("NEURPRISE_INPUT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
THREADS = 20
MIP_GAP = 1e-4
TIME_LIMIT = 600
N_INSTANCES = 3   # first 3 test instances
REL_TOL = 1e-4    # relative tolerance for V == Z

# -- Problem configs (from eval_regret.py PROBLEM_CONFIGS) --
CONFIGS = {
    "selection": {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection/inst_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
    },
    "vertex_cover": {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
    },
    "cflp_ebp": {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp/inst_seed42_cnt2500_m30_n30_s50_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
    },
}


def make_problem(name, cfg):
    if name == "vertex_cover":
        return VertexCoverProblem(
            n_nodes=cfg["n_nodes"], n_scenarios=cfg["n_scenarios"],
            p_edge=cfg["p_edge"])
    elif name == "selection":
        return SelectionProblem(
            n_nodes=cfg["n_nodes"], n_scenarios=cfg["n_scenarios"],
            p_select=cfg["p_select"])
    elif name == "cflp_ebp":
        return CFLPEntityBipartiteProblem(
            n_facilities=cfg["n_facilities"], n_customers=cfg["n_customers"],
            n_scenarios=cfg["n_scenarios"])
    raise ValueError(f"Unknown problem: {name}")


def evaluate_x_on_all_scenarios(prob, inst, x_star, n_scenarios, problem_name):
    """Replicates eval_regret.evaluate_x_on_all_scenarios exactly."""
    data = inst.data

    if problem_name == "cflp_ebp":
        y_star, z_star = x_star
        y_arr = np.round(np.asarray(y_star, dtype=float)).reshape(-1)
        z_arr = np.asarray(z_star, dtype=float).reshape(-1) if z_star is not None else None
        out = prob.evaluate_full_solution(inst, y=y_arr, z=z_arr, threads=THREADS)
        return float(out["obj"])

    x_star = np.asarray(x_star, dtype=float).reshape(-1)
    det = np.asarray(data["det"], dtype=float).reshape(-1)
    first_stage_cost = float(np.dot(det, x_star))

    max_recourse = -np.inf
    for s in range(n_scenarios):
        if problem_name == "selection":
            out = prob.solve_recourse_greedy(inst, scenario_index=s, x_star=x_star)
        else:  # vertex_cover
            out = prob.solve_recourse_mip(inst, s, x_star, threads=THREADS, time_limit=60)
        rc = float(out["obj"]) if out.get("has_incumbent", True) else float("inf")
        if rc > max_recourse:
            max_recourse = rc

    return first_stage_cost + max_recourse


def get_reduced_x(prob, inst, k, problem_name):
    """Solve k=1 reduced problem: pick k random scenarios and solve bigmilp on them.
    Returns (x_or_tuple, reduced_obj)."""
    from src.utils.batch_utils import _build_reduced_instance_from_np
    S = inst.data["sce"].shape[0] if "sce" in inst.data else inst.data["d"].shape[0]
    rng = np.random.RandomState(42)
    sce_idx = rng.choice(S, size=k, replace=False)
    sce_idx.sort()

    data_np = {key: np.array(v) if hasattr(v, '__len__') else v
               for key, v in inst.data.items()}
    reduced = _build_reduced_instance_from_np(prob, data_np, sce_idx)
    out = prob.solve_robust_bigmilp(reduced, time_limit=TIME_LIMIT,
                                     threads=THREADS, verbose=False,
                                     mip_gap=MIP_GAP)
    red_obj = float(out["obj"]) if out.get("obj") is not None else None

    if problem_name == "cflp_ebp":
        return (out.get("y"), out.get("z")), red_obj
    else:
        return out.get("x"), red_obj


def rel_diff(a, b):
    """Relative difference: |a - b| / max(|a|, |b|, 1e-10)."""
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


# ===========================================================================
#  Test runner
# ===========================================================================
def main():
    results = {}  # (test_name) -> (PASS/FAIL, details)
    overall_pass = True

    for problem_name in ["selection", "vertex_cover", "cflp_ebp"]:
        cfg = CONFIGS[problem_name]
        data_path = os.path.join(BASE, cfg["data"])
        print(f"\n{'='*70}")
        print(f"PROBLEM: {problem_name}")
        print(f"{'='*70}")

        prob = make_problem(problem_name, cfg)
        with open(data_path, "rb") as f:
            all_instances = pickle.load(f)

        # Use the last n_test instances (test split), take first N_INSTANCES
        n_test = cfg["n_test"]
        test_instances = all_instances[-n_test:][:N_INSTANCES]
        S = cfg["n_scenarios"]

        # =====================================================================
        # TEST 1: V(Xi) == Z(x*) for optimal x*
        # =====================================================================
        for idx in range(len(test_instances)):
            test_key = f"{problem_name}_inst{idx}_V_eq_Z"
            inst = test_instances[idx]
            print(f"\n--- Instance {idx}: solve bigmilp (all {S} scenarios) ---")

            t0 = time.time()
            try:
                out = prob.solve_robust_bigmilp(
                    inst, time_limit=TIME_LIMIT, threads=THREADS,
                    verbose=False, mip_gap=MIP_GAP)
            except Exception as e:
                msg = f"FAIL: bigmilp raised {e}"
                print(msg)
                traceback.print_exc()
                results[test_key] = ("FAIL", msg)
                overall_pass = False
                continue

            t_bigmilp = time.time() - t0

            if not out.get("has_incumbent"):
                msg = f"FAIL: bigmilp returned no incumbent (status={out.get('status')})"
                print(msg)
                results[test_key] = ("FAIL", msg)
                overall_pass = False
                continue

            V_xi = float(out["obj"])

            # Extract optimal first-stage solution
            if problem_name == "cflp_ebp":
                x_star = (out["y"], out["z"])
            else:
                x_star = out["x"]

            print(f"  V(Xi) = {V_xi:.6f}  (bigmilp, {t_bigmilp:.1f}s)")

            # Evaluate Z(x*) = c^T x* + max_xi Q(x*, xi)
            t0 = time.time()
            try:
                Z_xstar = evaluate_x_on_all_scenarios(
                    prob, inst, x_star, S, problem_name)
            except Exception as e:
                msg = f"FAIL: evaluate raised {e}"
                print(msg)
                traceback.print_exc()
                results[test_key] = ("FAIL", msg)
                overall_pass = False
                continue

            t_eval = time.time() - t0
            print(f"  Z(x*) = {Z_xstar:.6f}  (eval, {t_eval:.1f}s)")

            rd = rel_diff(V_xi, Z_xstar)
            print(f"  rel_diff = {rd:.2e}  (tol = {REL_TOL:.0e})")

            if rd <= REL_TOL:
                print(f"  PASS: V(Xi) == Z(x*)")
                results[test_key] = ("PASS", f"V={V_xi:.6f} Z={Z_xstar:.6f} rd={rd:.2e}")
            else:
                msg = f"FAIL: V(Xi)={V_xi:.6f} != Z(x*)={Z_xstar:.6f}, rel_diff={rd:.2e}"
                print(f"  {msg}")
                results[test_key] = ("FAIL", msg)
                overall_pass = False

        # =====================================================================
        # TEST 2: For suboptimal x (k=1 reduced), Z(x) >= V(Xi)
        # =====================================================================
        # Re-solve bigmilp for instance 0 to get V(Xi), then compare with k=1
        inst = test_instances[0]
        test_key = f"{problem_name}_suboptimal_Z_ge_V"

        print(f"\n--- Suboptimality test: k=1 reduced vs full ---")
        try:
            out_full = prob.solve_robust_bigmilp(
                inst, time_limit=TIME_LIMIT, threads=THREADS,
                verbose=False, mip_gap=MIP_GAP)
            V_xi = float(out_full["obj"])

            x_red, red_obj = get_reduced_x(prob, inst, k=1, problem_name=problem_name)
            if x_red is None or (problem_name == "cflp_ebp" and x_red[0] is None):
                msg = "FAIL: reduced problem returned no solution"
                print(f"  {msg}")
                results[test_key] = ("FAIL", msg)
                overall_pass = False
            else:
                Z_red = evaluate_x_on_all_scenarios(
                    prob, inst, x_red, S, problem_name)
                print(f"  V(Xi)     = {V_xi:.6f}")
                print(f"  Z(x_red)  = {Z_red:.6f}")
                print(f"  red_obj   = {red_obj:.6f}  (lower bound from k=1 sub-problem)")

                if Z_red >= V_xi - V_xi * REL_TOL:
                    gap_pct = (Z_red - V_xi) / V_xi * 100 if V_xi != 0 else 0
                    print(f"  PASS: Z(x_red) >= V(Xi)  (gap = {gap_pct:.2f}%)")
                    results[test_key] = ("PASS", f"Z_red={Z_red:.6f} >= V={V_xi:.6f}, gap={gap_pct:.2f}%")
                else:
                    msg = (f"FAIL: Z(x_red)={Z_red:.6f} < V(Xi)={V_xi:.6f} "
                           f"(violation = {(V_xi - Z_red)/V_xi*100:.4f}%)")
                    print(f"  {msg}")
                    results[test_key] = ("FAIL", msg)
                    overall_pass = False

        except Exception as e:
            msg = f"FAIL: suboptimality test raised {e}"
            print(f"  {msg}")
            traceback.print_exc()
            results[test_key] = ("FAIL", msg)
            overall_pass = False

    # =====================================================================
    # TEST 3: CFLP infeasibility — construct a first-stage that under-provisions
    # =====================================================================
    print(f"\n{'='*70}")
    print("CFLP INFEASIBILITY TEST")
    print(f"{'='*70}")
    test_key = "cflp_ebp_infeasible_recourse"
    try:
        cfg = CONFIGS["cflp_ebp"]
        prob = make_problem("cflp_ebp", cfg)
        data_path = os.path.join(BASE, cfg["data"])
        with open(data_path, "rb") as f:
            all_instances = pickle.load(f)
        inst = all_instances[-cfg["n_test"]]  # first test instance

        I = cfg["n_facilities"]
        J = cfg["n_customers"]
        S = cfg["n_scenarios"]

        # Build a deliberately under-provisioned first-stage:
        # Open only 1 facility with minimal capacity
        y_bad = np.zeros(I, dtype=float)
        y_bad[0] = 1.0  # open only facility 0
        # Set capacity to a tiny value (much less than total demand)
        K = np.asarray(inst.data["K"], dtype=float).reshape(-1)
        z_bad = np.zeros(I, dtype=float)
        z_bad[0] = 1.0  # capacity = 1 unit at facility 0

        out = prob.evaluate_full_solution(inst, y=y_bad, z=z_bad, threads=THREADS)
        rec_costs = out["recourse_costs"]
        n_inf = np.sum(np.isinf(rec_costs))
        obj_val = out["obj"]

        print(f"  Under-provisioned: y_bad opens 1/{I} facilities, z=1.0")
        print(f"  Recourse costs: {n_inf}/{S} scenarios infeasible (inf)")
        print(f"  Total obj = {obj_val}")

        if n_inf > 0 and np.isinf(obj_val):
            print(f"  PASS: infeasible recourse correctly returns inf")
            results[test_key] = ("PASS", f"{n_inf}/{S} scenarios infeasible, obj=inf")
        elif n_inf > 0:
            # Some scenarios infeasible but obj not inf? Check if worst-case is inf
            wc_cost = out["wc_cost"]
            if np.isinf(wc_cost):
                print(f"  PASS: worst-case recourse is inf (obj={obj_val})")
                results[test_key] = ("PASS", f"wc_cost=inf, {n_inf}/{S} infeasible")
            else:
                msg = f"FAIL: {n_inf} infeasible scenarios but wc_cost={wc_cost}"
                print(f"  {msg}")
                results[test_key] = ("FAIL", msg)
                overall_pass = False
        else:
            # All feasible? This means capacity was enough -- try even smaller
            msg = f"FAIL: expected infeasible recourse but all {S} scenarios feasible"
            print(f"  {msg}")
            results[test_key] = ("FAIL", msg)
            overall_pass = False

    except Exception as e:
        msg = f"FAIL: infeasibility test raised {e}"
        print(f"  {msg}")
        traceback.print_exc()
        results[test_key] = ("FAIL", msg)
        overall_pass = False

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_pass = sum(1 for v in results.values() if v[0] == "PASS")
    n_fail = sum(1 for v in results.values() if v[0] == "FAIL")
    n_total = len(results)

    for key, (status, detail) in sorted(results.items()):
        marker = "OK" if status == "PASS" else "XX"
        print(f"  [{marker}] {key}: {detail}")

    print(f"\n  {n_pass}/{n_total} PASSED, {n_fail}/{n_total} FAILED")
    if overall_pass:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
