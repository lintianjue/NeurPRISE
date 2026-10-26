"""Evaluate reduced-scenario decisions over the full uncertainty set.

Regret (%) is ``100 * (eval_obj - full_obj) / full_obj``. Infeasible
recourse gives +inf and is excluded from feasible-only means by callers.
Solver settings are keyword arguments; CLI wrappers forward their own settings.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from src.utils import regret_json as _rj

# Formal-evaluation defaults.
# These are defaults for direct library callers only; the script wrappers pass
# their own module globals explicitly on every call.
THREADS: int = 40
MIP_GAP: float = 1e-4
TIME_LIMIT: float = 600

# Vertex-cover recourse MIPs are small; they carry their own short limit.
RECOURSE_TIME_LIMIT: float = 60


def regret_pct(eval_obj: float, full_obj: float) -> float:
    """Return percent regret relative to the full-scenario optimum.

    Compute ``100 * (eval_obj - full_obj) / full_obj``; an infinite evaluation
    objective propagates, and a zero full objective returns NaN.
    """
    if full_obj == 0:
        return float("nan")
    return (eval_obj - full_obj) / full_obj * 100.0


def evaluate_x_on_all_scenarios(
    prob: Any,
    inst: Any,
    x_star: Any,
    n_scenarios: int,
    problem_name: str = "vertex_cover",
    *,
    threads: int = THREADS,
    recourse_time_limit: float = RECOURSE_TIME_LIMIT,
) -> float:
    """Evaluate a fixed first-stage decision over every scenario.

    Use binary recourse for SEL/VC and continuous recourse for CFLP.
    Returns the first-stage cost plus worst-case recourse; missing incumbents give +inf.
    """
    data = inst.data

    if problem_name == "cflp_ebp":
        # CFLP: x_star is a (y, z) tuple — do NOT convert to array first.
        y_star, z_star = x_star
        y_arr = np.round(np.asarray(y_star, dtype=float)).reshape(-1)
        z_arr = np.asarray(z_star, dtype=float).reshape(-1) if z_star is not None else None
        out = prob.evaluate_full_solution(inst, y=y_arr, z=z_arr, threads=threads)
        return float(out["obj"])

    x_star = np.asarray(x_star, dtype=float).reshape(-1)
    det = np.asarray(data["det"], dtype=float).reshape(-1)
    first_stage_cost = float(np.dot(det, x_star))

    max_recourse = -np.inf
    for s in range(n_scenarios):
        if problem_name == "selection":
            out = prob.solve_recourse_greedy(inst, scenario_index=s, x_star=x_star)
        else:  # vertex_cover
            out = prob.solve_recourse_mip(
                inst, s, x_star, threads=threads, time_limit=recourse_time_limit)
        rc = float(out["obj"]) if out.get("has_incumbent", True) else float("inf")
        if rc > max_recourse:
            max_recourse = rc

    return first_stage_cost + max_recourse


def solve_reduced_milp(
    prob: Any,
    inst: Any,
    sce_idx: Any,
    problem_name: str = "vertex_cover",
    *,
    mip_gap: float = MIP_GAP,
    threads: int = THREADS,
    time_limit: float = TIME_LIMIT,
    return_raw: bool = False,
):
    """Solve the selected-scenario MILP and return ``(solution, reduced_obj)``.

    The solution is ``x`` for SEL/VC or ``(y, z)`` for CFLP. With
    ``return_raw=True``, append the solver result dict, including status, gap,
    runtime and ``obj_bound`` for bound verification.
    """
    from src.utils.batch_utils import _build_reduced_instance_from_np

    data_np = {
        k: np.array(v) if hasattr(v, "__len__") else v
        for k, v in inst.data.items()
    }
    reduced = _build_reduced_instance_from_np(prob, data_np, sce_idx)
    out = prob.solve_robust_bigmilp(
        reduced, time_limit=time_limit, threads=threads, verbose=False, mip_gap=mip_gap)
    red_obj = float(out["obj"]) if out.get("obj") is not None else None

    if problem_name == "cflp_ebp":
        sol = (out.get("y"), out.get("z"))
    else:
        sol = out.get("x")
    return (sol, red_obj, out) if return_raw else (sol, red_obj)


def round_or_none(value, ndigits: int = 4):
    """Round for display, but emit None for non-finite (strict JSON).

    ``scripts/eval/eval_regret.py`` re-exports it as ``_round_or_none``.
    """
    v = float(value)
    return round(v, ndigits) if np.isfinite(v) else None


def time_components(results) -> Dict[str, Any]:
    """Time-column components measured off the per-instance rows.

    Never transcribed from a caller: the numbers summed here are the same ones
    persisted in `per_instance`, so the summary can be re-derived from the file.

    ``scripts/eval/eval_regret.py`` re-exports it as ``_time_components``, where
    ``tests/scripts/test_time_column_components.py`` imports it.
    """
    def _total(key):
        vals = []
        for r in results:
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        return (float(sum(vals)), len(vals))

    n = len(results)
    sel_total, sel_n = _total("selection_sec")
    # The OLD field name keeps the OLD span: wall clock over build+solve.
    # Gurobi's model.Runtime goes to its own key, so the two cannot be
    # confused for one another.
    red_total, red_n = _total("reduced_eval_wall_sec")
    gur_total, gur_n = _total("reduced_solve_sec")
    sel_ok = n > 0 and sel_n == n
    red_ok = n > 0 and red_n == n
    # True only if EVERY row says so: a mixed batch is not a precomputed method,
    # and `all()` over an empty list is True, which would mislabel an empty run.
    precomputed = n > 0 and all(bool(r.get("selection_is_precomputed")) for r in results)
    out = {
        "selection_time_total_sec": round(sel_total, 6) if sel_ok else None,
        "selection_time_per_instance_sec": round(sel_total / n, 6) if sel_ok else None,
        "selection_time_n_recorded": sel_n,
        "reduced_eval_time_total_sec": round(red_total, 6) if red_ok else None,
        "reduced_eval_time_per_instance_sec": round(red_total / n, 6) if red_ok else None,
        "reduced_eval_time_n_recorded": red_n,
        # Gurobi's own solve time, summed. Machine-comparable in a way the wall
        # clock is not, but NOT what the published column measured -- on
        # SEL-20-50 k=1 it is ~44x smaller than the printed 4.7 s.
        "reduced_solve_gurobi_total_sec": round(gur_total, 6) if (n > 0 and gur_n == n) else None,
        "reduced_solve_gurobi_n_recorded": gur_n,
        # What the paper's Time column would print for this run. Null unless both
        # halves are complete -- half a definition is not a smaller number, it is
        # a different quantity.
        "time_column_sec": round(sel_total + red_total, 6) if (sel_ok and red_ok) else None,
        # `sor`/`prise` read a precomputed index array, so their selection time is
        # a lookup and NOT the optimisation or greedy that produced the indices.
        # Their offline cost is not recoverable from this evaluator.
        "selection_is_precomputed": precomputed,
    }
    return out


def summarize(results) -> Dict[str, Any]:
    """The regret summary statistics ``save_results`` writes into its JSON.

    Used by ``save_results`` in ``scripts/eval/eval_regret.py``. Uses
    :func:`src.utils.regret_json.from_status` exactly as ``save_results`` does, so both the legacy (raw float) and strict (null +
    status) row encodings resolve to the same statistics.
    """
    regrets = [_rj.from_status(r.get("regret"), r.get("regret_status")) for r in results]
    feasible = [v for v in regrets if np.isfinite(v)]

    return {
        "n_instances": len(results),
        "n_infeasible": int(sum(1 for v in regrets if not np.isfinite(v))),
        "regret_mean": round_or_none(np.mean(regrets) if regrets else np.nan),
        "regret_std": round_or_none(np.std(regrets) if len(feasible) == len(regrets) > 0 else np.nan),
        "regret_median": round_or_none(np.median(regrets) if regrets else np.nan),
        "regret_mean_feasible_only": round_or_none(np.mean(feasible) if feasible else np.nan),
    }
