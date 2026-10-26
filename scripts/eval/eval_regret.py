#!/usr/bin/env python3
"""Isolated regret evaluator — evaluates first-stage x on ALL scenarios.

SAFETY:
- Reads saved model selections from eval/ — NEVER writes there
- Writes ONLY to eval_regret/ with naming: {problem}/{tag}_{method}_{params}.json
- Completely independent from main eval pipeline

Usage:
    # Smoke test (1 instance)
    python eval_regret.py --smoke-test

    # NeurPRISE k=2 on VC medium
    python eval_regret.py --problem vertex_cover --scale medium --method neurprise --k 2
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
import numpy as np

def _resolve_base() -> str:
    """Repo root for this *copy* of the evaluator — never a hardcoded path.

    Derived from this file's location: the module lives at
    ``<root>/scripts/eval/eval_regret.py``, so root == parents[2].

    We deliberately do NOT use ``src.utils.utils._find_project_root()``: it
    walks up from ``src/utils/utils.py`` (the *imported* tree, not the caller's
    copy) and, when the ``.git`` / ``README.md`` marker is missing — the normal
    state of a copied tree — it falls back to ``src/``.
    """
    here = os.path.dirname(os.path.abspath(__file__))          # <root>/scripts/eval
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


BASE = _resolve_base()
sys.path.insert(0, BASE)

REGRET_DIR = os.path.join(BASE, "eval_regret")
# INPUT_ROOT is where inputs are READ from (default BASE keeps the 43 existing
# importers working); RUN is the RunPaths of this invocation when the script
# was started via main() with --run-id (None for library/legacy callers).
INPUT_ROOT = BASE
RUN = None
# Further read-only roots a run declares (e.g. the run dir of the e2e run that made
# the selections), so --eval-base can point at it instead of a copy inside INPUT_ROOT.
EVAL_ROOTS: list = []
THREADS = 40
MIP_GAP = 1e-4
TIME_LIMIT = 600

from src.utils import regret_json as _rj  # noqa: E402  (needs BASE on sys.path)
from src.utils.run_io import open_run  # noqa: E402
from src.eval.regret import (  # noqa: E402
    regret_pct,
    evaluate_x_on_all_scenarios as _core_evaluate_x_on_all_scenarios,
    solve_reduced_milp as _core_solve_reduced_milp,
    round_or_none as _round_or_none,
    time_components as _time_components,
    summarize,
)


def assert_within_base(path: str) -> str:
    """Refuse to write anywhere outside this copy's own repo root.

    ``BASE`` is derived from this file's location, so in an isolated tree this
    makes an accidental write into the source tree a hard error instead of a
    silent overwrite.
    """
    abs_path = os.path.abspath(path)
    base = os.path.abspath(BASE)
    if os.path.commonpath([abs_path, base]) != base:
        raise RuntimeError(
            f"SAFETY: refusing to write outside project root.\n"
            f"  project root (BASE) : {base}\n"
            f"  attempted path      : {abs_path}"
        )
    return abs_path


def _assert_inside(path: str, root: str, what: str) -> str:
    """Read guard: ``path`` must resolve inside ``root`` (the declared input root)."""
    abs_path = os.path.realpath(path)
    abs_root = os.path.realpath(root)
    if os.path.commonpath([abs_path, abs_root]) != abs_root:
        raise RuntimeError(
            f"SAFETY: {what} must be inside the input root.\n"
            f"  input root     : {abs_root}\n"
            f"  attempted path : {abs_path}"
        )
    return abs_path


def _assert_inside_any(path: str, roots, what: str) -> str:
    """``_assert_inside`` for several declared read roots: inside any one of them is enough."""
    for root in roots:
        try:
            return _assert_inside(path, root, what)
        except RuntimeError:
            continue
    raise RuntimeError(f"SAFETY: {what} must be inside the input root or a declared --eval-root.\n"
                       f"  roots          : {[os.path.realpath(r) for r in roots]}\n"
                       f"  attempted path : {os.path.realpath(path)}")


# ── Problem registry ──
PROBLEM_CONFIGS = {
    ("vertex_cover", "small"): {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "VC-20-50",
    },
    ("vertex_cover", "medium"): {
        "n_nodes": 50, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt500_n50_s50_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "VC-50-50",
    },
    ("selection", "small"): {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection/inst_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "SEL-20-50",
    },
    ("selection", "medium"): {
        "n_nodes": 50, "n_scenarios": 50, "p_select": 25,
        "data": "data/selection/inst_seed42_cnt500_n50_s50_p25_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "SEL-50-50",
    },
    ("cflp_ebp", "small"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp/inst_seed42_cnt2500_m30_n30_s50_scen_uniform.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "CFLP-30-30-50",
    },
    ("cflp_ebp", "medium"): {
        "n_facilities": 70, "n_customers": 70, "n_scenarios": 50,
        "data": "data/cflp_ebp/inst_seed42_cnt500_m70_n70_s50_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "CFLP-70-70-50",
    },
    # Large (Table 8)
    ("vertex_cover", "large"): {
        "n_nodes": 50, "n_scenarios": 100, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt500_n50_s100_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "VC-50-100",
    },
    ("selection", "large"): {
        "n_nodes": 50, "n_scenarios": 100, "p_select": 25,
        "data": "data/selection/inst_seed42_cnt500_n50_s100_p25_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "SEL-50-100",
    },
    ("cflp_ebp", "large"): {
        "n_facilities": 70, "n_customers": 70, "n_scenarios": 100,
        "data": "data/cflp_ebp/inst_seed42_cnt500_m70_n70_s100_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "CFLP-70-70-100",
    },
    # Scenario scaling — same n, larger S (trained on s=50)
    ("vertex_cover", "s100"): {
        "n_nodes": 20, "n_scenarios": 100, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt500_n20_s100_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "VC-20-100",
    },
    ("vertex_cover", "s200"): {
        "n_nodes": 20, "n_scenarios": 200, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt500_n20_s200_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "VC-20-200",
    },
    ("selection", "s100"): {
        "n_nodes": 20, "n_scenarios": 100, "p_select": 10,
        "data": "data/selection/inst_seed42_cnt500_n20_s100_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "SEL-20-100",
    },
    ("selection", "s200"): {
        "n_nodes": 20, "n_scenarios": 200, "p_select": 10,
        "data": "data/selection/inst_seed42_cnt500_n20_s200_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "SEL-20-200",
    },
    ("cflp_ebp", "s100"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 100,
        "data": "data/cflp_ebp/inst_seed42_cnt500_m30_n30_s100_scen_iid-uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "CFLP-30-30-100",
    },
    ("cflp_ebp", "s200"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 200,
        "data": "data/cflp_ebp/inst_seed42_cnt500_m30_n30_s200_scen_iid-uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "CFLP-30-30-200",
    },
    # Distribution shift — cross-distribution (uniform-trained model on Normal/MM data)
    ("vertex_cover", "normal_x"): {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover_normal/inst_seed42_cnt2500_n20_s50_p10_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "VC-normal-x",
    },
    ("vertex_cover", "mm_x"): {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover_mm/inst_seed42_cnt2500_n20_s50_p10_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "VC-mm-x",
    },
    ("selection", "normal_x"): {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection_normal/inst_seed42_cnt2500_n20_s50_p10_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "SEL-normal-x",
    },
    ("selection", "mm_x"): {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection_mm/inst_seed42_cnt2500_n20_s50_p10_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "SEL-mm-x",
    },
    ("cflp_ebp", "normal_x"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp_normal/inst_seed42_cnt2500_m30_n30_s50_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "CFLP-normal-x",
    },
    ("cflp_ebp", "mm_x"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp_mm/inst_seed42_cnt2500_m30_n30_s50_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "CFLP-mm-x",
    },
    # Distribution shift — in-distribution (trained ON Normal/MM)
    ("cflp_ebp", "normal_id"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp_normal/inst_seed42_cnt2500_m30_n30_s50_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "CFLP-normal-id",
    },
    ("cflp_ebp", "mm_id"): {
        "n_facilities": 30, "n_customers": 30, "n_scenarios": 50,
        "data": "data/cflp_ebp_mm/inst_seed42_cnt2500_m30_n30_s50_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "CFLP-mm-id",
    },
    ("vertex_cover", "normal_id"): {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover_normal/inst_seed42_cnt2500_n20_s50_p10_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "VC-normal-id",
    },
    ("vertex_cover", "mm_id"): {
        "n_nodes": 20, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover_mm/inst_seed42_cnt2500_n20_s50_p10_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "VC-mm-id",
    },
    ("selection", "normal_id"): {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection_normal/inst_seed42_cnt2500_n20_s50_p10_scen_normal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "SEL-normal-id",
    },
    ("selection", "mm_id"): {
        "n_nodes": 20, "n_scenarios": 50, "p_select": 10,
        "data": "data/selection_mm/inst_seed42_cnt2500_n20_s50_p10_scen_multimodal.pkl",
        "n_test": 250, "n_total": 2500,
        "tag": "SEL-mm-id",
    },
    # Ultra-large (Table 2) — trained on small, tested on ultra
    ("vertex_cover", "ultra"): {
        "n_nodes": 100, "n_scenarios": 50, "p_edge": 10,
        "data": "data/vertex_cover/inst_seed42_cnt500_n100_s50_p10_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "VC-100-50",
    },
    ("selection", "ultra"): {
        "n_nodes": 100, "n_scenarios": 50, "p_select": 50,
        "data": "data/selection/inst_seed42_cnt500_n100_s50_p50_scen_uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "SEL-100-50",
    },
    ("cflp_ebp", "ultra"): {
        "n_facilities": 150, "n_customers": 150, "n_scenarios": 50,
        "data": "data/cflp_ebp/inst_seed42_cnt500_m150_n150_s50_scen_iid-uniform.pkl",
        "n_test": 50, "n_total": 500,
        "tag": "CFLP-150-150-50",
    },
}

# Map k → k_ratio for each (problem, scale)
K_RATIOS = {
    50: {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 6: 0.12},
    100: {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 6: 0.06, 8: 0.08, 12: 0.12},
    200: {1: 0.005, 2: 0.01, 4: 0.02, 6: 0.03, 8: 0.04, 16: 0.08, 24: 0.12},
}


def make_problem(problem_name, cfg):
    if problem_name == "vertex_cover":
        from src.problems.vertex_cover import VertexCoverProblem
        return VertexCoverProblem(n_nodes=cfg["n_nodes"], n_scenarios=cfg["n_scenarios"], p_edge=cfg["p_edge"])
    elif problem_name == "selection":
        from src.problems.selection import SelectionProblem
        return SelectionProblem(n_nodes=cfg["n_nodes"], n_scenarios=cfg["n_scenarios"], p_select=cfg["p_select"])
    elif problem_name == "cflp_ebp":
        from src.problems.cflp import CFLPEntityBipartiteProblem
        return CFLPEntityBipartiteProblem(
            n_facilities=cfg["n_facilities"], n_customers=cfg["n_customers"],
            n_scenarios=cfg["n_scenarios"])
    raise ValueError(f"Unknown problem: {problem_name}")


def evaluate_x_on_all_scenarios(prob, inst, x_star, n_scenarios, problem_name="vertex_cover"):
    """Evaluate c^T x + max_{ξ∈Ξ} Q(x, ξ). Returns total objective.

    Thin wrapper over :func:`src.eval.regret.evaluate_x_on_all_scenarios`.
    It reads the module global ``THREADS`` **at call time** and forwards it, so
    the nine call sites that do ``eval_regret.THREADS = n`` keep working and
    ``configure_true_regret_solver``'s ``co_names`` guard still sees ``THREADS``.
    Do not turn ``THREADS`` into a default argument here — that binds at import
    time and silently ignores every later assignment.
    """
    return _core_evaluate_x_on_all_scenarios(
        prob, inst, x_star, n_scenarios, problem_name=problem_name,
        threads=THREADS)


def get_neurprise_x(prob, inst, sce_idx, mip_gap=None, problem_name="vertex_cover",
                    time_limit=None, return_raw=False):
    """Solve reduced MILP → return (first_stage_solution, reduced_obj).

    For VC/SEL: returns x (binary first-stage).
    For CFLP: returns (y, z) tuple (facility open + capacity).

    ``time_limit`` overrides the module default.
    ``return_raw=True`` additionally returns the solver dict — status, MIP gap,
    runtime — which the replay-evidence artifact records per instance.

    Thin wrapper over :func:`src.eval.regret.solve_reduced_milp`; reads the
    module globals ``THREADS`` / ``TIME_LIMIT`` / ``MIP_GAP`` at call time.
    ``mip_gap`` defaults to ``None`` rather than to ``MIP_GAP`` itself: that
    default would bind at import time, so a later ``module.MIP_GAP = g`` would be
    silently ignored — the same trap ``THREADS`` avoids.
    """
    return _core_solve_reduced_milp(
        prob, inst, sce_idx, problem_name=problem_name,
        mip_gap=MIP_GAP if mip_gap is None else mip_gap,
        threads=THREADS,
        time_limit=TIME_LIMIT if time_limit is None else time_limit,
        return_raw=return_raw)


def true_regret_single(prob, inst, sce_idx, full_obj, n_scenarios,
                       problem_name="vertex_cover", mip_gap=None):
    """Table-1 out-of-sample true regret for ONE instance.

    Solves the reduced (K-scenario) problem for the first-stage decision x*,
    then re-evaluates x* over ALL n_scenarios (worst-case recourse), and returns
    the Table-1 regret in percent: 100*(eval_on_xi - full_obj)/full_obj.

    Infeasible recourse in any scenario => eval_on_xi = +inf => feasible=False;
    the caller excludes such instances from feasible-only means and counts them
    as n_infeasible (repo convention, matches Table 1).

    This composes the same get_neurprise_x + evaluate_x_on_all_scenarios path
    that produced the paper's eval_regret/*.json numbers.

    It deliberately calls the two *module-level* wrappers above rather than the
    library functions directly, so a caller's ``eval_regret.THREADS = n`` still
    reaches the solver.  The percent formula itself comes from
    :func:`src.eval.regret.regret_pct`.
    """
    sol, red_obj = get_neurprise_x(prob, inst, sce_idx, mip_gap=mip_gap,
                                   problem_name=problem_name)
    if problem_name == "cflp_ebp":
        x = None if (sol is None or sol[0] is None) else sol
    else:
        x = sol
    if x is None:
        return {"eval_on_xi": float("nan"), "reduced_obj": red_obj,
                "regret_pct": float("nan"), "feasible": False}
    eval_obj = evaluate_x_on_all_scenarios(prob, inst, x, n_scenarios,
                                           problem_name=problem_name)
    feasible = bool(np.isfinite(eval_obj))
    pct = regret_pct(eval_obj, full_obj) if feasible else float("nan")
    return {"eval_on_xi": float(eval_obj) if feasible else float("inf"),
            "reduced_obj": red_obj, "regret_pct": pct, "feasible": feasible}


EVAL_DIRS = {
    ("vertex_cover", "small"): "eval/vertex_cover/s42_i2500_n20_s50",
    ("vertex_cover", "medium"): "eval/vertex_cover/s42_i500_n50_s50",
    ("selection", "small"): "eval/selection/s42_i2500_n20_s50",
    ("selection", "medium"): "eval/selection/s42_i500_n50_s50",
    ("cflp_ebp", "small"): "eval/cflp_ebp/s42_i2500_m30_n30_s50",
    ("cflp_ebp", "medium"): "eval/cflp_ebp/s42_i500_m70_n70_s50",
    # Distribution shift — cross-dist (uniform-trained on Normal/MM)
    ("vertex_cover", "normal_x"): "eval/vertex_cover_normal/s42_i2500_n20_s50",
    ("vertex_cover", "mm_x"): "eval/vertex_cover_mm/s42_i2500_n20_s50",
    ("selection", "normal_x"): "eval/selection_normal/s42_i2500_n20_s50",
    ("selection", "mm_x"): "eval/selection_mm/s42_i2500_n20_s50",
    ("cflp_ebp", "normal_x"): "eval/cflp_ebp_normal/s42_i2500_m30_n30_s50",
    ("cflp_ebp", "mm_x"): "eval/cflp_ebp_mm/s42_i2500_m30_n30_s50",
    # Dist shift — in-dist
    ("cflp_ebp", "normal_id"): "eval/cflp_ebp_normal/s42_i2500_m30_n30_s50",
    ("cflp_ebp", "mm_id"): "eval/cflp_ebp_mm/s42_i2500_m30_n30_s50",
    ("vertex_cover", "normal_id"): "eval/vertex_cover_normal/s42_i2500_n20_s50",
    ("vertex_cover", "mm_id"): "eval/vertex_cover_mm/s42_i2500_n20_s50",
    ("selection", "normal_id"): "eval/selection_normal/s42_i2500_n20_s50",
    ("selection", "mm_id"): "eval/selection_mm/s42_i2500_n20_s50",
    ("vertex_cover", "s100"): "eval/vertex_cover/s42_i500_n20_s100",
    ("vertex_cover", "s200"): "eval/vertex_cover/s42_i500_n20_s200",
    ("selection", "s100"): "eval/selection/s42_i500_n20_s100",
    ("selection", "s200"): "eval/selection/s42_i500_n20_s200",
    ("cflp_ebp", "s100"): "eval/cflp_ebp/s42_i500_m30_n30_s100",
    ("cflp_ebp", "s200"): "eval/cflp_ebp/s42_i500_m30_n30_s200",
    ("vertex_cover", "large"): "eval/vertex_cover/s42_i500_n50_s100",
    ("selection", "large"): "eval/selection/s42_i500_n50_s100",
    ("cflp_ebp", "large"): "eval/cflp_ebp/s42_i500_m70_n70_s100",
    ("vertex_cover", "ultra"): "eval/vertex_cover/s42_i500_n100_s50",
    ("selection", "ultra"): "eval/selection/s42_i500_n100_s50",
    ("cflp_ebp", "ultra"): "eval/cflp_ebp/s42_i500_m150_n150_s50",
}

def find_neurprise_eval_base(problem_name, inst_tag, scale=None):
    """Find paper-default NeurPRISE eval directory."""
    import glob
    # Try explicit mapping first
    for key, eval_dir in EVAL_DIRS.items():
        # If scale given, match exactly to avoid cross-directory ambiguity
        if scale is not None:
            if key != (problem_name, scale):
                continue
        elif not (key[0] == problem_name and inst_tag in eval_dir):
            continue
        full_dir = os.path.join(INPUT_ROOT, eval_dir)
        # Prefer: scale_norm + _test, then _all, avoid cross-scale (tr-/ev-)
        patterns = [
            "ML_s42_i*_t-train_scale_norm_m-det_source_sce_mean_l-score_weighted*_test/e2e_2ro",
            "ML_indist_*m-det_source_sce_mean*score_weighted*_test/e2e_2ro",
            "ML_s42_i*_t-train_scale_norm_m-det_source_sce_mean*score_weighted*_all/e2e_2ro",
            "ML_crossdist_*m-det_source_sce_mean*score_weighted*_test/e2e_2ro",
            "ML_s42_i*m-det_source_sce_mean*score_weighted*_test/e2e_2ro",
            "ML_s42_i*m-det_source_sce_mean*score_weighted*_all/e2e_2ro",
            "ML_*m-det_source_sce_mean*score_weighted*/e2e_2ro",
        ]
        for p in patterns:
            matches = glob.glob(os.path.join(full_dir, p))
            # For in-distribution (small/medium): filter out cross-scale evals
            if key[1] in ("small", "medium"):
                matches = [m for m in matches if "_tr-" not in m and "_ev-" not in m]
            # For dist-shift in-dist (_id): prefer ML_indist_
            elif key[1].endswith("_id"):
                matches = [m for m in matches if "indist" in m]
            # For dist-shift cross-dist (_x): prefer ML_s42_ or ML_crossdist_
            elif key[1].endswith("_x"):
                matches = [m for m in matches if "indist" not in m and "det_source_sce_mean" in m]
            # For generalization (ultra/large/s100/s200): KEEP cross-scale
            else:
                matches = [m for m in matches if "det_source_sce_mean" in m]
            if matches:
                return _single_eval_base(matches)
    # Fallback: search by inst_tag
    patterns = [
        f"eval/{problem_name}/{inst_tag}*/ML_*det_source_sce_mean*score_weighted*_all/e2e_2ro",
        f"eval/{problem_name}/{inst_tag}*/ML_*det_source_sce_mean*score_weighted*_test/e2e_2ro",
    ]
    for p in patterns:
        matches = glob.glob(os.path.join(INPUT_ROOT, p))
        if matches:
            return _single_eval_base(matches)
    return None


class AmbiguousEvalBase(RuntimeError):
    """More than one e2e directory matched; which one glob lists first is filesystem order."""


def _single_eval_base(matches):
    """With two matching e2e dirs, ``glob(...)[0]`` would make the scored selection depend on
    directory order. Refuse instead; ``--eval-base`` names one."""
    if len(matches) > 1:
        raise AmbiguousEvalBase(
            "several NeurPRISE e2e dirs match; pass --eval-base with one of:\n  "
            + "\n  ".join(sorted(matches)))
    return matches[0]


class IndexAlignmentError(RuntimeError):
    """The e2e selection file's rows cannot be matched to the test instances."""


def _align_test_rows(loaded_idx, n_test, n_instances, idx_path=""):
    """(selections for the test instances, how many) -- row i must be test instance i.

    ``all_idx.npy`` from an ``*_all`` e2e run has one row per instance of the WHOLE
    data file, in file order, while the test instances are ``all_instances[-n_test:]`` and
    ``load_full_objs`` takes the csv's LAST rows. ``loaded_idx[:n_test]`` would therefore
    score test instance i with the selection made for instance i of the file (a training
    instance). Like ``eval_regret_ablations.py``, this takes the last ``n_test`` rows, and
    it refuses any row count it cannot place.
      rows == n_test        -> test split, as is
      rows <  n_test        -> a partial test-split run (--max-instances), first rows
      rows == n_instances   -> whole-file run, last n_test rows
      anything else         -> IndexAlignmentError
    """
    rows = loaded_idx.shape[0]
    if rows <= n_test:
        return loaded_idx, rows
    if rows == n_instances:
        return loaded_idx[-n_test:], n_test
    raise IndexAlignmentError(
        f"{idx_path}: {rows} selection rows, but the test split has {n_test} and the data file "
        f"{n_instances} instances -- cannot tell which rows are the test instances")


def load_full_objs(eval_base, k_ratio, n_test):
    """Load V(Ξ) per instance from existing eval."""
    csv_path = os.path.join(eval_base, f"k_{k_ratio}", "all_per_instance.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(eval_base, f"k_{k_ratio}", "test_per_instance.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    objs = [float(r['full_obj']) for r in rows if r.get('full_obj', '') != '']
    return objs[-n_test:] if len(objs) > n_test else objs


def _read_selection_time(eval_base, k_ratio, n_test):
    """NeurPRISE selection time -- read, never recomputed here.

    The selection (which K scenarios to keep) was made by the e2e run that
    produced ``all_idx.npy``/``test_idx.npy`` (``scripts/04_05_eval_ml_2ro_e2e.py``),
    not by this evaluator, so the only honest source is the ``timing.json`` that
    run wrote alongside those indices. Absent or unparseable is a genuine
    "not measured", not a zero -- every field stays ``None`` in that case.
    """
    timing_path = os.path.join(eval_base, f"k_{k_ratio}", "timing.json")
    out = {
        "selection_time_total_sec": None,
        "selection_time_per_instance_sec": None,
        "selection_time_source": None,
    }
    try:
        with open(timing_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return out

    total = data.get("selection_time_total_sec")
    if not isinstance(total, (int, float)):
        return out
    per_instance = data.get("selection_time_per_instance_sec")
    if not isinstance(per_instance, (int, float)):
        per_instance = (total / n_test) if n_test else None

    out["selection_time_total_sec"] = float(total)
    out["selection_time_per_instance_sec"] = (
        float(per_instance) if per_instance is not None else None)
    out["selection_time_source"] = os.path.relpath(timing_path, INPUT_ROOT)
    return out


def run_neurprise_regret(problem_name, scale, k, mip_gap=None,
                         eval_base_override=None):
    """Compute NeurPRISE regret for given (problem, scale, k).

    ``mip_gap`` defaults to ``None`` and is forwarded as-is to ``get_neurprise_x``,
    which resolves it against the module global ``MIP_GAP`` at call time. A
    ``mip_gap=MIP_GAP`` default would bind at import time and silently ignore a
    later ``eval_regret.MIP_GAP = g``.
    """
    cfg = PROBLEM_CONFIGS[(problem_name, scale)]
    prob = make_problem(problem_name, cfg)
    S = cfg["n_scenarios"]

    with open(os.path.join(INPUT_ROOT, cfg["data"]), "rb") as f:
        all_instances = pickle.load(f)
    n_test = cfg["n_test"]
    test_instances = all_instances[-n_test:]

    k_ratio = K_RATIOS[S][k]
    # Use the known eval directory
    eval_dir_key = EVAL_DIRS.get((problem_name, scale), "")
    inst_tag = os.path.basename(eval_dir_key)  # e.g., "s42_i2500_n20_s50"

    if eval_base_override is None:
        eval_base = find_neurprise_eval_base(problem_name, inst_tag, scale=scale)
    else:
        eval_base = os.path.abspath(os.path.expanduser(str(eval_base_override)))
        # Read-only path (indices / per-instance csv / timing.json are loaded, nothing
        # is written here), so the write guard does not apply; the read contract is
        # "inside the declared input root" (run_io).
        _assert_inside_any(eval_base, [INPUT_ROOT, *EVAL_ROOTS], "explicit --eval-base")
        if not os.path.isdir(eval_base):
            print(f"ERROR: explicit eval base is not a directory: {eval_base}")
            return None
    if eval_base is None:
        print(f"ERROR: eval base not found for {problem_name}/{inst_tag}")
        return None

    # Try all_idx.npy first, fall back to test_idx.npy
    idx_path = os.path.join(eval_base, f"k_{k_ratio}", "all_idx.npy")
    if not os.path.exists(idx_path):
        idx_path = os.path.join(eval_base, f"k_{k_ratio}", "test_idx.npy")
    if not os.path.exists(idx_path):
        print(f"ERROR: indices not found for k_{k_ratio}")
        return None

    loaded_idx = np.load(idx_path)
    test_idx, n_avail = _align_test_rows(loaded_idx, n_test, len(all_instances), idx_path)
    test_inst = test_instances[:n_avail]

    full_objs = load_full_objs(eval_base, k_ratio, n_avail)
    if not full_objs or len(full_objs) < n_avail:
        # An e2e run on a 450-record trace file (CFLP-70-70-100) yields a
        # 45-row csv with empty full_obj; row i must exist for every test instance i.
        print(f"ERROR: full_objs has {0 if not full_objs else len(full_objs)} rows "
              f"< {n_avail} test instances under {eval_base}")
        return None

    results = []
    for i in range(n_avail):
        # ``return_raw`` also hands back the solver dict, so the reduced solve's
        # certified best bound survives into the artifact.  ``B_R`` is hypothesis (H2) of
        # Theorem 1 and is the ONLY term of the bracket
        # ``B_R <= V(Xi) <= Z(x_hat_R) <= U_bar_R`` that was not already persisted
        # (``U_bar_R`` is ``eval_on_xi``, ``V(Xi)`` is ``full_obj``).  Recording it makes
        # the certificate unconditional instead of resting on an exactness argument.
        # Measure model construction and the reduced solve together.
        _t_red = time.perf_counter()
        sol, red_obj, solve_out = get_neurprise_x(
            prob, test_inst[i], test_idx[i], mip_gap=mip_gap, problem_name=problem_name,
            return_raw=True)
        reduced_eval_wall_sec = time.perf_counter() - _t_red
        # Check for None (CFLP returns tuple, others return array)
        if problem_name == "cflp_ebp":
            if sol is None or sol[0] is None:
                continue
            x = sol  # (y, z) tuple
        else:
            if sol is None:
                continue
            x = sol

        fo = full_objs[i]

        eval_obj = evaluate_x_on_all_scenarios(prob, test_inst[i], x, S, problem_name=problem_name)
        regret = regret_pct(eval_obj, fo)

        results.append({
            "instance_id": i,
            "full_obj": fo,
            "reduced_obj": red_obj,
            "eval_on_xi": eval_obj,
            "regret": round(regret, 4),
            # Theorem 1 evidence (additive; readers that ignore these are unaffected).
            "reduced_obj_bound": solve_out.get("obj_bound"),
            "reduced_mip_gap": solve_out.get("mip_gap"),
            "reduced_status": solve_out.get("status"),
            "reduced_solve_sec": solve_out.get("runtime_sec"),
            "reduced_eval_wall_sec": round(float(reduced_eval_wall_sec), 6),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_avail}] regret={regret:.2f}%")

    return results


def _read_machine_id():
    """/etc/machine-id, or None. Measured on the box, never supplied by a caller.

    Chosen over the primary IPv4 because an IP is network-assigned, can change
    under DHCP without the machine changing, and is the identifier a future reader
    is most likely to RECONSTRUCT from a maintained host->IP table -- a composed
    value, not a measurement. machine-id is a property of the machine, needs no
    network and no root, and survives a rename.
    """
    try:
        with open("/etc/machine-id") as f:
            v = f.read().strip()
        return v or None
    except OSError:
        return None



def _result_name(problem_name, scale, method, k=None, mip_gap=None, extra=""):
    """Build the {tag}_{method}_k{k}_gap{gap}{extra}.json basename.

    Shared by ``save_results`` and ``main``'s ``--dry-run`` so the printed
    would-be OUT path is always the name that would actually be written.
    """
    tag = PROBLEM_CONFIGS[(problem_name, scale)]["tag"]
    parts = [tag, method]
    if k is not None:
        parts.append(f"k{k}")
    if mip_gap is not None and mip_gap != MIP_GAP:
        parts.append(f"gap{mip_gap}")
    if extra:
        parts.append(extra)
    return "_".join(parts) + ".json"


def save_results(results, problem_name, scale, method, k=None, mip_gap=None, extra="",
                  threads=None, server=None, selection_time_info=None):
    """Save to eval_regret/ with descriptive naming.

    ``selection_time_info`` (optional): the dict ``_read_selection_time``
    returns, for methods (NeurPRISE) whose selection was made by an earlier
    run and is not one of these rows' ``selection_sec`` -- the generic
    ``time_components`` below has no way to see it. When given, it overrides
    the ``selection_time_*``/``time_column_sec`` fields ``time_components``
    computed from the rows and marks ``selection_is_precomputed``. Every
    other caller (including tests) omits it and keeps the generic fields.
    """
    import socket, datetime
    cfg = PROBLEM_CONFIGS[(problem_name, scale)]
    tag = cfg["tag"]

    name = _result_name(problem_name, scale, method, k=k, mip_gap=mip_gap, extra=extra)
    rel = os.path.join("eval_regret", problem_name, name)

    summary = {
        # Strict-JSON encoding. Non-finite numbers are serialized as null
        # with a companion *_status field; readers use src/utils/regret_json.py.
        "schema_version": _rj.SCHEMA_VERSION,
        "tag": tag,
        "method": method,
        "k": k,
        "mip_gap": mip_gap,
        **summarize(results),
        # Traceability. `eval_server` is ALWAYS the measured host and is never
        # overridable by a caller: a caller-supplied name would identify no machine
        # and could exempt the run from host checks that read this field. A caller's
        # string (`--server`) lands in `eval_run_label`, which no guard reads.
        "eval_server": socket.gethostname(),
        "eval_run_label": server,
        # Second MEASURED identifier, because a hostname is not always unique (one
        # host's name can be a prefix of another's). Read from the machine at eval
        # time; there is deliberately no parameter to pass one in.
        "eval_machine_id": _read_machine_id(),
        "eval_threads": threads or THREADS,
        "eval_date": datetime.datetime.now().isoformat(),
        "note": "Regret is a post-hoc quality evaluation; its time fields are not a controlled timing measurement",
        # --- Time-column components, AGGREGATED FROM THE ROWS BELOW ---------
        # The paper's Time column is defined as
        #     sum-of-Gurobi-solves + selection time,   for every method
        # These four fields carry one component each, so two runs can be compared
        # field to field rather than by eyeballing one total.
        #
        # A total is emitted ONLY when every instance contributed. A partial sum
        # prints as a smaller number, not as an error, and would read as a faster
        # method; `*_n_recorded` says how many rows the sum actually saw, and the
        # total is null when that is short of `n_instances`.
        **_time_components(results),
        "per_instance": [_rj.annotate_record(r) for r in results],
    }

    if selection_time_info is not None:
        # Overrides what `_time_components` guessed from the rows (0 recorded,
        # since these rows carry no `selection_sec` -- the selection ran
        # outside this evaluator). `time_column_sec` is null unless BOTH halves
        # are real measurements; half a definition is a different quantity, not
        # a smaller number (same rule `_time_components` already applies).
        summary["selection_time_total_sec"] = selection_time_info["selection_time_total_sec"]
        summary["selection_time_per_instance_sec"] = selection_time_info["selection_time_per_instance_sec"]
        summary["selection_time_source"] = selection_time_info["selection_time_source"]
        summary["selection_is_precomputed"] = True
        sel_total = summary["selection_time_total_sec"]
        red_total = summary.get("reduced_eval_time_total_sec")
        summary["time_column_sec"] = (
            round(sel_total + red_total, 6)
            if (sel_total is not None and red_total is not None) else None
        )

    if RUN is not None:
        out_path = str(_rj.dump_strict_new(summary, RUN, rel))
    else:
        # Legacy path: no --run-id (direct/library callers, tests). Refuses to
        # overwrite: an eval rerun needs a fresh name/run, not a silent clobber.
        out_dir = os.path.join(REGRET_DIR, problem_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name)
        # SAFETY: assert we're NOT writing to eval/
        assert "eval_regret" in out_path, f"SAFETY: refusing to write outside eval_regret/: {out_path}"
        assert_within_base(out_path)
        if os.path.exists(out_path):
            raise FileExistsError(f"refusing to overwrite existing result {out_path}")
        with open(out_path, "x") as f:
            _rj.dump_strict_fh(summary, f)
    print(f"  Saved: {out_path}")

    # POST-SAVE GUARD: verify what we wrote is valid
    expected_n = cfg.get("n_test", 50)
    if len(results) != expected_n:
        print(f"  WARNING: saved {len(results)} instances but expected {expected_n}. "
              f"File may have wrong instance count.")

    return summary


def smoke_test():
    """Run on 1 instance to verify everything works."""
    print("=== SMOKE TEST (1 instance, VC medium, k=2) ===\n")

    cfg = PROBLEM_CONFIGS[("vertex_cover", "medium")]
    prob = make_problem("vertex_cover", cfg)

    with open(os.path.join(INPUT_ROOT, cfg["data"]), "rb") as f:
        all_instances = pickle.load(f)
    inst = all_instances[-1]  # last test instance

    # Get NeurPRISE indices
    eval_base = find_neurprise_eval_base("vertex_cover", "s42_i500_n50_s50")
    assert eval_base is not None, "Eval base not found"

    k_ratio = 0.04
    idx = np.load(os.path.join(eval_base, f"k_{k_ratio}", "all_idx.npy"))[-1]

    # Get full_obj
    full_objs = load_full_objs(eval_base, k_ratio, 1)
    fo = full_objs[0]

    print(f"  Instance: last test instance")
    print(f"  V(Ξ) = {fo}")
    print(f"  Selected scenarios: {idx}")

    # Step 1: solve reduced MILP
    x, red_obj = get_neurprise_x(prob, inst, idx)
    assert x is not None, "Failed to solve reduced MILP"
    print(f"  V(R_2) = {red_obj}")
    print(f"  x = {x[:10]}... (first 10)")

    # Step 2: evaluate x on all scenarios
    eval_obj = evaluate_x_on_all_scenarios(prob, inst, x, cfg["n_scenarios"], problem_name="vertex_cover")
    regret = (eval_obj - fo) / fo * 100
    print(f"  Eval on all Ξ = {eval_obj}")
    print(f"  Regret = {regret:.2f}%")

    # Step 3: verify regret ≥ 0 (x evaluated on Ξ ≥ V(Ξ))
    assert eval_obj >= fo - 1e-6, f"FAIL: eval_obj ({eval_obj}) < V(Ξ) ({fo})"
    print(f"\n  ASSERTIONS:")
    print(f"    V(R_2) ≤ V(Ξ): {red_obj:.2f} ≤ {fo:.2f} ✓")
    print(f"    eval_on_Ξ ≥ V(Ξ): {eval_obj:.2f} ≥ {fo:.2f} ✓")
    print(f"    regret ≥ 0: {regret:.2f}% ≥ 0 ✓")

    print(f"\n=== SMOKE TEST PASSED ===")


def _validate_run_args(parser, args):
    """Every run names its read root; every non-smoke run gets its own run dir."""
    if args.smoke_test:
        if not args.input_root:
            parser.error("--smoke-test requires --input-root")
        return
    if not (args.input_root and args.run_id):
        parser.error("--input-root and --run-id are required")
    if not args.problem or not args.scale or not args.method:
        parser.error("--problem, --scale, and --method are required (or use --smoke-test)")


def _resolve_eval_base(args):
    """``--eval-base`` if given (same normalization ``run_neurprise_regret``
    applies to ``eval_base_override``), else the glob search."""
    if args.eval_base:
        return os.path.abspath(os.path.expanduser(args.eval_base))
    inst_tag = os.path.basename(EVAL_DIRS.get((args.problem, args.scale), ""))
    return find_neurprise_eval_base(args.problem, inst_tag, scale=args.scale)


def _print_dry_run(args):
    """Print the resolved read paths and the would-be output path; solve nothing."""
    cfg = PROBLEM_CONFIGS[(args.problem, args.scale)]
    data_path = os.path.join(INPUT_ROOT, cfg["data"])
    eval_base = _resolve_eval_base(args)
    name = _result_name(args.problem, args.scale, args.method, k=args.k, mip_gap=args.mip_gap)
    out_path = os.path.join(RUN.run_dir, "eval_regret", args.problem, name) if RUN is not None else None
    print(f"DATA={data_path}")
    print(f"EVAL_BASE={eval_base}")
    print(f"OUT={out_path}")


def _run_neurprise(args):
    """Solve, save and report one NeurPRISE (problem, scale, k) cell."""
    print(f"=== NeurPRISE k={args.k} on {args.problem} {args.scale} (threads={args.threads}) ===")
    results = run_neurprise_regret(args.problem, args.scale, args.k, mip_gap=args.mip_gap,
                                    eval_base_override=args.eval_base)
    if results:
        # Historical CFLP in-distribution artifacts use ``neurprise_id``;
        # preserve that contract so verify_all_tables --rerun reads the
        # artifact produced by this invocation instead of a stale file.
        output_method = (
            "neurprise_id"
            if args.problem == "cflp_ebp" and args.scale.endswith("_id")
            else "neurprise"
        )
        # Same eval_base the solve above read indices/full_objs from (either
        # --eval-base, or found the same way `_print_dry_run` resolves it) --
        # `timing.json` lives beside them, one directory per k_<ratio>.
        cfg = PROBLEM_CONFIGS[(args.problem, args.scale)]
        k_ratio = K_RATIOS[cfg["n_scenarios"]][args.k]
        eval_base = _resolve_eval_base(args)
        selection_time_info = (
            _read_selection_time(eval_base, k_ratio, len(results)) if eval_base else None
        )
        summary = save_results(results, args.problem, args.scale, output_method,
                               k=args.k, mip_gap=args.mip_gap,
                               threads=THREADS, server=args.server,
                               selection_time_info=selection_time_info)
        regret = summary.get("regret_mean_feasible_only")
        regret_text = "n/a" if regret is None else f"{regret:.2f}%"
        print(f"  Regret (feasible only): {regret_text}")
        print(f"  Infeasible: {100 * summary['n_infeasible'] / summary['n_instances']:.2f}%")


def main():
    global INPUT_ROOT, RUN

    parser = argparse.ArgumentParser(description="Isolated regret evaluator")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--problem", choices=["vertex_cover", "selection", "cflp_ebp"])
    parser.add_argument("--scale", choices=["small", "medium", "large", "ultra", "s100", "s200",
                                              "normal_x", "mm_x", "normal_id", "mm_id"])
    parser.add_argument("--threads", type=int, default=THREADS, help="Gurobi threads (default: 40)")
    parser.add_argument("--server", type=str, default=None, help="Server name for traceability logging")
    parser.add_argument("--method", choices=["neurprise"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--mip-gap", type=float, default=MIP_GAP)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help="Gurobi time limit (s) for the reduced / full solves (default %(default)s, "
                             "kept for old callers; reportable runs pass 900). Recorded in RUN.json argv")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--input-root", type=str, default=None,
                        help="Root to READ inputs from (data/, eval/); required")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Exclusive run-directory id (src.utils.run_io); required unless --smoke-test")
    parser.add_argument("--batch", type=str, default="eval_regret",
                        help="run_io batch name for --run-id (default: eval_regret)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved read/write paths and exit before any solve")
    parser.add_argument("--eval-base", type=str, default=None,
                        help="Explicit e2e eval base holding k_<r>/{all_idx,test_idx}.npy, "
                             "all_per_instance.csv, timing.json; must lie inside an input root. "
                             "Skips the glob search in find_neurprise_eval_base.")
    parser.add_argument("--eval-root", action="append", default=[],
                        help="Extra READ-ONLY root (repeatable) that --eval-base may lie in, e.g. the "
                             "run dir of the e2e run that made the selections; declared in RUN.json")
    args = parser.parse_args()

    _validate_run_args(parser, args)
    if args.input_root:
        INPUT_ROOT = os.path.realpath(args.input_root)
        print(f"INPUT_ROOT={INPUT_ROOT}")
        if args.run_id:
            EVAL_ROOTS[:] = [os.path.realpath(os.path.expanduser(r)) for r in args.eval_root]
            RUN = open_run(args.batch, args.run_id, [INPUT_ROOT, *EVAL_ROOTS])
            print(f"RUN_DIR={RUN.run_dir}")

    if args.smoke_test:
        smoke_test()
        return

    if args.dry_run:
        _print_dry_run(args)
        return

    # Override module-level THREADS / TIME_LIMIT (both read at call time)
    sys.modules[__name__].THREADS = args.threads
    sys.modules[__name__].TIME_LIMIT = args.time_limit

    if args.method == "neurprise":
        _run_neurprise(args)


if __name__ == "__main__":
    from src.utils.log_utils import timestamp_stdout
    timestamp_stdout()   # script runs only: tests call main() and parse its output
    main()
