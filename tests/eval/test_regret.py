"""Tests for src/eval/regret.py and the thin wrappers left in the eval_regret script.

These pin the behaviour that moving the maths into src/eval/regret.py had to
preserve exactly: the percent formula (including how it treats infeasible and zero-denominator rows),
and the module-global solver-setting idiom that nine call sites depend on.

No Gurobi: every problem object here is a fake that records what it was handed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.regret import (
    evaluate_x_on_all_scenarios,
    regret_pct,
    solve_reduced_milp,
)
from scripts.eval import eval_regret as er


# --------------------------------------------------------------------------
# The formula
# --------------------------------------------------------------------------

def test_regret_pct_is_percent_excess_over_full_objective():
    # Motivation: this single expression is every regret number in the paper.
    assert regret_pct(110.0, 100.0) == pytest.approx(10.0)
    assert regret_pct(100.0, 100.0) == pytest.approx(0.0)
    assert regret_pct(95.0, 100.0) == pytest.approx(-5.0)


def test_regret_pct_propagates_infeasible_as_inf_not_nan():
    # Motivation: the batch driver writes this value into per-instance rows and
    # n_infeasible counts non-finite entries. Collapsing +inf to NaN would
    # silently rewrite every infeasible row.
    assert regret_pct(float("inf"), 100.0) == float("inf")


def test_regret_pct_zero_denominator_is_nan():
    # Motivation: the formula guards `if full_obj != 0`.
    assert np.isnan(regret_pct(10.0, 0.0))


# --------------------------------------------------------------------------
# Solver settings reach the solver
# --------------------------------------------------------------------------

class _RecordingVertexCover:
    """Fake VC problem: records the threads/time_limit each recourse solve got."""

    def __init__(self):
        self.recourse_calls = []
        self.bigmilp_calls = []

    def solve_recourse_mip(self, inst, s, x_star, threads=None, time_limit=None):
        self.recourse_calls.append({"threads": threads, "time_limit": time_limit})
        return {"obj": 1.0, "has_incumbent": True}

    def solve_robust_bigmilp(self, reduced, time_limit=None, threads=None,
                             verbose=False, mip_gap=None):
        self.bigmilp_calls.append(
            {"threads": threads, "time_limit": time_limit, "mip_gap": mip_gap})
        return {"obj": 7.0, "x": np.array([1.0, 0.0])}


def _vc_instance():
    return SimpleNamespace(data={"det": np.array([1.0, 2.0])})


def test_library_takes_threads_as_an_explicit_argument():
    # Motivation: the library must have no hidden global state; that is the
    # whole point of moving the maths out of the script.
    prob = _RecordingVertexCover()
    evaluate_x_on_all_scenarios(
        prob, _vc_instance(), np.array([1.0, 0.0]), n_scenarios=2,
        problem_name="vertex_cover", threads=7)
    assert [c["threads"] for c in prob.recourse_calls] == [7, 7]


def test_patching_script_module_global_still_reaches_the_solver(monkeypatch):
    # Motivation: THE regression guard for this refactor. Nine call sites do
    # `eval_regret.THREADS = n`. Had the wrapper bound THREADS as a default, or
    # had the globals moved into src/, every one of those assignments would have
    # become a silent no-op and the paper's timing numbers would be invalid
    # while configure_true_regret_solver's co_names guard still reported success.
    prob = _RecordingVertexCover()
    monkeypatch.setattr(er, "THREADS", 7)
    er.evaluate_x_on_all_scenarios(
        prob, _vc_instance(), np.array([1.0, 0.0]), 2, problem_name="vertex_cover")
    assert [c["threads"] for c in prob.recourse_calls] == [7, 7]


def test_patching_time_limit_and_threads_reaches_the_reduced_solve(monkeypatch):
    # Motivation: get_neurprise_x is the other function configure_true_regret_solver
    # proves consumption for; it must read both globals at call time.
    prob = _RecordingVertexCover()
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    monkeypatch.setattr(er, "THREADS", 11)
    monkeypatch.setattr(er, "TIME_LIMIT", 123)
    er.get_neurprise_x(prob, _vc_instance(), np.array([0, 1]), problem_name="vertex_cover")
    assert prob.bigmilp_calls == [{"threads": 11, "time_limit": 123, "mip_gap": er.MIP_GAP}]


def test_patching_mip_gap_now_takes_effect(monkeypatch):
    # Motivation: bound as a default argument, a later `module.MIP_GAP = g`
    # would be silently ignored -- the exact trap THREADS is written to avoid.
    # MIP_GAP is read at call time too, and this test keeps it that way.
    prob = _RecordingVertexCover()
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    monkeypatch.setattr(er, "MIP_GAP", 0.05)
    er.get_neurprise_x(prob, _vc_instance(), np.array([0, 1]), problem_name="vertex_cover")
    assert prob.bigmilp_calls[0]["mip_gap"] == 0.05


def test_explicit_mip_gap_argument_still_wins_over_the_global(monkeypatch):
    # Motivation: callers that pass --mip-gap must not be overridden by a global.
    prob = _RecordingVertexCover()
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    monkeypatch.setattr(er, "MIP_GAP", 0.05)
    er.get_neurprise_x(prob, _vc_instance(), np.array([0, 1]),
                       mip_gap=0.001, problem_name="vertex_cover")
    assert prob.bigmilp_calls[0]["mip_gap"] == 0.001


# --------------------------------------------------------------------------
# Infeasibility semantics
# --------------------------------------------------------------------------

class _NoSolutionProblem(_RecordingVertexCover):
    def solve_robust_bigmilp(self, reduced, **kwargs):
        return {"obj": None, "x": None}


class _InfeasibleRecourseProblem(_RecordingVertexCover):
    def solve_recourse_mip(self, inst, s, x_star, threads=None, time_limit=None):
        return {"obj": 0.0, "has_incumbent": False}


def test_missing_reduced_solution_records_nan_not_inf(monkeypatch):
    # Motivation: "no measurement" (NaN) and "measured and infeasible" (+inf) are
    # different rows; this pins them apart.
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    out = er.true_regret_single(
        _NoSolutionProblem(), _vc_instance(), np.array([0, 1]),
        full_obj=100.0, n_scenarios=2, problem_name="vertex_cover")
    assert out["feasible"] is False
    assert np.isnan(out["eval_on_xi"])
    assert np.isnan(out["regret_pct"])


def test_infeasible_recourse_records_inf(monkeypatch):
    # Motivation: repo convention -- any scenario without an incumbent makes the
    # instance infeasible and it is excluded from feasible-only means.
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    out = er.true_regret_single(
        _InfeasibleRecourseProblem(), _vc_instance(), np.array([0, 1]),
        full_obj=100.0, n_scenarios=2, problem_name="vertex_cover")
    assert out["feasible"] is False
    assert out["eval_on_xi"] == float("inf")


def test_feasible_path_returns_the_percent_formula(monkeypatch):
    # Motivation: end-to-end through the wrapper -- first-stage cost 1*1+2*0 = 1,
    # worst recourse 1.0, so eval_on_xi = 2.0 against full_obj 1.6 -> +25%.
    monkeypatch.setattr(
        "src.utils.batch_utils._build_reduced_instance_from_np",
        lambda p, d, idx: SimpleNamespace(data=d))
    out = er.true_regret_single(
        _RecordingVertexCover(), _vc_instance(), np.array([0, 1]),
        full_obj=1.6, n_scenarios=2, problem_name="vertex_cover")
    assert out["feasible"] is True
    assert out["eval_on_xi"] == pytest.approx(2.0)
    assert out["regret_pct"] == pytest.approx(25.0)


# --------------------------------------------------------------------------
# The library and the wrapper must not drift apart
# --------------------------------------------------------------------------

def test_wrapper_reads_globals_at_call_time_as_the_guard_requires():
    # Motivation: configure_true_regret_solver refuses to run unless these two
    # functions reference THREADS/TIME_LIMIT in their own code objects. If a
    # future edit turns them into default arguments the guard raises at runtime;
    # this test catches it in CI instead.
    assert "THREADS" in er.get_neurprise_x.__code__.co_names
    assert "TIME_LIMIT" in er.get_neurprise_x.__code__.co_names
    assert "THREADS" in er.evaluate_x_on_all_scenarios.__code__.co_names


def test_wrapper_globals_live_in_the_script_module_the_callers_patch():
    # Motivation: the nine call sites patch `scripts.eval.eval_regret`.
    # If the globals ever move to src/eval/regret.py, those assignments go to a
    # module the functions no longer read -- silently. Assert the ownership.
    assert er.get_neurprise_x.__globals__ is vars(er)
    assert er.evaluate_x_on_all_scenarios.__globals__ is vars(er)
