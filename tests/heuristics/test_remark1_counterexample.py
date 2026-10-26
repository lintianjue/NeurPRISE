"""
Unit tests for the Remark 1 counterexample (non-submodularity of v).

Two counterexamples are tested:

1. **Paper counterexample** (3 decisions, 3 scenarios):
   All PRISE gains are strictly positive (5, 1, 2), so default early stopping
   (stop_M=1, eps_gain=0.0001) never triggers. The gains increase from step 2
   to step 3 (1 → 2), violating submodularity. PRISE recovers the optimal
   v(full) = 8.

2. **Old counterexample** (2 decisions, 3 scenarios):
   Regression test for the original 2-decision example where gain=0 at step 2
   causes early stopping. Kept to verify stopping logic.
"""

import numpy as np
import pytest

from src.heuristics.search_framework import StepwiseGreedySearch
from src.utils.eval_utils import _compute_plateau_stop_step, _compute_step_mask
import src.heuristics.eval as eval_mod


# ======================================================================
# Paper counterexample: 3 decisions {a, b, c}, 3 scenarios {s1, s2, s3}
#
# f(x, s)    s1   s2   s3
# ─────────────────────────
#   a         9    1    5
#   b         1    9    6
#   c         4    4    8
#
# v(R) = min_{x in {a,b,c}} max_{s in R} f(x,s)
# ======================================================================

COST_TABLE_PAPER = {
    "a": {0: 9, 1: 1, 2: 5},   # s1=idx0, s2=idx1, s3=idx2
    "b": {0: 1, 1: 9, 2: 6},
    "c": {0: 4, 1: 4, 2: 8},
}


def v_paper(scenario_indices):
    """Compute v(R) for the paper counterexample."""
    if not scenario_indices:
        return 0.0
    vals = []
    for x_label, costs in COST_TABLE_PAPER.items():
        vals.append(max(costs[s] for s in scenario_indices))
    return float(min(vals))


class PaperSolver:
    """Mock Incremental2ROSolver for the paper counterexample."""

    def __init__(self, *args, **kwargs):
        self.committed = []

    def evaluate_subset(self, candidate_indices):
        all_indices = self.committed + list(candidate_indices)
        obj = v_paper(all_indices)
        return obj, obj, 0.0, True

    def commit_subset(self, indices):
        self.committed.extend(indices)

    def cleanup(self):
        pass


# ---- v(R) sanity checks ----

class TestVPaper:
    def test_singletons(self):
        assert v_paper([0]) == 1.0    # v({s1}) = min(9, 1, 4) = 1
        assert v_paper([1]) == 1.0    # v({s2}) = min(1, 9, 4) = 1
        assert v_paper([2]) == 5.0    # v({s3}) = min(5, 6, 8) = 5

    def test_pairs(self):
        assert v_paper([0, 1]) == 4.0   # v({s1,s2}) = min(9, 9, 4) = 4
        assert v_paper([0, 2]) == 6.0   # v({s1,s3}) = min(9, 6, 8) = 6
        assert v_paper([1, 2]) == 5.0   # v({s2,s3}) = min(5, 9, 8) = 5

    def test_full(self):
        assert v_paper([0, 1, 2]) == 8.0  # v({s1,s2,s3}) = min(9, 9, 8) = 8


# ---- PRISE trace on paper counterexample ----

class TestPaperCounterexample:

    def test_prise_trace_all_gains_positive(self):
        """PRISE trace: s3(gain=5) → s1(gain=1) → s2(gain=2). No early stop."""
        solver = PaperSolver()
        searcher = StepwiseGreedySearch(
            solver=solver,
            S=3,
            stop_M=1,
            eps_gain=0.0001,
            log_prefix="[Paper] ",
        )
        trace = searcher.search(K=3)

        # Step 0: s3 (idx=2), score=5, gain=5
        assert trace.selected_idx[0] == 2
        assert trace.best_scores[0] == pytest.approx(5.0)
        assert trace.gains[0] == pytest.approx(5.0)

        # Step 1: s1 (idx=0), score=6, gain=1
        assert trace.selected_idx[1] == 0
        assert trace.best_scores[1] == pytest.approx(6.0)
        assert trace.gains[1] == pytest.approx(1.0)

        # Step 2: s2 (idx=1), score=8, gain=2
        assert trace.selected_idx[2] == 1
        assert trace.best_scores[2] == pytest.approx(8.0)
        assert trace.gains[2] == pytest.approx(2.0)

    def test_no_early_stop(self):
        """Default params: all gains > eps_gain, so no early stopping."""
        solver = PaperSolver()
        searcher = StepwiseGreedySearch(
            solver=solver,
            S=3,
            stop_M=1,
            eps_gain=0.0001,
            log_prefix="[Paper] ",
        )
        trace = searcher.search(K=3)

        assert trace.stop_step == 3
        assert trace.step_mask.tolist() == [True, True, True]

    def test_recovers_optimal(self):
        """PRISE selects all 3 scenarios and v(R_3) = v(full) = 8."""
        solver = PaperSolver()
        searcher = StepwiseGreedySearch(
            solver=solver,
            S=3,
            stop_M=1,
            eps_gain=0.0001,
            log_prefix="[Paper] ",
        )
        trace = searcher.search(K=3)

        active_idx = trace.selected_idx[trace.step_mask]
        active_idx = active_idx[active_idx >= 0]
        assert set(active_idx.tolist()) == {0, 1, 2}
        assert trace.best_scores[-1] == pytest.approx(8.0)
        assert v_paper([0, 1, 2]) == pytest.approx(8.0)

    def test_submodularity_violation(self):
        """
        Non-submodularity: marginal gain of s2 increases as base set grows.

        S = {s3}, T = {s3, s1}, e = s2:
          v({s3, s2}) - v({s3}) = 5 - 5 = 0
          v({s3, s1, s2}) - v({s3, s1}) = 8 - 6 = 2
          0 < 2  →  violation
        """
        gain_small = v_paper([2, 1]) - v_paper([2])          # 5 - 5 = 0
        gain_large = v_paper([2, 0, 1]) - v_paper([2, 0])    # 8 - 6 = 2

        assert gain_small == pytest.approx(0.0)
        assert gain_large == pytest.approx(2.0)
        assert gain_small < gain_large  # submodularity violated

    def test_gains_non_monotone(self):
        """PRISE gains are [5, 1, 2] — gain increases from step 2 to step 3."""
        solver = PaperSolver()
        searcher = StepwiseGreedySearch(
            solver=solver,
            S=3,
            stop_M=1,
            eps_gain=0.0001,
            log_prefix="[Paper] ",
        )
        trace = searcher.search(K=3)

        gains = trace.gains.tolist()
        assert gains[0] == pytest.approx(5.0)
        assert gains[1] == pytest.approx(1.0)
        assert gains[2] == pytest.approx(2.0)
        # Gain increases from step 1 to step 2 (non-monotone)
        assert gains[1] < gains[2]

    def test_forward_with_trace_integration(self, monkeypatch):
        """Integration: _forward_with_trace with paper counterexample."""
        monkeypatch.setattr(eval_mod, "Incremental2ROSolver", PaperSolver)

        data_np = {"sce": np.zeros((3, 1))}
        trace = eval_mod._forward_with_trace(
            problem=_DummyProblem(),
            data_np=data_np,
            full_obj_val=8.0,
            K=3,
            milp_time_limit=None,
        )
        assert trace.stop_step == 3  # no early stop
        assert all(trace.step_mask)
        assert trace.best_scores[-1] == pytest.approx(8.0)


# ---- _compute_plateau_stop_step on paper gains ----

class TestPlateauStopPaper:

    def test_paper_gains_no_stop(self):
        """best_scores=[5, 6, 8] with stop_M=1, eps=0.0001 never stops."""
        stop = _compute_plateau_stop_step(
            [5.0, 6.0, 8.0], stop_M=1, eps_gain=0.0001
        )
        assert stop is None


# ======================================================================
# Old counterexample (regression): 2 decisions, 3 scenarios
# f(a,s) = [0, 10, 6], f(b,s) = [10, 0, 6]
# PRISE: s3(6) → s1(0) → s2(4) — gain=0 triggers early stop
# ======================================================================

COST_TABLE_OLD = {
    "a": {0: 0, 1: 10, 2: 6},
    "b": {0: 10, 1: 0, 2: 6},
}


def v_old(scenario_indices):
    """Compute v(R) for the old 2-decision counterexample."""
    if not scenario_indices:
        return 0.0
    vals = []
    for x_label, costs in COST_TABLE_OLD.items():
        vals.append(max(costs[s] for s in scenario_indices))
    return float(min(vals))


class OldSolver:
    """Mock solver for old counterexample."""

    def __init__(self, *args, **kwargs):
        self.committed = []

    def evaluate_subset(self, candidate_indices):
        all_indices = self.committed + list(candidate_indices)
        obj = v_old(all_indices)
        return obj, obj, 0.0, True

    def commit_subset(self, indices):
        self.committed.extend(indices)

    def cleanup(self):
        pass


class TestOldCounterexample:
    """Regression: old example where gain=0 causes early stop."""

    def test_v_old_values(self):
        assert v_old([0]) == 0.0
        assert v_old([1]) == 0.0
        assert v_old([2]) == 6.0
        assert v_old([2, 0]) == 6.0
        assert v_old([2, 1]) == 6.0
        assert v_old([0, 1, 2]) == 10.0

    def test_early_stop_on_zero_gain(self):
        """Default params stop at step 1 due to gain=0."""
        solver = OldSolver()
        searcher = StepwiseGreedySearch(
            solver=solver, S=3, stop_M=1, eps_gain=0.0001,
            log_prefix="[Old] ",
        )
        trace = searcher.search(K=3)

        assert trace.stop_step == 1
        assert trace.step_mask.tolist() == [True, False, False]
        active_idx = trace.selected_idx[trace.step_mask]
        assert len(active_idx) == 1
        assert active_idx[0] == 2  # only s3

    def test_eps_zero_still_stops(self):
        """eps_gain=0.0 still triggers on gain=0 (condition is <=)."""
        solver = OldSolver()
        searcher = StepwiseGreedySearch(
            solver=solver, S=3, stop_M=1, eps_gain=0.0,
            log_prefix="[Old-eps0] ",
        )
        trace = searcher.search(K=3)
        assert trace.stop_step == 1

    def test_stop_M2_survives(self):
        """stop_M=2 survives a single zero-gain step."""
        solver = OldSolver()
        searcher = StepwiseGreedySearch(
            solver=solver, S=3, stop_M=2, eps_gain=0.0001,
            log_prefix="[Old-M2] ",
        )
        trace = searcher.search(K=3)
        assert trace.stop_step == 3
        assert trace.best_scores[2] == pytest.approx(10.0)

    def test_plateau_stop_step_old(self):
        """_compute_plateau_stop_step on [6, 6, 10]."""
        assert _compute_plateau_stop_step([6.0, 6.0, 10.0], stop_M=1, eps_gain=0.0001) == 1
        assert _compute_plateau_stop_step([6.0, 6.0, 10.0], stop_M=2, eps_gain=0.0001) is None


# ---- Helper ----

class _DummyProblem:
    def get_scenarios(self, data_np):
        return data_np["sce"]

    def build_base_model(self, *args, **kwargs):
        pass

    def add_scenario_to_model(self, *args, **kwargs):
        pass
