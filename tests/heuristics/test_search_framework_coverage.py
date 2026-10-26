"""
Additional coverage tests for search_framework.py.

Targets uncovered lines: 31, 69, 128-130, 185, 188, 222, 226, 230-232.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.core.types import Instance, ForwardTrace
from src.heuristics.search_framework import Incremental2ROSolver, StepwiseGreedySearch


# ---------------------------------------------------------------------------
# Incremental2ROSolver tests
# ---------------------------------------------------------------------------

class TestSolverNotImplemented(unittest.TestCase):
    """Line 31: raise NotImplementedError when problem lacks required methods."""

    def test_missing_build_base_model(self):
        problem = MagicMock(spec=[])  # no attributes
        inst = MagicMock(spec=Instance)
        with self.assertRaises(NotImplementedError):
            Incremental2ROSolver(problem, inst, time_limit=1, threads=1, verbose=False)

    def test_missing_add_scenario_to_model(self):
        problem = MagicMock(spec=["build_base_model"])  # has build but not add
        inst = MagicMock(spec=Instance)
        with self.assertRaises(NotImplementedError):
            Incremental2ROSolver(problem, inst, time_limit=1, threads=1, verbose=False)


class TestSolverInfeasible(unittest.TestCase):
    """Line 69: evaluate_subset returns inf when model is infeasible."""

    def setUp(self):
        self.mock_problem = MagicMock()
        self.mock_master = MagicMock()
        self.mock_problem.build_base_model.return_value = self.mock_master
        self.solver = Incremental2ROSolver(
            self.mock_problem, MagicMock(spec=Instance),
            time_limit=1, threads=1, verbose=False,
        )

    def test_infeasible_returns_inf_tuple(self):
        mock_copy = MagicMock()
        mock_copy.Status = 3  # GRB.INFEASIBLE
        mock_copy.SolCount = 0

        with patch("src.heuristics.search_framework.copy_model_with_payload",
                    return_value=mock_copy):
            result = self.solver.evaluate_subset([0])

        self.assertEqual(result, (float("inf"), 0.0, 0.0, False))

    def test_time_limit_no_solution(self):
        """Status TIME_LIMIT but SolCount=0 → infeasible path."""
        mock_copy = MagicMock()
        mock_copy.Status = 9  # GRB.TIME_LIMIT
        mock_copy.SolCount = 0

        with patch("src.heuristics.search_framework.copy_model_with_payload",
                    return_value=mock_copy):
            result = self.solver.evaluate_subset([0])

        self.assertEqual(result, (float("inf"), 0.0, 0.0, False))


class TestSolverCommitAndCleanup(unittest.TestCase):
    """Lines covering commit_subset and cleanup edge cases."""

    def setUp(self):
        self.mock_problem = MagicMock()
        self.mock_master = MagicMock()
        self.mock_problem.build_base_model.return_value = self.mock_master
        self.solver = Incremental2ROSolver(
            self.mock_problem, MagicMock(spec=Instance),
            time_limit=1, threads=1, verbose=False,
        )

    def test_commit_multiple_scenarios(self):
        # __init__ already calls update once; reset to isolate commit_subset
        self.mock_master.update.reset_mock()
        self.solver.commit_subset([0, 1, 2])
        self.assertEqual(self.mock_problem.add_scenario_to_model.call_count, 3)
        self.mock_master.update.assert_called_once()

    def test_cleanup_sets_none(self):
        self.solver.cleanup()
        self.assertIsNone(self.solver.master_model)
        self.mock_master.dispose.assert_called_once()

    def test_cleanup_when_already_none(self):
        """Calling cleanup twice should not error."""
        self.solver.cleanup()
        self.solver.cleanup()  # master_model is None → no dispose


# ---------------------------------------------------------------------------
# StepwiseGreedySearch tests
# ---------------------------------------------------------------------------

class TestSearchK1(unittest.TestCase):
    """K=1 edge case: single step search."""

    def setUp(self):
        self.mock_solver = MagicMock(spec=Incremental2ROSolver)
        self.S = 3

    def test_k1_normal(self):
        searcher = StepwiseGreedySearch(
            self.mock_solver, S=self.S, stop_M=2, eps_gain=0.001
        )

        def mock_eval(indices):
            idx = indices[0]
            scores = {0: 10.0, 1: 30.0, 2: 20.0}
            return (scores.get(idx, 0.0), 0.0, 0.0, True)

        self.mock_solver.evaluate_subset.side_effect = mock_eval
        trace = searcher.search(K=1)

        self.assertIsInstance(trace, ForwardTrace)
        self.assertEqual(trace.K, 1)
        self.assertEqual(trace.S, self.S)
        self.assertEqual(trace.selected_idx[0], 1)  # scenario 1 has best score
        self.assertEqual(trace.best_scores[0], 30.0)


class TestSearchEarlyStopping(unittest.TestCase):
    """Lines 128-130: _evaluate_candidates returns None → early break."""

    def setUp(self):
        self.mock_solver = MagicMock(spec=Incremental2ROSolver)
        self.S = 2

    def test_no_candidates_triggers_fill_remaining(self):
        """When _evaluate_candidates returns None, search fills dummy and breaks."""
        searcher = StepwiseGreedySearch(
            self.mock_solver, S=self.S, stop_M=2, eps_gain=0.001
        )

        with patch.object(searcher, "_evaluate_candidates") as mock_eval:
            mock_eval.return_value = None  # no feasible candidates at step 1
            trace = searcher.search(K=3)

        # All steps should be dummy-filled
        self.assertEqual(len(trace.selected_idx), 3)
        np.testing.assert_array_equal(trace.selected_idx, [-1, -1, -1])

    def test_candidates_exhausted_mid_search(self):
        """Step 1 works, step 2 returns None → remaining filled with dummies."""
        searcher = StepwiseGreedySearch(
            self.mock_solver, S=self.S, stop_M=2, eps_gain=0.001
        )

        with patch.object(searcher, "_evaluate_candidates") as mock_eval:
            mock_eval.side_effect = [
                (0, 50.0, 0.0),  # Step 1: valid
                None,            # Step 2: no candidates
            ]
            trace = searcher.search(K=4)

        self.assertEqual(len(trace.selected_idx), 4)
        self.assertEqual(trace.selected_idx[0], 0)
        # Steps 2-4 should be dummies
        np.testing.assert_array_equal(trace.selected_idx[1:], [-1, -1, -1])


class TestEvaluateCandidatesInfeasible(unittest.TestCase):
    """Lines 185, 188: infeasible candidates in _evaluate_candidates."""

    def test_all_infeasible_returns_none(self):
        """Line 188: when all candidates are infeasible, return None."""
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        mock_solver.evaluate_subset.return_value = (0.0, 0.0, 0.0, False)

        searcher = StepwiseGreedySearch(mock_solver, S=3, stop_M=1, eps_gain=0.001)
        result = searcher._evaluate_candidates(t=1, K=5)

        self.assertIsNone(result)

    def test_some_infeasible_skipped(self):
        """Line 185: infeasible candidates get gap=0.0, score stays -inf."""
        mock_solver = MagicMock(spec=Incremental2ROSolver)

        def mock_eval(indices):
            idx = indices[0]
            if idx == 1:
                return (42.0, 0.0, 0.05, True)
            return (0.0, 0.0, 0.0, False)  # infeasible

        mock_solver.evaluate_subset.side_effect = mock_eval

        searcher = StepwiseGreedySearch(mock_solver, S=3, stop_M=1, eps_gain=0.001)
        result = searcher._evaluate_candidates(t=1, K=5)

        self.assertIsNotNone(result)
        best_j, best_score, avg_gap = result
        self.assertEqual(best_j, 1)
        self.assertEqual(best_score, 42.0)


class TestCheckStoppingCondition(unittest.TestCase):
    """_check_stopping_condition: plateau triggers stop, no-plateau continues."""

    def test_plateau_triggers_stop(self):
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        searcher = StepwiseGreedySearch(
            mock_solver, S=3, stop_M=2, eps_gain=1.0
        )
        searcher.best_scores_list = [100.0, 100.1, 100.2]

        # Two consecutive small gains → should trigger
        searcher._check_stopping_condition(t=2, gain=0.1)
        self.assertEqual(searcher.plateau_count, 1)
        self.assertIsNone(searcher.stop_step)

        searcher._check_stopping_condition(t=3, gain=0.1)
        self.assertEqual(searcher.plateau_count, 2)
        self.assertIsNotNone(searcher.stop_step)

    def test_large_gain_resets_plateau(self):
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        searcher = StepwiseGreedySearch(
            mock_solver, S=3, stop_M=2, eps_gain=1.0
        )
        searcher.best_scores_list = [100.0, 100.1, 200.0]

        searcher._check_stopping_condition(t=2, gain=0.1)
        self.assertEqual(searcher.plateau_count, 1)

        searcher._check_stopping_condition(t=3, gain=99.9)
        self.assertEqual(searcher.plateau_count, 0)
        self.assertIsNone(searcher.stop_step)


class TestRecordDummyStepEmpty(unittest.TestCase):
    """Lines 222, 226: _record_dummy_step when lists are empty."""

    def test_dummy_step_empty_lists(self):
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        searcher = StepwiseGreedySearch(mock_solver, S=2, stop_M=1, eps_gain=0.001)

        # Both lists are empty
        self.assertEqual(len(searcher.best_scores_list), 0)
        self.assertEqual(len(searcher.step_cumulative_time_sec), 0)

        searcher._record_dummy_step()

        # Line 222: appends 0.0 when best_scores_list was empty
        self.assertEqual(searcher.best_scores_list, [0.0])
        # Line 226: appends 0.0 when step_cumulative_time_sec was empty
        self.assertEqual(searcher.step_cumulative_time_sec, [0.0])
        # selected gets -1
        self.assertEqual(searcher.selected, [-1])
        # candidate_scores_steps gets a dummy array
        self.assertEqual(len(searcher.candidate_scores_steps), 1)
        np.testing.assert_array_equal(
            searcher.candidate_scores_steps[0], np.full((2,), -1.0)
        )


class TestFillRemainingDummy(unittest.TestCase):
    """Lines 230-232: _fill_remaining_dummy fills from current position."""

    def test_fills_correct_count(self):
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        searcher = StepwiseGreedySearch(mock_solver, S=2, stop_M=1, eps_gain=0.001)

        # Simulate 1 step already done
        searcher.selected = [0]
        searcher.best_scores_list = [10.0]
        searcher.candidate_scores_steps = [np.array([10.0, 5.0])]
        searcher.step_cumulative_time_sec = [1.0]

        searcher._fill_remaining_dummy(K=4, current_t=2)

        # Should have added 3 dummy steps (4 - 1 existing)
        self.assertEqual(len(searcher.selected), 4)
        self.assertEqual(searcher.selected[1:], [-1, -1, -1])
        self.assertEqual(len(searcher.best_scores_list), 4)
        # Dummy scores should repeat last valid
        self.assertEqual(searcher.best_scores_list[1:], [10.0, 10.0, 10.0])


class TestSearchNormalCompletion(unittest.TestCase):
    """Full K-step search without early stopping or plateau."""

    def test_completes_all_steps(self):
        mock_solver = MagicMock(spec=Incremental2ROSolver)
        S = 4
        searcher = StepwiseGreedySearch(
            mock_solver, S=S, stop_M=10, eps_gain=0.001  # high stop_M = no plateau
        )

        call_count = [0]

        def mock_eval(indices):
            idx = indices[0]
            call_count[0] += 1
            # Each step, the score increases by 10 * (idx+1)
            return (float(10 * (idx + 1)), 0.0, 0.01, True)

        mock_solver.evaluate_subset.side_effect = mock_eval

        trace = searcher.search(K=3)

        self.assertEqual(trace.K, 3)
        self.assertEqual(len(trace.selected_idx), 3)
        # All steps should have valid selections (not -1)
        self.assertTrue(all(s >= 0 for s in trace.selected_idx))
        # stop_step should be T (no early stop)
        self.assertEqual(trace.stop_step, 3)
        # step_mask all True
        np.testing.assert_array_equal(trace.step_mask, [True, True, True])


if __name__ == "__main__":
    unittest.main()
