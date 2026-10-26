import unittest
import numpy as np
from unittest.mock import MagicMock, patch, call
from src.core.types import Instance, ForwardTrace

# Import the new framework
from src.heuristics.search_framework import Incremental2ROSolver, StepwiseGreedySearch
# Import refactored functions from eval
from src.heuristics import eval as eval_module


class TestIncrementalSolver(unittest.TestCase):
    """
    Tier 1: Test the wrapper around Gurobi to ensure safe model management.
    """

    def setUp(self):
        self.mock_problem = MagicMock()
        # Mock the Gurobi model returned by build_base_model
        self.mock_master_model = MagicMock()
        self.mock_problem.build_base_model.return_value = self.mock_master_model

        self.mock_instance = MagicMock(spec=Instance)

        self.solver_wrapper = Incremental2ROSolver(
            self.mock_problem,
            self.mock_instance,
            time_limit=10,
            threads=1,
            verbose=False
        )

    def test_evaluate_subset_uses_copy(self):
        """Ensure evaluate_subset operates on a model copy, not the master."""
        # Setup copy behavior
        mock_copy_model = MagicMock()
        mock_copy_model.Status = 2  # GRB.OPTIMAL
        mock_copy_model.SolCount = 1
        mock_copy_model.ObjVal = 123.0
        mock_copy_model.ObjBound = 100.0
        mock_copy_model.MIPGap = 0.23

        self.mock_master_model.copy.return_value = mock_copy_model

        # Action
        indices = [1, 5]
        res = self.solver_wrapper.evaluate_subset(indices)

        # Assertions
        self.mock_master_model.copy.assert_called_once()
        # Ensure add_scenario was called on the COPY, not the master
        self.mock_problem.add_scenario_to_model.assert_has_calls([
            call(mock_copy_model, self.mock_instance, 1),
            call(mock_copy_model, self.mock_instance, 5)
        ])
        self.assertEqual(res, (123.0, 100.0, 0.23, True))

    def test_commit_subset_modifies_master(self):
        """Ensure commit_subset adds constraints directly to the master model."""
        indices = [2]
        self.solver_wrapper.commit_subset(indices)

        # Should NOT copy
        self.mock_master_model.copy.assert_not_called()
        # Should modify master
        self.mock_problem.add_scenario_to_model.assert_called_with(
            self.mock_master_model, self.mock_instance, 2
        )
        self.mock_master_model.update.assert_called()

    def test_cleanup_disposes_master(self):
        self.solver_wrapper.cleanup()
        self.mock_master_model.dispose.assert_called_once()


class TestStepwiseSearch(unittest.TestCase):
    """
    Tier 2: Test the search logic (Greedy selection, Plateau detection, Trace building).
    """

    def setUp(self):
        self.mock_solver = MagicMock(spec=Incremental2ROSolver)
        self.S = 4
        self.searcher = StepwiseGreedySearch(
            self.mock_solver, S=self.S, stop_M=2, eps_gain=1.0, log_prefix="TEST"
        )

    def test_search_selects_best_candidate(self):
        """Verify the loop picks the best score and commits it."""

        # S=4. Candidates [0,1,2,3].
        # Mock evaluate_subset to return scores based on input index
        # Let's say scenario 2 is best (score 50), then 0 (score 10), etc.

        def mock_eval(indices):
            idx = indices[0]
            if idx == 2: return (50.0, 0, 0, True)
            if idx == 0: return (10.0, 0, 0, True)
            return (0.0, 0, 0, True)

        self.mock_solver.evaluate_subset.side_effect = mock_eval

        # Run 1 step
        trace = self.searcher.search(K=1)

        # Assertions
        self.assertEqual(trace.selected_idx[0], 2)
        self.assertEqual(trace.best_scores[0], 50.0)

        # Verify it committed the winner
        self.mock_solver.commit_subset.assert_called_with([2])

    def test_plateau_stopping_logic(self):
        """
        Verify search stops when gain <= eps_gain for stop_M consecutive steps.
        eps_gain = 1.0, stop_M = 2
        """
        # Motivation: ensure selected_idx is masked to -1 at and after stop_step.
        # Step 1: Best Score 100.0 (Gain 100) -> OK
        # Step 2: Best Score 100.5 (Gain 0.5 <= 1.0) -> Plateau Count 1
        # Step 3: Best Score 101.0 (Gain 0.5 <= 1.0) -> Plateau Count 2 -> STOP

        # We simulate the _evaluate_candidates return values for 3 steps
        # Returns: (best_j, best_score, avg_gap)
        with patch.object(self.searcher, '_evaluate_candidates') as mock_eval:
            mock_eval.side_effect = [
                (0, 100.0, 0.0),  # Step 1
                (1, 100.5, 0.0),  # Step 2
                (2, 101.0, 0.0),  # Step 3 (Trigger stop)
                (3, 200.0, 0.0)  # Step 4 (Should not be reached)
            ]

            trace = self.searcher.search(K=5)

            self.assertEqual(len(trace.selected_idx), 5)
            # Stop step should be 2 (0-indexed), so steps >= 2 are masked.
            self.assertEqual(trace.selected_idx[:2].tolist(), [0, 1])
            self.assertEqual(trace.selected_idx[2:].tolist(), [-1, -1, -1])

            # Stop step should be 2 (0-indexed, so it points to the 3rd element as the stop point)
            # Actually, logic says stop_step is the index where condition met.
            # At step 3 (idx 2), plateau=2 >= 2. So stop_step=2.
            self.assertEqual(trace.stop_step, 2)

            # Mask should be True for 0, 1. False for 2, 3, 4.
            self.assertEqual(trace.step_mask.astype(int).tolist(), [1, 1, 0, 0, 0])


class TestEvalIntegration(unittest.TestCase):
    """
    Tier 3: Integration tests crossing eval.py and search_framework.py
    """

    def setUp(self):
        self.mock_problem = MagicMock()
        # Ensure these attributes exist so the check in eval.py passes
        self.mock_problem.build_base_model = MagicMock()
        self.mock_problem.add_scenario_to_model = MagicMock()
        self.mock_problem.get_scenarios = MagicMock()

        self.dummy_data = {
            "sce": np.random.rand(5, 10),  # 5 scenarios
            "det": np.random.rand(10),
            "adjacency": np.zeros((10, 10))
        }
        self.mock_problem.get_scenarios.return_value = self.dummy_data["sce"]

    @patch('src.heuristics.eval.StepwiseGreedySearch')
    @patch('src.heuristics.eval.Incremental2ROSolver')
    def test_forward_with_trace_wiring(self, MockSolver, MockSearcher):
        """
        Verify _forward_with_trace correctly instantiates Solver and Searcher
        and returns the trace from searcher.
        """
        # Setup
        mock_search_instance = MockSearcher.return_value
        expected_trace = MagicMock(spec=ForwardTrace)
        mock_search_instance.search.return_value = expected_trace

        # Execute
        result = eval_module._forward_with_trace(
            self.mock_problem,
            self.dummy_data,
            full_obj_val=0.0,
            K=3,
            milp_time_limit=5.0,
            stop_M=2,
            eps_gain=0.01
        )

        # Assertions
        # 1. Solver init
        MockSolver.assert_called_once()
        _, kwargs = MockSolver.call_args
        self.assertEqual(kwargs['time_limit'], 5.0)

        # 2. Searcher init
        MockSearcher.assert_called_once()
        _, kwargs = MockSearcher.call_args
        self.assertEqual(kwargs['solver'], MockSolver.return_value)
        self.assertEqual(kwargs['S'], 5)
        self.assertEqual(kwargs['stop_M'], 2)

        # 3. Execution
        mock_search_instance.search.assert_called_with(3)
        self.assertEqual(result, expected_trace)

        # 4. Cleanup
        MockSolver.return_value.cleanup.assert_called_once()




if __name__ == '__main__':
    unittest.main()
