import numpy as np
import pytest
import src.heuristics.eval as eval_mod

class DummyProblem:
    def get_scenarios(self, data_np):
        if "sce" in data_np:
            return data_np["sce"]
        if "scenarios" in data_np:
            return data_np["scenarios"]
        raise KeyError("No scenarios found in data_np")

class MockIncrementalSolver:
    """
    Replaces Incremental2ROSolver for testing.
    Implements the simple deterministic objective:
      reduced_obj = sum of scenario indices + 1.0
    """

    def __init__(self, problem, instance, *args, **kwargs):
        self.problem = problem
        self.instance = instance
        self.committed_indices = []

    def evaluate_subset(self, candidate_indices):
        # Combine currently committed indices with the new candidates
        all_indices = self.committed_indices + list(candidate_indices)

        # Logic: reduced_obj = sum(indices) + 1.0
        if not all_indices:
            red_obj = 1.0
        else:
            red_obj = float(np.sum(all_indices) + 1.0)

        # Returns: (obj, bound, gap, feasible)
        return red_obj, 90.0, 0.0, True

    def commit_subset(self, indices):
        self.committed_indices.extend(indices)

    def cleanup(self):
        pass

@pytest.fixture(autouse=True)
def patch_solver(monkeypatch):
    """
    Automatically patch Incremental2ROSolver in eval module
    to use MockIncrementalSolver for ALL tests in this file.
    """
    monkeypatch.setattr(eval_mod, "Incremental2ROSolver", MockIncrementalSolver)

def test_compute_effective_K_uses_at_least_one(monkeypatch):
    # override _compute_K_from_ratio to return 0 to check the clamp
    monkeypatch.setattr(eval_mod, "_compute_K_from_ratio", lambda S, r: 0)
    assert eval_mod._compute_effective_K(S=10, k_ratio=0.0) == 1

def test_forward_with_step_margins_selects_largest_sum(monkeypatch):
    """
    Check that forward greedy selects the subset with the largest cumulative sum.
    Uses the MockIncrementalSolver via the autouse fixture.
    """
    # S = 4 scenarios, indices {0,1,2,3}
    data_np = {"sce": np.zeros((4, 1))}
    full_obj_val = 0.0
    K = 3

    trace = eval_mod._forward_with_trace(
        problem=DummyProblem(),
        data_np=data_np,
        full_obj_val=full_obj_val,
        K=K,
        milp_time_limit=None,
        log_prefix="[TEST] ",
    )

    selected_idx = trace.selected_idx
    step_scores = trace.best_scores

    # Step 1: candidates 0,1,2,3. Max is 3 (obj 4.0). Selected=[3]
    # Step 2: candidates 0,1,2. Max is 2 (obj sum(3,2)+1 = 6.0). Selected=[3,2]
    # Step 3: candidates 0,1. Max is 1 (obj sum(3,2,1)+1 = 7.0). Selected=[3,2,1]

    assert selected_idx.tolist() == [3, 2, 1]
    assert step_scores.shape == (K,)
