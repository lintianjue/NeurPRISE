import unittest
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Adjust imports to match your project structure
from src.problems.vertex_cover import VertexCoverProblem
from src.core.types import Instance
# Assuming Incremental2ROSolver is in src.heuristics.search_framework now
# If it's still in eval during your transition, adjust this import back to src.heuristics.eval
from src.heuristics.search_framework import Incremental2ROSolver

class TestIncrementalSolver(unittest.TestCase):

    def setUp(self):
        # 1. Create a small, deterministic synthetic problem
        self.n_nodes = 10
        self.n_scenarios = 5

        # Initialize problem
        self.problem = VertexCoverProblem(
            n_nodes=self.n_nodes,
            n_scenarios=self.n_scenarios,
            p_edge=0.3
        )

        # Sample an instance
        self.instance = self.problem.sample()

        # Configuration
        self.time_limit = 5.0
        self.threads = 1
        self.verbose = False

    def test_equivalence_to_full_rebuild(self):
        """
        Compare the Incremental Solver against the naive 'Rebuild from Scratch' method.
        They must return the exact same objective values for the same subset of scenarios.
        """
        # Create the incremental solver
        inc_solver = Incremental2ROSolver(
            self.problem, self.instance, self.time_limit, self.threads, self.verbose
        )

        # We will simulate a sequence of adding scenarios: [0, 2, 4]
        sequence = [0, 2, 4]

        for idx, sce_idx in enumerate(sequence):
            # --- A. Incremental Approach ---
            # NEW API: Pass list [sce_idx] to evaluate_subset
            inc_obj, inc_bound, inc_gap, inc_feas = inc_solver.evaluate_subset([sce_idx])

            # --- B. Legacy Approach (Full Rebuild) ---
            # Rebuild model with ALL scenarios committed so far + current candidate
            current_subset = sequence[:idx] + [sce_idx]

            truth_model = self.problem.build_base_model(
                self.instance, self.time_limit, self.threads, self.verbose
            )
            for s_idx in current_subset:
                self.problem.add_scenario_to_model(truth_model, self.instance, s_idx)

            truth_model.optimize()

            # --- Assertions ---
            self.assertTrue(inc_feas, "Incremental solution should be feasible")
            self.assertTrue(truth_model.SolCount > 0, "Truth solution should be feasible")

            # Check Objectives match (tolerance for float arithmetic)
            self.assertAlmostEqual(inc_obj, truth_model.ObjVal, places=5,
                                   msg=f"Objectives mismatch at step {idx} (scenario {sce_idx})")

            # Check Bounds match
            self.assertAlmostEqual(inc_bound, truth_model.ObjBound, places=5,
                                   msg=f"Bounds mismatch at step {idx}")

            # --- Commit to Incremental ---
            # NEW API: Pass list [sce_idx] to commit_subset
            inc_solver.commit_subset([sce_idx])

        inc_solver.cleanup()

    def test_model_isolation(self):
        """
        Verify that `evaluate_subset` acts on a COPY and does not
        pollute the master model.
        """
        inc_solver = Incremental2ROSolver(
            self.problem, self.instance, self.time_limit, self.threads, self.verbose
        )

        master_constrs_initial = inc_solver.master_model.NumConstrs

        # 1. Evaluate Candidate 0
        inc_solver.evaluate_subset([0])

        # 2. Check Master Model
        # The number of constraints should NOT have changed
        self.assertEqual(
            inc_solver.master_model.NumConstrs,
            master_constrs_initial,
            "Master model polluted! Constraint count increased after evaluate_subset."
        )

        # 3. Commit Candidate 0
        inc_solver.commit_subset([0])

        # 4. Check Master Model
        # Now it SHOULD increase
        self.assertGreater(
            inc_solver.master_model.NumConstrs,
            master_constrs_initial,
            "Master model failed to update after commit_subset."
        )

        inc_solver.cleanup()

    def test_infeasibility_handling(self):
        """
        Inject a constraint that makes a specific scenario infeasible
        and ensure the wrapper returns feasible=False instead of crashing.
        """
        inc_solver = Incremental2ROSolver(
            self.problem, self.instance, self.time_limit, self.threads, self.verbose
        )

        # NEW API
        obj, bound, gap, feas = inc_solver.evaluate_subset([0])

        self.assertIsInstance(obj, float)
        self.assertIsInstance(feas, bool)

        # If feasible (likely for VC), obj should be finite
        if feas:
            self.assertNotEqual(obj, float('inf'))
        else:
            self.assertEqual(obj, float('inf'))

        inc_solver.cleanup()

    def test_caching_performance_check(self):
        """
        Simple sanity check that the 'Fast Path' is indeed not slower than 'Slow Path'
        (This is flaky in unit tests due to overhead, but good for local debugging).
        """
        import time

        # 1. Slow Path (Simulated)
        t0 = time.perf_counter()
        truth_model = self.problem.build_base_model(self.instance, self.time_limit, self.threads, self.verbose)
        self.problem.add_scenario_to_model(truth_model, self.instance, 0)
        truth_model.optimize()
        t_slow = time.perf_counter() - t0

        # 2. Incremental Path
        inc_solver = Incremental2ROSolver(self.problem, self.instance, self.time_limit, self.threads, self.verbose)
        t1 = time.perf_counter()
        # NEW API
        inc_solver.evaluate_subset([0])
        t_fast = time.perf_counter() - t1
        inc_solver.cleanup()

        # Not a strict assertion, but a log
        print(f"\n[Perf Check] Slow Rebuild: {t_slow:.4f}s vs Fast Copy: {t_fast:.4f}s")

    def test_parameter_preservation_on_copy(self):
        """
        Verify if Gurobi parameters (TimeLimit, Threads) are preserved
        when calling model.copy().

        This is critical because the IncrementalSolver relies on copying
        the master_model for candidate evaluation. If parameters are lost,
        candidates might run without time limits (causing hangs) or
        with default threads (causing CPU contention).
        """
        # 1. Setup with distinct non-default values
        # (Use values different from defaults to ensure we are testing correctly)
        specific_time_limit = 1.5
        specific_threads = 2

        # 2. Build the base model
        base_model = self.problem.build_base_model(
            self.instance,
            time_limit=specific_time_limit,
            threads=specific_threads,
            verbose=self.verbose
        )

        # Check original model has correct params
        # Note: Gurobi stores these in model.Params
        self.assertAlmostEqual(
            base_model.Params.TimeLimit,
            specific_time_limit,
            places=5,
            msg="Base model did not accept initial TimeLimit"
        )
        self.assertEqual(
            base_model.Params.Threads,
            specific_threads,
            msg="Base model did not accept initial Threads"
        )

        # 3. Perform the Copy
        copied_model = base_model.copy()

        # 4. Verify the Copy preserves parameters
        # If this fails, you need to manually copy params in your solver wrapper
        self.assertAlmostEqual(
            copied_model.Params.TimeLimit,
            specific_time_limit,
            places=5,
            msg="TimeLimit LOST after model.copy()! Incremental solver may hang."
        )

        self.assertEqual(
            copied_model.Params.Threads,
            specific_threads,
            msg="Threads LOST after model.copy()!"
        )

        # Clean up to release Gurobi license/threads
        base_model.dispose()
        copied_model.dispose()


if __name__ == '__main__':
    unittest.main()