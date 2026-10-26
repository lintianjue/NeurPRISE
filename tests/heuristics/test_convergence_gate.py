"""Contract tests for the convergence gate.

SCOPE, STATED HONESTLY. These are CONTRACT tests -- defaults, counter surface, and the
documented semantics. They are not the behavioural proof, because exercising the retry path
requires a real Gurobi solve that hits a time limit, which is far too slow for a suite.

The behavioural proof is an end-to-end run: the same greedy over the same instances, gate
off then gate on, 68/601 impossible decreases before and 0/469 after.
"""
from __future__ import annotations

import inspect

from src.heuristics.search_framework import Incremental2ROSolver


def test_gate_is_on_by_default() -> None:
    """A default-off gate is not a gate. This is the whole point of the parameter's default."""
    sig = inspect.signature(Incremental2ROSolver.__init__)
    assert sig.parameters["require_converged"].default is True


def test_retry_limit_is_bounded_by_default() -> None:
    """An unbounded retry turns one pathological subproblem into a hung labelling run."""
    sig = inspect.signature(Incremental2ROSolver.__init__)
    limit = sig.parameters["unconverged_retry_time_limit"].default
    assert limit is not None and limit > 0


def test_counters_are_declared_on_the_class_surface() -> None:
    """A repair whose incidence nobody can read is a repair nobody can audit."""
    src = inspect.getsource(Incremental2ROSolver.__init__)
    for counter in ("n_unconverged_retries", "n_retry_rescued", "n_excluded_unconverged"):
        assert f"self.{counter} = 0" in src


def test_gate_keys_on_OPTIMAL_not_on_a_gap_threshold() -> None:
    """The arm's OWN tolerance defines convergence.

    With mip_gap=0.05 a cheap arm's GRB.OPTIMAL is a 5%-gap answer -- its declared
    approximation, not a defect. Keying the gate on an absolute gap would silently re-solve
    the cheap arms to exactness and erase the knob the whole study varies. TIME_LIMIT is the
    defect: the arm failed to reach the target it asked for.
    """
    src = inspect.getsource(Incremental2ROSolver.evaluate_subset)
    assert "self.require_converged and cand_model.Status != GRB.OPTIMAL" in src


def test_unrescued_candidate_cannot_win_the_argmax() -> None:
    """A candidate that will not converge must be returned not-feasible, which the caller
    maps to -inf, so it can never be selected."""
    src = inspect.getsource(Incremental2ROSolver.evaluate_subset)
    assert "self.n_excluded_unconverged += 1" in src
    after = src.split("self.n_excluded_unconverged += 1", 1)[1]
    assert 'return (float("inf"), 0.0, 0.0, False)' in after
    assert "logger.warning" in after, "a silently dropped candidate must not look like a clean step"
