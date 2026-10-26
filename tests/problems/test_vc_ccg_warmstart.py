"""Warm-start hook for CCG on vertex cover — the actual algorithm of Goerigk & Kurtz (2025).

The distinction these tests pin down is the whole point of the experiment:

  `scenario_indices`          restricts the CANDIDATE set. Separation only ever
                              looks at those scenarios, so the answer is exact
                              only with respect to the subset. This is scenario
                              *reduction*.
  `initial_active_scenarios`  seeds the master with those scenarios but leaves
                              separation over ALL of them. The answer stays
                              exact; only the iteration count moves. This is
                              GK's predict-then-CCG.

If a warm start could change the objective, the experiment would be measuring
a different algorithm than the one we claim to run, so `test_warm_start_matches_
cold_objective` is the load-bearing test here — not the speed.
"""
import numpy as np
import pytest

from src.core.types import Instance
from src.problems.vertex_cover import VertexCoverProblem


@pytest.fixture
def inst():
    """n=12, S=8 — big enough that cold CCG needs several iterations."""
    rng = np.random.RandomState(7)
    N, S = 12, 8
    det = rng.randint(1, 100, size=N).astype(float)
    sce = rng.randint(1, 100, size=(S, N)).astype(float)
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.35:
                adj[i, j] = adj[j, i] = 1
    return Instance({
        "n_nodes": np.array([N]), "n_scenarios": np.array([S]),
        "n_min_cost": np.array([1]), "n_max_cost": np.array([100]),
        "p_edge": np.array([35.0]), "det": det, "sce": sce, "adjacency": adj,
    })


@pytest.fixture
def prob():
    return VertexCoverProblem(n_nodes=12, n_scenarios=8)


def _ccg(prob, inst, **kw):
    return prob.solve_robust_ccg(inst, threads=1, verbose=False,
                                 max_iterations=40, **kw)


# --------------------------------------------------------------------------
# The hook must exist and must be additive
# --------------------------------------------------------------------------

def test_accepts_initial_active_scenarios(prob, inst):
    out = _ccg(prob, inst, initial_active_scenarios=[0, 3])
    assert out["has_incumbent"]


def test_default_none_is_unchanged(prob, inst):
    """Regression guard: omitting the kwarg must reproduce old behaviour."""
    a = _ccg(prob, inst)
    b = _ccg(prob, inst, initial_active_scenarios=None)
    assert a["obj"] == pytest.approx(b["obj"], rel=1e-9)
    assert a["ccg_iterations"] == b["ccg_iterations"]


# --------------------------------------------------------------------------
# Exactness — the load-bearing property
# --------------------------------------------------------------------------

def test_warm_start_matches_cold_objective(prob, inst):
    cold = _ccg(prob, inst)
    for seed_set in ([0], [1, 5], [0, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6, 7]):
        warm = _ccg(prob, inst, initial_active_scenarios=seed_set)
        assert warm["obj"] == pytest.approx(cold["obj"], rel=1e-6), (
            f"warm start {seed_set} changed the optimum: "
            f"{warm['obj']} vs cold {cold['obj']}"
        )


def test_separation_still_covers_all_scenarios(prob, inst):
    """A warm start must NOT narrow the candidate set.

    Seeding with one scenario and separating over all 8 must find the same
    optimum as cold. Contrast with scenario_indices, which genuinely restricts
    and is allowed to differ — asserted in the next test.
    """
    cold = _ccg(prob, inst)
    warm = _ccg(prob, inst, initial_active_scenarios=[0])
    assert warm["obj"] == pytest.approx(cold["obj"], rel=1e-6)


def test_scenario_indices_still_restricts(prob, inst):
    """Guard against 'fixing' the two kwargs into one.

    A restricted candidate set is a weaker relaxation, so its objective must be
    <= the full one. If someone makes scenario_indices behave like a warm start,
    this fails.
    """
    full = _ccg(prob, inst)
    restricted = _ccg(prob, inst, scenario_indices=[0, 1])
    assert restricted["obj"] <= full["obj"] + 1e-6


# --------------------------------------------------------------------------
# Convergence bookkeeping
# --------------------------------------------------------------------------

def test_seeding_all_scenarios_converges_immediately(prob, inst):
    """With every scenario already in the master, iteration 1 is optimal."""
    out = _ccg(prob, inst, initial_active_scenarios=list(range(8)))
    assert out["ccg_converged"]
    assert out["ccg_iterations"] == 1


def test_warm_start_does_not_increase_iterations(prob, inst):
    cold = _ccg(prob, inst)
    warm = _ccg(prob, inst, initial_active_scenarios=list(range(8)))
    assert warm["ccg_iterations"] <= cold["ccg_iterations"]


def test_active_set_reported_includes_the_seed(prob, inst):
    """The reported active set must show what was seeded, or the run record
    cannot distinguish a warm start from a cold one after the fact."""
    out = _ccg(prob, inst, initial_active_scenarios=[2, 5])
    assert {2, 5}.issubset(set(out["ccg_active_scenarios"]))


# --------------------------------------------------------------------------
# Input validation — a silent drop here would fake a speedup
# --------------------------------------------------------------------------

def test_out_of_range_seed_raises(prob, inst):
    with pytest.raises(ValueError):
        _ccg(prob, inst, initial_active_scenarios=[0, 99])


def test_seed_outside_candidate_set_raises(prob, inst):
    """Seeding a scenario that separation cannot see is incoherent."""
    with pytest.raises(ValueError):
        _ccg(prob, inst, scenario_indices=[0, 1], initial_active_scenarios=[5])


def test_empty_seed_falls_back_to_cold(prob, inst):
    cold = _ccg(prob, inst)
    out = _ccg(prob, inst, initial_active_scenarios=[])
    assert out["obj"] == pytest.approx(cold["obj"], rel=1e-9)


def test_duplicate_seed_indices_are_deduped(prob, inst):
    out = _ccg(prob, inst, initial_active_scenarios=[3, 3, 3])
    act = out["ccg_active_scenarios"]
    assert len(act) == len(set(act))
