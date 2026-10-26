"""Tests for NormalScenarioGenerator (graph-problem normal-distribution costs)."""

import numpy as np
import pytest

from src.problems.scenario_generator import NormalScenarioGenerator


@pytest.fixture
def gen() -> NormalScenarioGenerator:
    return NormalScenarioGenerator(
        n_min_cost=1,
        n_max_cost=100,
        mu_min=25.0,
        mu_max=75.0,
        sigma_ratio=0.15,
        sigma_min=3.0,
        sigma_max=15.0,
        fixed_profile=True,
        rng=np.random.default_rng(42),
    )


def test_output_shapes(gen: NormalScenarioGenerator) -> None:
    """det and sce have correct shapes [n_nodes] and [n_scenarios, n_nodes]."""
    det, sce = gen.sample_costs(n_nodes=20, n_scenarios=50)
    assert det.shape == (20,)
    assert sce.shape == (50, 20)


def test_costs_clipped(gen: NormalScenarioGenerator) -> None:
    """All costs are within [n_min_cost, n_max_cost]."""
    det, sce = gen.sample_costs(n_nodes=20, n_scenarios=200)
    assert det.min() >= 1
    assert det.max() <= 100
    assert sce.min() >= 1
    assert sce.max() <= 100


def test_integer_costs(gen: NormalScenarioGenerator) -> None:
    """Costs are integer-valued (rounded)."""
    det, sce = gen.sample_costs(n_nodes=20, n_scenarios=50)
    np.testing.assert_array_equal(det, det.astype(int))
    np.testing.assert_array_equal(sce, sce.astype(int))


def test_fixed_profile_reuses_means(gen: NormalScenarioGenerator) -> None:
    """With fixed_profile=True, same mu/sigma profile is reused across calls."""
    det1, _ = gen.sample_costs(n_nodes=20, n_scenarios=10)
    det2, _ = gen.sample_costs(n_nodes=20, n_scenarios=10)
    # det = rounded means, should be identical across calls
    np.testing.assert_array_equal(det1, det2)


def test_non_fixed_profile_varies() -> None:
    """With fixed_profile=False, mu profiles differ across calls."""
    gen = NormalScenarioGenerator(
        fixed_profile=False,
        rng=np.random.default_rng(42),
    )
    det1, _ = gen.sample_costs(n_nodes=20, n_scenarios=10)
    det2, _ = gen.sample_costs(n_nodes=20, n_scenarios=10)
    # Very unlikely to be identical with different random draws
    assert not np.array_equal(det1, det2)


def test_scenario_distribution_clusters_around_means(gen: NormalScenarioGenerator) -> None:
    """Scenario costs cluster around node means (lower std than uniform)."""
    det, sce = gen.sample_costs(n_nodes=20, n_scenarios=1000)
    # Per-node std should be much less than uniform's ~29 (for U[1,100])
    per_node_std = sce.std(axis=0)
    assert per_node_std.mean() < 20.0, (
        f"Mean per-node std {per_node_std.mean():.1f} is too high for normal"
    )


def test_scenario_means_near_det(gen: NormalScenarioGenerator) -> None:
    """Scenario means should be close to deterministic costs."""
    det, sce = gen.sample_costs(n_nodes=20, n_scenarios=2000)
    sce_means = sce.mean(axis=0)
    # Allow some deviation due to clipping, but should be within ~5
    max_dev = np.abs(sce_means - det).max()
    assert max_dev < 10.0, f"Max deviation {max_dev:.1f} between sce means and det"


def test_validation_errors() -> None:
    """Constructor validates parameter bounds."""
    with pytest.raises(ValueError, match="mu_max"):
        NormalScenarioGenerator(mu_min=80, mu_max=20)
    with pytest.raises(ValueError, match="sigma_ratio"):
        NormalScenarioGenerator(sigma_ratio=-0.1)
    with pytest.raises(ValueError, match="sigma bounds"):
        NormalScenarioGenerator(sigma_min=10, sigma_max=5)


def test_sample_costs_validation(gen: NormalScenarioGenerator) -> None:
    """sample_costs rejects non-positive dimensions."""
    with pytest.raises(ValueError, match="n_nodes"):
        gen.sample_costs(n_nodes=0, n_scenarios=10)
    with pytest.raises(ValueError, match="n_scenarios"):
        gen.sample_costs(n_nodes=10, n_scenarios=0)
