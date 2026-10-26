import numpy as np

from src.problems.cflp import CFLPProblem
from src.problems.scenario_generator import BaseScenarioGenerator, CFLPMultimodalDemandScenarioGenerator


class _FixedDemandGenerator(BaseScenarioGenerator):
    def __init__(self, demand_template: np.ndarray):
        super().__init__(n_min_cost=0, n_max_cost=1, rng=np.random.default_rng(0))
        self._demand_template = np.asarray(demand_template, dtype=np.float32)
        self.calls = 0

    def sample_costs(self, n_nodes: int, n_scenarios: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        det = np.zeros((n_nodes,), dtype=np.float32)
        demand = self._demand_template.copy()
        if demand.shape != (n_scenarios, n_nodes):
            raise ValueError(
                f"test generator shape mismatch: got {demand.shape}, expected {(n_scenarios, n_nodes)}."
            )
        return det, demand


def test_cflp_multimodal_generator_shape_and_range():
    """
    Motivation:
    - Guards the new CFLP multimodal generator contract (shape/range).
    - Ensures generated demands stay inside configured demand bounds.
    """
    generator = CFLPMultimodalDemandScenarioGenerator(
        demand_min=10.0,
        demand_max=500.0,
        mode_count_min=3,
        mode_count_max=6,
        mode_mu_min=80.0,
        mode_mu_max=380.0,
        mode_delta_min=0.05,
        mode_delta_max=0.20,
        shared_shock_std=0.08,
        clip_to_bounds=True,
    )

    det, sce = generator.sample_costs(n_nodes=30, n_scenarios=50)
    assert det.shape == (30,)
    assert sce.shape == (50, 30)
    assert float(np.min(sce)) >= 10.0
    assert float(np.max(sce)) <= 500.0
    assert float(np.std(sce)) > 0.0


def test_cflp_custom_generator_is_used_and_clipped():
    """
    Motivation:
    - Verifies CFLP uses injected scenario_generator when provided.
    - Protects clipping behavior to [d_min, d_max] before solver feasibility checks.
    """
    gen = _FixedDemandGenerator(
        demand_template=np.array(
            [
                [600.0, 50.0, -20.0],
                [400.0, 120.0, 80.0],
                [200.0, 20.0, 15.0],
                [300.0, 40.0, 30.0],
            ],
            dtype=np.float32,
        )
    )
    problem = CFLPProblem(
        n_facilities=3,
        n_customers=3,
        n_scenarios=4,
        K_min=250.0,
        K_max=250.0,
        d_min=10.0,
        d_max=500.0,
        scenario_name="multimodal",
        scenario_generator=gen,
    )

    _, demand = problem._sample_capacities_and_demands()
    assert gen.calls >= 1
    assert demand.shape == (4, 3)
    assert float(np.min(demand)) >= 10.0
    assert float(np.max(demand)) <= 500.0


def test_cflp_custom_generator_fallback_scales_infeasible_demands():
    """
    Motivation:
    - Ensures capacity-feasibility fallback still works with custom generators.
    - Prevents regressions where always-infeasible sampled demands are returned unscaled.
    """
    gen = _FixedDemandGenerator(
        demand_template=np.full((2, 4), 500.0, dtype=np.float32)
    )
    problem = CFLPProblem(
        n_facilities=2,
        n_customers=4,
        n_scenarios=2,
        K_min=120.0,
        K_max=120.0,
        d_min=10.0,
        d_max=500.0,
        scenario_name="multimodal",
        scenario_generator=gen,
    )

    K, demand = problem._sample_capacities_and_demands()
    total_capacity = float(np.sum(K))
    max_total_demand = float(np.max(np.sum(demand, axis=1)))

    # max_attempts in CFLPProblem._sample_capacities_and_demands
    assert gen.calls == 100
    assert max_total_demand <= 0.99 * total_capacity + 1e-3
