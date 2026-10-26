import numpy as np

from src.problems.scenario_generator import CFLPNormalDemandScenarioGenerator


def test_cflp_normal_generator_fixed_profile_reuses_means():
    """
    Motivation:
    - Ensures easy-unimodal setup is stable across generated instances.
    - Guards the fixed_profile behavior so mu/sigma are reused for the same n_nodes.
    """
    generator = CFLPNormalDemandScenarioGenerator(
        demand_min=10.0,
        demand_max=500.0,
        mu_min=180.0,
        mu_max=260.0,
        sigma_ratio=0.08,
        sigma_min=12.0,
        sigma_max=22.0,
        fixed_profile=True,
        clip_to_bounds=True,
    )

    det_1, sce_1 = generator.sample_costs(n_nodes=30, n_scenarios=60)
    det_2, sce_2 = generator.sample_costs(n_nodes=30, n_scenarios=60)

    assert det_1.shape == (30,)
    assert sce_1.shape == (60, 30)
    assert det_2.shape == (30,)
    assert sce_2.shape == (60, 30)
    assert np.allclose(det_1, det_2, atol=1e-7)
    assert float(np.min(det_1)) >= 180.0
    assert float(np.max(det_1)) <= 260.0
    assert float(np.std(sce_1)) > 0.0


def test_cflp_normal_generator_clips_to_bounds():
    """
    Motivation:
    - Protects bound clipping for generated normal demands.
    - Ensures samples remain inside configured [demand_min, demand_max].
    """
    generator = CFLPNormalDemandScenarioGenerator(
        demand_min=100.0,
        demand_max=220.0,
        mu_min=180.0,
        mu_max=260.0,
        sigma_ratio=0.12,
        sigma_min=15.0,
        sigma_max=40.0,
        fixed_profile=True,
        clip_to_bounds=True,
    )

    det, sce = generator.sample_costs(n_nodes=20, n_scenarios=400)
    assert float(np.min(det)) >= 100.0
    assert float(np.max(det)) <= 220.0
    assert float(np.min(sce)) >= 100.0
    assert float(np.max(sce)) <= 220.0


def test_cflp_normal_generator_empirical_std_matches_sigma():
    """
    Motivation:
    - Validates the intended sigma design (sigma = ratio * mu with clipping).
    - Guards against accidental changes that distort normal spread.
    """
    generator = CFLPNormalDemandScenarioGenerator(
        demand_min=0.0,
        demand_max=1000.0,
        mu_min=200.0,
        mu_max=200.0,
        sigma_ratio=0.08,
        sigma_min=0.0,
        sigma_max=100.0,
        fixed_profile=True,
        clip_to_bounds=True,
    )

    det, sce = generator.sample_costs(n_nodes=1, n_scenarios=20000)
    expected_sigma = 16.0
    empirical_sigma = float(np.std(sce[:, 0], ddof=0))

    assert det.shape == (1,)
    assert float(det[0]) == 200.0
    assert abs(empirical_sigma - expected_sigma) < 0.7
