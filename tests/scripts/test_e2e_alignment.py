"""E2E pipeline alignment tests.

Verifies that the evaluation pipeline does not introduce data alignment
or solver timeout issues.
"""
import numpy as np
import pytest

from src.core.types import Instance
from src.utils.batch_utils import _build_reduced_instance_from_np

@pytest.fixture(scope="module")
def vc_instances():
    """Small reproducible instances, independent of downloaded experiment data."""
    problem = _get_vc_problem()
    rng = np.random.default_rng(42)
    problem._rng = rng
    problem.scenario_generator._rng = rng
    return [problem.sample() for _ in range(10)]


def _get_vc_problem():
    from src.problems.vertex_cover import VertexCoverProblem
    return VertexCoverProblem(n_nodes=8, n_scenarios=6, p_edge=3.0)


# ======================================================================
# Test 1: DataModule iteration order consistency
# ======================================================================
def test_dm_ml_dm_2ro_same_order(vc_instances):
    """Two DataModules (normalized vs raw) must iterate instances in the same order.

    The e2e pipeline creates dm_ml (with normalize=True) for ML inference
    and dm_2ro (with normalize=False) for BigMILP solving. If these iterate
    in different order, scenario indices from dm_ml applied to dm_2ro will
    be misaligned.

    We verify that each normalized scenario matrix matches the shared det/sce
    min-max transform of the raw instance at the same position.
    """
    instances = vc_instances

    # Simulate what DataModule does: normalize float keys
    from src.training.datamodule import DataModule
    dm = DataModule.__new__(DataModule)

    # Raw instances: extract sce arrays
    raw_sces = []
    for inst in instances:
        raw_sces.append(np.asarray(inst.data["sce"]))

    # Normalized instances
    normalized = dm._normalize_instances(instances)
    norm_sces = []
    for inst in normalized:
        v = inst.data["sce"]
        if hasattr(v, 'numpy'):
            norm_sces.append(v.numpy())
        else:
            norm_sces.append(np.asarray(v))

    # Verify instance ordering: normalization should be a deterministic
    # per-instance linear transform. We check that norm_sce[i] is a linear
    # transform of raw_sce[i] (proving same instance at same position).
    for i in range(len(raw_sces)):
        raw_sce = raw_sces[i]
        norm_sce = norm_sces[i]
        assert raw_sce.shape == norm_sce.shape, f"Instance {i}: shape mismatch"

        # det and sce share min/max to preserve their relative cost scale.
        raw = raw_sce.astype(float)
        norm = norm_sce.astype(float)
        det = np.asarray(instances[i].data["det"], dtype=float)
        vmin, vmax = min(raw.min(), det.min()), max(raw.max(), det.max())
        if vmax > vmin:
            reconstructed = (raw - vmin) / (vmax - vmin)
            assert np.allclose(norm, reconstructed, atol=1e-3), (
                f"Instance {i}: normalized sce does not match expected (raw-min)/(max-min). "
                f"Instance ordering may differ between normalized and raw DataModules."
            )
        else:
            # All det/sce costs are constant, so normalization produces zeros
            assert np.allclose(norm, 0.0, atol=1e-6), (
                f"Instance {i}: constant sce should normalize to 0."
            )


# ======================================================================
# Test 2: Time limit on small instances (should not matter)
# ======================================================================
def test_reduced_solve_timelimit(vc_instances):
    """For small instances (n=8, K=2), time_limit=1 and time_limit=60
    should produce the same reduced objective.

    This verifies that the solver converges within 1s for small problems.
    For large problems (n=100), we expect time_limit=1 to be insufficient.
    """
    instances = vc_instances[:3]
    problem = _get_vc_problem()
    K = 2

    for i, inst in enumerate(instances):
        data_np = {k: np.asarray(v) for k, v in inst.data.items()}

        # Fix a subset to isolate reduced-solver behavior.
        sce_idx = np.arange(K)

        # Solve with time_limit=1
        reduced_inst_1 = _build_reduced_instance_from_np(problem, data_np, sce_idx)
        labels_1 = problem.solve_exact(reduced_inst_1, time_limit=1.0, threads=2)

        # Solve with time_limit=60
        reduced_inst_60 = _build_reduced_instance_from_np(problem, data_np, sce_idx)
        labels_60 = problem.solve_exact(reduced_inst_60, time_limit=60.0, threads=2)

        # For n=8, both should converge to the same solution
        assert abs(float(labels_1.obj) - float(labels_60.obj)) < 1.0, (
            f"Instance {i}: time_limit=1 obj={labels_1.obj:.1f} vs "
            f"time_limit=60 obj={labels_60.obj:.1f} — "
            f"small instances should solve to optimality within 1s"
        )


# ======================================================================
# Test 3: Fixed-subset self-consistency
# ======================================================================
def test_subset_self_consistency(vc_instances):
    """Fixed scenario indices → reduced BigMILP solve → regret must be ≤ 0.

    Regret = reduced_obj - full_obj. Since the reduced problem optimizes over
    fewer scenario constraints, reduced_obj ≤ full_obj for this minimization problem.
    Equivalently, regret ≤ 0.
    """
    instances = vc_instances[:5]
    problem = _get_vc_problem()
    K = 4

    for i, inst in enumerate(instances):
        data_np = {k: np.asarray(v) for k, v in inst.data.items()}

        # Fix a subset to isolate reduced-solver behavior.
        sce_idx = np.arange(K)

        # Full solve
        full_labels = problem.solve_exact(inst, time_limit=30.0, threads=2)
        full_obj = float(full_labels.obj)

        # Reduced solve
        reduced_inst = _build_reduced_instance_from_np(problem, data_np, sce_idx)
        red_labels = problem.solve_exact(reduced_inst, time_limit=30.0, threads=2)
        red_obj = float(red_labels.obj)

        regret = red_obj - full_obj
        assert regret <= 1.0, (  # small tolerance for numerical noise
            f"Instance {i}: regret={regret:.2f} > 0 — "
            f"reduced_obj ({red_obj:.1f}) should be ≤ full_obj ({full_obj:.1f})"
        )


# ======================================================================
# Test 4: Full objective consistency across data paths
# ======================================================================
def test_e2e_indices_alignment(vc_instances):
    """full_obj computed from a raw Instance should match
    full_obj computed from numpy-dict extraction of the same Instance.

    This verifies that _instance_to_numpy_dict → solve_exact produces
    the same result as solving the original Instance directly. If not,
    the e2e pipeline (which extracts numpy dicts from batched data)
    would get different objectives.
    """
    from src.utils.batch_utils import _instance_to_numpy_dict
    import torch

    instances = vc_instances[:3]
    problem = _get_vc_problem()

    for i, inst in enumerate(instances):
        # Path A: solve original Instance directly
        labels_a = problem.solve_exact(inst, time_limit=30.0, threads=2)
        obj_a = float(labels_a.obj)

        # Path B: convert to batched tensor → extract numpy dict → rebuild Instance → solve
        # Simulate what DataModule does
        batched_data = {}
        for k, v in inst.data.items():
            arr = np.asarray(v)
            t = torch.from_numpy(arr).unsqueeze(0)  # add batch dim
            batched_data[k] = t
        batched_inst = Instance(batched_data)

        data_np = _instance_to_numpy_dict(batched_inst, idx_in_batch=0)
        rebuilt_inst = Instance(data={k: v for k, v in data_np.items()})
        labels_b = problem.solve_exact(rebuilt_inst, time_limit=30.0, threads=2)
        obj_b = float(labels_b.obj)

        assert abs(obj_a - obj_b) < 1.0, (
            f"Instance {i}: direct solve obj={obj_a:.1f} vs "
            f"numpy-dict roundtrip obj={obj_b:.1f} — "
            f"data extraction introduces objective difference"
        )
