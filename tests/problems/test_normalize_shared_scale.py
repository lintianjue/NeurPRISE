import numpy as np

from src.core.types import Instance
from src.training.datamodule import DataModule


def test_vertex_cover_normalize_shared_scale():
    """
    Motivation:
    - det and sce should share the same normalization scale.
    - Guards against per-tensor scaling that changes the objective balance.
    """
    det = np.array([1.0, 3.0], dtype=np.float32)
    sce = np.array([[2.0, 4.0], [5.0, 1.0]], dtype=np.float32)
    inst = Instance(
        {
            "n_nodes": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "det": det,
            "sce": sce,
            "adjacency": np.zeros((2, 2), dtype=np.int64),
        }
    )

    inst_norm = DataModule._normalize_instance_shared_pairs(inst)
    det_norm = inst_norm.data["det"].detach().cpu().numpy()
    sce_norm = inst_norm.data["sce"].detach().cpu().numpy()

    assert np.allclose(det_norm, np.array([0.0, 0.5], dtype=np.float32))
    assert np.allclose(
        sce_norm,
        np.array([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32),
    )


def test_cflp_normalize_shared_scale():
    """
    Motivation:
    - CFLP has multiple cost tensors; they should share a single scale.
    """
    f = np.array([100.0, 200.0], dtype=np.float32)
    a = np.array([10.0, 20.0], dtype=np.float32)
    K = np.array([200.0, 300.0], dtype=np.float32)
    c = np.array([[1.0, 4.0], [5.0, 2.0]], dtype=np.float32)
    d = np.array([[10.0, 50.0], [100.0, 20.0]], dtype=np.float32)

    inst = Instance(
        {
            "n_facilities": np.array([2], dtype=np.int64),
            "n_customers": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "f": f,
            "a": a,
            "K": K,
            "c": c,
            "d": d,
        }
    )

    inst_norm = DataModule._normalize_instance_shared_pairs(inst)

    # Each tensor is normalized independently in CFLP.
    # f: min=100, max=200 -> [0,1]
    assert np.isclose(inst_norm.data["f"][0].item(), 0.0, atol=1e-6)
    assert np.isclose(inst_norm.data["f"][1].item(), 1.0, atol=1e-6)

    # c: min=1, max=5 -> [0,1]
    assert np.isclose(inst_norm.data["c"][0, 0].item(), 0.0, atol=1e-6)
    assert np.isclose(inst_norm.data["c"][1, 0].item(), 1.0, atol=1e-6)


def test_vertex_cover_normalize_scale_mode_avoids_shift() -> None:
    """
    Motivation:
    - scale mode should preserve relative costs without min-subtraction.
    - Guards against objective-shift distortions when using scale-only normalization.
    """
    det = np.array([1.0, 3.0], dtype=np.float32)
    sce = np.array([[2.0, 4.0], [5.0, 1.0]], dtype=np.float32)
    inst = Instance(
        {
            "n_nodes": np.array([2], dtype=np.int64),
            "n_scenarios": np.array([2], dtype=np.int64),
            "det": det,
            "sce": sce,
            "adjacency": np.zeros((2, 2), dtype=np.int64),
        }
    )

    inst_norm = DataModule._normalize_instance_shared_pairs(inst, mode="scale")
    det_norm = inst_norm.data["det"].detach().cpu().numpy()
    sce_norm = inst_norm.data["sce"].detach().cpu().numpy()

    expected_det = np.array([0.2, 0.6], dtype=np.float32)
    expected_sce = np.array([[0.4, 0.8], [1.0, 0.2]], dtype=np.float32)

    assert np.allclose(det_norm, expected_det, atol=1e-6)
    assert np.allclose(sce_norm, expected_sce, atol=1e-6)
