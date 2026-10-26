import importlib.util
from pathlib import Path

import numpy as np

from src.core.types import ForwardTrace, Labels


def _load_prise_label_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "02_prise_label.py"
    spec = importlib.util.spec_from_file_location("prise_label", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_round_label_floats_to_4_decimals():
    """Motivation: label dump should round all float fields to 4 decimals."""
    trace = ForwardTrace(
        K=2,
        S=3,
        selected_idx=np.array([0, 1], dtype=int),
        candidate_scores=[np.array([1.23456, 2.34567, 3.45678], dtype=float)],
        best_scores=np.array([4.56789, 5.67891], dtype=float),
        gains=np.array([6.78912, 7.89123], dtype=float),
        stop_step=2,
        step_mask=np.array([True, False]),
    )
    label = Labels(
        obj=np.array(1.234567, dtype=float),
        feasible=True,
        x=np.array([1.11111, 2.22222], dtype=float),
        sce_target_idx=np.array([1], dtype=int),
        sce_ranking=np.array([2, 1, 0], dtype=int),
        sce_scores=np.array([9.87654, 8.76543, 7.65432], dtype=float),
        sce_top_idx=np.array([0, 2], dtype=int),
        obj_bound=np.array(3.14159, dtype=float),
        mip_gap=np.array(0.00012345, dtype=float),
        meta={"foo": 2.34567, "bar": {"baz": 7.89123}},
        forward_trace=trace,
    )

    module = _load_prise_label_module()
    out = module._round_label_floats(label, decimals=4)

    assert float(out.obj) == 1.2346
    assert np.allclose(out.x, np.array([1.1111, 2.2222], dtype=float))
    assert np.allclose(out.sce_scores, np.array([9.8765, 8.7654, 7.6543], dtype=float))
    assert float(out.obj_bound) == 3.1416
    assert float(out.mip_gap) == 0.0001
    assert out.meta["foo"] == 2.3457
    assert out.meta["bar"]["baz"] == 7.8912
    assert np.allclose(out.forward_trace.candidate_scores[0], np.array([1.2346, 2.3457, 3.4568], dtype=float))
    assert np.allclose(out.forward_trace.best_scores, np.array([4.5679, 5.6789], dtype=float))
    assert np.allclose(out.forward_trace.gains, np.array([6.7891, 7.8912], dtype=float))
