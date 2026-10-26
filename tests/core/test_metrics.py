"""
TC-15: test_metric_obj_mean
TC-16: test_metric_feas_partial_feasibility
TC-17: test_compute_dispatches_to_correct_metric
"""
import pytest
import torch
import numpy as np

from src.core.metrics import metric_obj, metric_feas, compute
from src.core.types import Labels


def _make_labels(obj_vals, feas_vals):
    """Helper: build a minimal Labels with obj and feasible as torch tensors."""
    lab = Labels(
        obj=torch.tensor(obj_vals, dtype=torch.float32),
        feasible=torch.tensor(feas_vals, dtype=torch.bool),
    )
    return lab


# TC-15
def test_metric_obj_mean():
    """metric_obj returns the mean of labels.obj."""
    labels = _make_labels([1.0, 3.0, 5.0], [True, True, True])
    result = metric_obj(labels)
    assert float(result.item()) == pytest.approx(3.0, abs=1e-6)


def test_metric_obj_single():
    """metric_obj works for a single-element tensor."""
    labels = _make_labels([7.5], [True])
    assert float(metric_obj(labels).item()) == pytest.approx(7.5, abs=1e-6)


# TC-16
def test_metric_feas_partial_feasibility():
    """metric_feas returns the fraction of True in labels.feasible."""
    # 2 out of 4 are True => 0.5
    labels = _make_labels([0.0, 0.0, 0.0, 0.0], [True, False, True, False])
    result = metric_feas(labels)
    assert float(result.item()) == pytest.approx(0.5, abs=1e-6)


def test_metric_feas_all_true():
    labels = _make_labels([1.0, 2.0], [True, True])
    assert float(metric_feas(labels).item()) == pytest.approx(1.0, abs=1e-6)


def test_metric_feas_all_false():
    labels = _make_labels([1.0, 2.0], [False, False])
    assert float(metric_feas(labels).item()) == pytest.approx(0.0, abs=1e-6)


# TC-17
def test_compute_dispatches_to_correct_metric():
    """compute() dispatches correctly to obj and feas metrics."""
    labels = _make_labels([2.0, 4.0], [True, False])
    result = compute(["obj", "feas"], labels)

    assert "obj" in result
    assert "feas" in result
    assert result["obj"] == pytest.approx(3.0, abs=1e-6)
    assert result["feas"] == pytest.approx(0.5, abs=1e-6)


def test_compute_raises_key_error_for_unknown_metric():
    """compute() raises KeyError for unknown metric names."""
    labels = _make_labels([1.0], [True])
    with pytest.raises(KeyError):
        compute(["nonexistent_metric"], labels)


def test_compute_empty_list_returns_empty_dict():
    """compute() with empty list returns empty dict."""
    labels = _make_labels([1.0], [True])
    result = compute([], labels)
    assert result == {}
