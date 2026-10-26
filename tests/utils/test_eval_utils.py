import pytest
import numpy as np
import torch
import types
from src.utils.eval_utils import _compute_K_from_ratio, infer_T_from_labels

def test_compute_k_from_ratio_boundary_exactly_1():
    """k_ratio=1.0 returns K=S."""
    S = 10
    K = _compute_K_from_ratio(S, 1.0)
    assert K == S

def test_compute_k_from_ratio_small_ratio():
    """k_ratio=0.1 with S=10 returns K=1."""
    K = _compute_K_from_ratio(10, 0.1)
    assert K == 1

def test_compute_k_from_ratio_invalid_zero_raises():
    """k_ratio=0.0 raises ValueError."""
    with pytest.raises(ValueError):
        _compute_K_from_ratio(10, 0.0)

def test_compute_k_from_ratio_invalid_gt1_raises():
    """k_ratio > 1.0 raises ValueError."""
    with pytest.raises(ValueError):
        _compute_K_from_ratio(10, 1.5)

def test_infer_T_from_labels_fallback_to_sce_top_idx():
    """When forward_trace is None, fallback to sce_top_idx shape[1]."""
    labels = types.SimpleNamespace(
        forward_trace=None,
        sce_top_idx=torch.zeros((2, 7), dtype=torch.long),  # K=7
    )
    T = infer_T_from_labels(labels)
    assert T == 7

def test_infer_T_from_labels_from_forward_trace_2d():
    """When forward_trace.selected_idx is 2D [B, T], return T."""
    ft = types.SimpleNamespace(selected_idx=torch.zeros((3, 5), dtype=torch.long))
    labels = types.SimpleNamespace(forward_trace=ft, sce_top_idx=None)
    T = infer_T_from_labels(labels)
    assert T == 5

def test_infer_T_from_labels_from_forward_trace_1d():
    """When forward_trace.selected_idx is 1D [T], return T."""
    ft = types.SimpleNamespace(selected_idx=torch.zeros((4,), dtype=torch.long))
    labels = types.SimpleNamespace(forward_trace=ft, sce_top_idx=None)
    T = infer_T_from_labels(labels)
    assert T == 4

def test_infer_T_from_labels_none_when_no_info():
    """Returns None when there is no forward_trace and no sce_top_idx."""
    labels = types.SimpleNamespace(forward_trace=None, sce_top_idx=None)
    T = infer_T_from_labels(labels)
    assert T is None
