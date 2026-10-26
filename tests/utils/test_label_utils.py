import numpy as np
import torch

from src.core.types import ForwardTrace
from src.utils.label_utils import _collate_forward_trace_from_traces


def _to_tensor(x):
    # minimal to_tensor_fn for unit tests
    return torch.as_tensor(x)


def test_collate_forward_trace_copies_all_fields_and_padding():
    # Trace 1: T=3, step_mask cuts off last step
    tr1 = ForwardTrace(
        K=5,
        S=4,
        selected_idx=np.array([2, 0, 1], dtype=np.int64),
        candidate_scores=[
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        ],
        best_scores=np.array([4.0, 8.0, 12.0], dtype=np.float32),
        gains=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        stop_step=2,
        step_mask=np.array([True, True, False], dtype=bool),
    )

    # Trace 2: T=5, full mask
    tr2 = ForwardTrace(
        K=5,
        S=4,
        selected_idx=np.array([3, 1, 0, 2, 1], dtype=np.int64),
        candidate_scores=[
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32),
            np.array([12.0, 13.0, 14.0, 15.0], dtype=np.float32),
            np.array([16.0, 17.0, 18.0, 19.0], dtype=np.float32),
        ],
        best_scores=np.array([3.0, 7.0, 11.0, 15.0, 19.0], dtype=np.float32),
        gains=np.array([3.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float32),
        stop_step=5,
        step_mask=np.array([True, True, True, True, True], dtype=bool),
    )

    batch = _collate_forward_trace_from_traces([tr1, tr2], _to_tensor)

    assert batch.K == 5
    assert batch.S == 4
    assert batch.selected_idx.shape == (2, 5)
    assert batch.candidate_scores.shape == (2, 5, 4)
    assert batch.step_mask.shape == (2, 5)
    assert batch.best_scores.shape == (2, 5)
    assert batch.gains.shape == (2, 5)
    assert batch.stop_step.shape == (2,)
    assert batch.T.tolist() == [3, 5]

    # selected_idx padding + masking (step_mask False => -1)
    assert batch.selected_idx[0].tolist() == [2, 0, -1, -1, -1]
    assert batch.selected_idx[1].tolist() == [3, 1, 0, 2, 1]

    # step_mask MUST match trace.step_mask (not derived from len(selected_idx))
    assert batch.step_mask[0].tolist() == [True, True, False, False, False]
    assert batch.step_mask[1].tolist() == [True, True, True, True, True]

    # best_scores / gains copied and padded with NaN
    assert torch.allclose(batch.best_scores[0, :3], torch.tensor([4.0, 8.0, 12.0]))
    assert torch.isnan(batch.best_scores[0, 3:]).all()
    assert torch.allclose(batch.gains[0, :3], torch.tensor([4.0, 4.0, 4.0]))
    assert torch.isnan(batch.gains[0, 3:]).all()

    assert torch.allclose(batch.best_scores[1, :5], torch.tensor([3.0, 7.0, 11.0, 15.0, 19.0]))
    assert torch.allclose(batch.gains[1, :5], torch.tensor([3.0, 4.0, 4.0, 4.0, 4.0]))

    # stop_step copied
    assert batch.stop_step.tolist() == [2, 5]

    # candidate_scores copied + padded with -inf
    assert torch.allclose(batch.candidate_scores[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(batch.candidate_scores[0, 2], torch.tensor([9.0, 10.0, 11.0, 12.0]))
    assert torch.isneginf(batch.candidate_scores[0, 3:]).all()


def test_negative_original_bug_step_mask_incorrectly_derived_from_selected_idx_len():
    """
    Reproduces the original bug behavior:
    if step_mask is incorrectly derived from len(selected_idx), then any False entries in
    trace.step_mask are lost (becoming True up to Ti), which changes supervision.
    """
    tr = ForwardTrace(
        K=5,
        S=4,
        selected_idx=np.array([2, 0, 1], dtype=np.int64),  # len=3
        candidate_scores=[
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
        ],
        best_scores=np.array([4.0, 8.0, 12.0], dtype=np.float32),
        gains=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        stop_step=2,
        step_mask=np.array([True, True, False], dtype=bool),
    )

    # Correct behavior (current implementation)
    batch = _collate_forward_trace_from_traces([tr], _to_tensor)
    assert batch.step_mask[0, :3].tolist() == [True, True, False]

    # Buggy behavior (what the old code effectively did for step_mask)
    buggy_step_mask = torch.zeros((1, 3), dtype=torch.bool)
    buggy_step_mask[0, : len(tr.selected_idx)] = True
    assert buggy_step_mask[0].tolist() == [True, True, True]

    # This assertion is the "negative reproduction": it would fail if the bug comes back.
    assert batch.step_mask[0, :3].tolist() != buggy_step_mask[0].tolist()

import pytest


def _make_trace(S=4, T=3, stop_step=3):
    """Helper to build a valid ForwardTrace."""
    return ForwardTrace(
        K=T,
        S=S,
        selected_idx=np.arange(T, dtype=np.int64),
        candidate_scores=[np.ones(S, dtype=np.float32) for _ in range(T)],
        best_scores=np.ones(T, dtype=np.float32),
        gains=np.ones(T, dtype=np.float32),
        stop_step=stop_step,
        step_mask=np.ones(T, dtype=bool),
    )


# ---------------------------------------------------------------------------
# TC-13: test_collate_forward_trace_inconsistent_S_raises
# ---------------------------------------------------------------------------
def test_collate_forward_trace_inconsistent_S_raises():
    """Inconsistent S across traces should raise ValueError."""
    tr1 = _make_trace(S=4, T=3)
    tr2 = _make_trace(S=5, T=3)  # different S!

    with pytest.raises(ValueError, match="Inconsistent forward_trace.S"):
        _collate_forward_trace_from_traces([tr1, tr2], _to_tensor)


# ---------------------------------------------------------------------------
# TC-14: test_collate_forward_trace_empty_trace_T_zero_raises
# ---------------------------------------------------------------------------
def test_collate_forward_trace_empty_trace_T_zero_raises():
    """A trace with T=0 (empty step_mask) should raise ValueError."""
    tr = ForwardTrace(
        K=0,
        S=4,
        selected_idx=np.array([], dtype=np.int64),
        candidate_scores=[],
        best_scores=np.array([], dtype=np.float32),
        gains=np.array([], dtype=np.float32),
        stop_step=0,
        step_mask=np.array([], dtype=bool),  # length 0
    )

    with pytest.raises(ValueError):
        _collate_forward_trace_from_traces([tr], _to_tensor)
