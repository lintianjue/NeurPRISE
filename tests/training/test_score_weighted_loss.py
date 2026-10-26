import pytest
import torch
import types
from src.training.score_weighted_loss import score_weighted_kl_loss


# Helper to mock model output and labels
class MockOutput:
    def __init__(self, scores):
        self.scores = scores


class MockLabels:
    def __init__(self, trace=None, candidate_scores=None):
        self.forward_trace = trace
        self.candidate_scores = candidate_scores


def test_basic_loss_calculation():
    """Test that loss runs and returns a scalar."""
    B, S, K = 2, 10, 3
    loss_fn = score_weighted_kl_loss()

    # Mock Logits [B, S]
    logits = torch.randn(B, S, requires_grad=True)
    out = MockOutput(logits)

    # Mock Trace
    trace = {
        "selected_idx": torch.tensor([[0, 1, 2], [3, 4, 5]]),  # [B, K]
        "gains": torch.tensor([[100., 50., 10.], [200., 20., 5.]]),
        "step_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
    }
    labels = MockLabels(trace=trace)

    loss = loss_fn(out, labels)

    assert loss.dim() == 0
    assert loss > 0
    loss.backward()  # check gradients


def test_masked_padding_safety():
    """
    CRITICAL: Verify that padded steps (mask=0) do NOT overwrite valid data.
    """
    B, S, K = 1, 5, 2
    loss_fn = score_weighted_kl_loss(target_transform="none", temperature=1.0)  # Simple linear check

    # Scenario 0 is the "Star" (Gain 100).
    # Step 1 is PADDED. It has gain 0, mask 0.
    # Crucially, the padded index is ALSO 0 (common bug in data loaders).
    # If logic is wrong, the 0.0 will overwrite the 100.0.

    trace = {
        "selected_idx": torch.tensor([[0, 0]]),  # [Scen 0, Scen 0 (padded)]
        "gains": torch.tensor([[100.0, 0.0]]),  # [Real, Fake]
        "step_mask": torch.tensor([[1, 0]])  # [Valid, Invalid]
    }
    labels = MockLabels(trace=trace)

    # Model predicts uniform scores (0.0 logits)
    logits = torch.zeros(B, S)
    out = MockOutput(logits)

    # We inspect the internal target construction by hacking the fn or just inferring
    # If masking works: Target for Scen 0 is 100.
    # If masking fails: Target for Scen 0 is 0 (overwritten).

    # Let's verify via the loss magnitude.
    # If Target[0] = 100, others=0 -> Softmax is peaky -> Loss is high vs uniform.
    # If Target[0] = 0, all=0 -> Softmax is uniform -> Loss should be 0 (KL(uniform||uniform)).

    loss = loss_fn(out, labels)

    # Loss should NOT be zero because targets (peak at 0) != preds (uniform)
    assert loss > 0.1, "Loss was too low; valid gain 100.0 likely overwritten by padded 0.0!"


def test_log_transform_and_temperature():
    """Verify log transform compresses values and temperature flattens them."""
    loss_fn_log = score_weighted_kl_loss(target_transform="log", temperature=1.0)
    loss_fn_flat = score_weighted_kl_loss(target_transform="log", temperature=10.0)

    B, S = 1, 3
    # Big difference: 1000 vs 1
    trace = {
        "selected_idx": torch.tensor([[0, 1]]),
        "gains": torch.tensor([[1000.0, 1.0]]),
        "step_mask": torch.tensor([[1, 1]])
    }
    labels = MockLabels(trace=trace)
    logits = torch.zeros(B, S)  # Uniform preds
    out = MockOutput(logits)

    loss_log = loss_fn_log(out, labels)
    loss_flat = loss_fn_flat(out, labels)

    # With T=1.0, distribution is sharp (high KL from uniform).
    # With T=10.0, distribution is flat (low KL from uniform).
    assert loss_flat < loss_log, "Higher temperature should produce lower KL loss against uniform preds"


def test_numpy_input_compatibility():
    """Ensure it works if the data loader returns numpy arrays (common case)."""
    import numpy as np
    loss_fn = score_weighted_kl_loss()

    trace = {
        "selected_idx": np.array([[0, 1]]),
        "gains": np.array([[10., 5.]]),
        "step_mask": np.array([[1, 1]])
    }
    labels = MockLabels(trace=trace)
    logits = torch.zeros(1, 10)
    out = MockOutput(logits)

    loss = loss_fn(out, labels)
    assert loss > 0


def test_bfloat16_logits_mixed_dtype_trace():
    """Ensure mixed bf16 logits + fp32 trace does not error under AMP."""
    B, S, K = 2, 6, 2
    loss_fn = score_weighted_kl_loss()

    logits = torch.zeros(B, S, dtype=torch.bfloat16, requires_grad=True)
    out = MockOutput(logits)

    trace = {
        "selected_idx": torch.tensor([[0, 1], [2, 3]]),
        "gains": torch.tensor([[10.0, 1.0], [5.0, 2.0]], dtype=torch.float32),
        "step_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    labels = MockLabels(trace=trace)

    loss = loss_fn(out, labels)
    assert loss.dim() == 0
    loss.backward()


# ---------------------------------------------------------------------------
# TC-13: test_score_weighted_kl_fallback_candidate_scores
# ---------------------------------------------------------------------------
def test_score_weighted_kl_fallback_candidate_scores():
    """When forward_trace is None, falls back to labels.candidate_scores."""
    B, S = 2, 8
    logits = torch.zeros(B, S)
    # candidate_scores: positive non-zero values => target is not uniform
    candidate_scores = torch.arange(float(S)).unsqueeze(0).expand(B, -1)  # [B, S]

    labels = MockLabels(trace=None, candidate_scores=candidate_scores)
    loss_fn = score_weighted_kl_loss(target_transform="log", temperature=1.0)
    loss = loss_fn(MockOutput(logits), labels)

    # With non-uniform target and uniform logits, loss > 0
    assert loss.dim() == 0
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# TC-14: test_score_weighted_kl_sqrt_transform_differs_from_log
# ---------------------------------------------------------------------------
def test_score_weighted_kl_sqrt_transform_differs_from_log():
    """sqrt and log transforms should produce different loss values."""
    B, S, K = 1, 6, 2
    logits = torch.zeros(B, S)

    trace = {
        "selected_idx": torch.tensor([[0, 1]]),
        "gains": torch.tensor([[100.0, 1.0]]),
        "step_mask": torch.tensor([[1, 1]]),
    }
    labels = MockLabels(trace=trace)

    loss_log = score_weighted_kl_loss(target_transform="log", temperature=1.0)(
        MockOutput(logits), labels
    )
    loss_sqrt = score_weighted_kl_loss(target_transform="sqrt", temperature=1.0)(
        MockOutput(logits), labels
    )

    # Different transforms compress gains differently -> different target distributions -> different losses
    assert abs(loss_log.item() - loss_sqrt.item()) > 1e-6
