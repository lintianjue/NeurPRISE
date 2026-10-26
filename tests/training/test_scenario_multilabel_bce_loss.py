import types
import torch
import torch.nn.functional as F

from src.training.scenario_multilabel_bce_loss import scenario_multilabel_bce_loss


def _make_labels(sce_top_idx, step_mask):
    """
    Create a minimal labels object:
      labels.sce_top_idx: LongTensor [B,T]
      labels.forward_trace.step_mask: Bool/IntTensor [B,T]
    """
    labels = types.SimpleNamespace()
    labels.sce_top_idx = torch.tensor(sce_top_idx, dtype=torch.long)
    trace = types.SimpleNamespace()
    trace.step_mask = torch.tensor(step_mask)
    labels.forward_trace = trace
    return labels


def test_multilabel_bce_targets_follow_step_mask():
    # B=1, S=5, T=3
    # sce_top_idx says [1, 3, 4], but step_mask uses only first two -> positives {1,3}
    logits = torch.tensor([[0.0, 2.0, -1.0, 1.0, 0.5]], dtype=torch.float32)  # [1,5]
    labels = _make_labels(sce_top_idx=[[1, 3, 4]], step_mask=[[1, 1, 0]])

    loss_fn = scenario_multilabel_bce_loss()
    loss = loss_fn({"logits": logits}, labels)

    # Build expected y manually: [0,1,0,1,0]
    y = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)
    expected = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")

    assert torch.allclose(loss, expected, atol=1e-6), (loss.item(), expected.item())


def test_multilabel_bce_dynamic_num_positives_zero_positive_case():
    # step_mask all zero -> y is all zeros
    logits = torch.tensor([[0.0, 2.0, -1.0, 1.0]], dtype=torch.float32)  # [1,4]
    labels = _make_labels(sce_top_idx=[[0, 2]], step_mask=[[0, 0]])

    loss_fn = scenario_multilabel_bce_loss()
    loss = loss_fn({"logits": logits}, labels)

    y = torch.zeros_like(logits)
    expected = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")

    assert torch.allclose(loss, expected, atol=1e-6)


def test_multilabel_bce_batch_two_instances_different_masks():
    # B=2, S=6, T=4
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, -1.0, 0.5, 0.0],   # b0
            [1.0, -2.0, 0.0, 0.0, 3.0, -1.0],  # b1
        ],
        dtype=torch.float32,
    )
    labels = _make_labels(
        sce_top_idx=[
            [3, 4, 1, 0],   # b0 top list
            [4, 2, 5, 1],   # b1 top list
        ],
        step_mask=[
            [1, 0, 1, 0],   # b0 positives {3,1}
            [1, 1, 0, 0],   # b1 positives {4,2}
        ],
    )

    loss_fn = scenario_multilabel_bce_loss()
    loss = loss_fn({"logits": logits}, labels)

    y = torch.zeros_like(logits)
    y[0, 3] = 1.0
    y[0, 1] = 1.0
    y[1, 4] = 1.0
    y[1, 2] = 1.0

    expected = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# TC-11: test_multilabel_bce_pos_weight_inflates_loss
# ---------------------------------------------------------------------------
def test_multilabel_bce_pos_weight_inflates_loss():
    """pos_weight > 1.0 should increase loss relative to pos_weight=None."""
    B, S = 1, 5
    logits = torch.zeros(B, S)   # uniform prediction
    # one positive at index 2
    labels = _make_labels(sce_top_idx=[[2, 0]], step_mask=[[1, 0]])

    loss_no_pw = scenario_multilabel_bce_loss(pos_weight=None)({"logits": logits}, labels)
    loss_pw = scenario_multilabel_bce_loss(pos_weight=5.0)({"logits": logits}, labels)

    # pos_weight inflates the positive BCE term => higher loss
    assert loss_pw.item() > loss_no_pw.item()


# ---------------------------------------------------------------------------
# TC-12: test_multilabel_bce_3d_logits_squeezed
# ---------------------------------------------------------------------------
def test_multilabel_bce_3d_logits_squeezed():
    """Logits with shape [B, S, 1] are squeezed to [B, S] before BCE."""
    B, S = 2, 4
    # Shape [B, S, 1]
    logits_3d = torch.zeros(B, S, 1)
    labels = _make_labels(sce_top_idx=[[1, 3], [0, 2]], step_mask=[[1, 1], [1, 0]])

    loss_fn = scenario_multilabel_bce_loss()
    # Should not raise a RuntimeError (would fail if 3D not squeezed)
    loss = loss_fn({"logits": logits_3d}, labels)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0
