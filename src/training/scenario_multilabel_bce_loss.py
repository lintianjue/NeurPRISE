from typing import Callable, Optional
import torch
import torch.nn.functional as F



def scenario_multilabel_bce_loss(
    name: str = "scenario_multilabel_bce_loss",
    top_idx_attr: str = "sce_top_idx",
    trace_attr: str = "forward_trace",
    step_mask_attr: str = "step_mask",
    reduction: str = "mean",  # "mean" | "sum"
    pos_weight: Optional[float] = None,  # optional: upweight positives
) -> Callable:
    """Build multi-label BCE targets from scenarios selected at active trace steps.

    Accept logits [B, S] or [B, S, 1], selected indices [B, T] and a mask [B, T].
    Returns a callable mapping model outputs and labels to a scalar loss tensor.
    """

    assert reduction in ("mean", "sum")

    def loss_fn(out, labels, instance=None):
        # 1) logits
        if isinstance(out, dict):
            logits = out.get("logits", None)
        else:
            logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError(
                f"{name}: expected 'logits' in model output (out['logits'] or out.logits)."
            )
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.dim() != 2:
            raise RuntimeError(
                f"{name}: expected logits shape [B,S] (or [B,S,1]), got {tuple(logits.shape)}"
            )
        B, S = logits.shape

        # 2) top indices [B,T]
        top_idx = getattr(labels, top_idx_attr, None)
        if top_idx is None:
            raise RuntimeError(f"{name}: labels missing '{top_idx_attr}'.")
        top_idx = top_idx.to(device=logits.device).long()
        if top_idx.dim() != 2 or top_idx.shape[0] != B:
            raise RuntimeError(
                f"{name}: expected labels.{top_idx_attr} shape [B,T], got {tuple(top_idx.shape)}"
            )
        _, T = top_idx.shape

        # 3) step_mask [B,T]
        trace = getattr(labels, trace_attr, None)
        if trace is None:
            raise RuntimeError(f"{name}: labels missing '{trace_attr}'.")
        step_mask = getattr(trace, step_mask_attr, None)
        if step_mask is None:
            raise RuntimeError(f"{name}: labels.{trace_attr} missing '{step_mask_attr}'.")
        step_mask = step_mask.to(device=logits.device)
        if step_mask.dim() != 2 or step_mask.shape != (B, T):
            raise RuntimeError(
                f"{name}: expected step_mask shape [B,T]={B,T}, got {tuple(step_mask.shape)}"
            )
        step_mask_bool = step_mask.bool()

        # 4) build multi-hot targets y: [B,S]
        y = torch.zeros((B, S), device=logits.device, dtype=logits.dtype)

        # scatter positives for masked steps only
        # For each b, we want indices top_idx[b, step_mask[b]] set to 1.
        for b in range(B):
            idx = top_idx[b][step_mask_bool[b]]  # [num_pos_b]
            if idx.numel() > 0:
                y[b].scatter_(0, idx, 1.0)

        # 5) BCE with logits
        if pos_weight is not None:
            # pos_weight must be a tensor for BCEWithLogits; using scalar is fine
            pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        else:
            pw = None

        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction=reduction)

        # if "sum", keep it sum; if "mean", it's already averaged over B*S
        return loss

    return loss_fn