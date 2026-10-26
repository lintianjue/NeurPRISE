from typing import Any, Callable
import torch
import torch.nn.functional as F


def log_bce_loss(
        name: str = "log_bce_loss",
        gains_attr: str = "gains",
        target_transform: str = "log",
        epsilon: float = 1e-8,
        reduction: str = "mean",
) -> Callable:
    """Build a BCE loss that gives higher-gain selected scenarios more weight.

    Transform and normalize positive marginal gains; negative targets keep unit weight.
    Returns a callable mapping model outputs and labels to a scalar loss tensor.
    """
    assert reduction in ("mean", "sum")

    def loss_fn(out: Any, labels: Any, instance: Any = None) -> torch.Tensor:
        # 1. Extract logits
        if isinstance(out, dict):
            logits = out.get("scores", out.get("logits"))
        else:
            logits = getattr(out, "scores", getattr(out, "logits", None))

        if logits is None:
            raise ValueError("Model output must contain 'scores' or 'logits'.")

        B, S = logits.shape
        device = logits.device
        logits_f = logits.float()

        # 2. Extract trace data
        trace = getattr(labels, "forward_trace", None)

        if trace is not None:
            if isinstance(trace, dict):
                selected_idx = trace.get("selected_idx")
                gains = trace.get(gains_attr)
                step_mask = trace.get("step_mask")
            else:
                selected_idx = getattr(trace, "selected_idx")
                gains = getattr(trace, gains_attr)
                step_mask = getattr(trace, "step_mask")

            def to_tensor(x):
                return x if torch.is_tensor(x) else torch.tensor(x, device=device)

            selected_idx = to_tensor(selected_idx).long()   # [B, K]
            gains = to_tensor(gains).float()                 # [B, K]
            step_mask = to_tensor(step_mask).bool()          # [B, K]

            # 3. Build binary multi-hot targets [B, S] and gain weights [B, S]
            y = torch.zeros((B, S), device=device, dtype=torch.float32)
            gain_weights = torch.zeros((B, S), device=device, dtype=torch.float32)

            batch_indices = torch.arange(B, device=device)
            K = selected_idx.shape[1]

            for k in range(K):
                idx_k = selected_idx[:, k]
                gain_k = gains[:, k]
                mask_k = step_mask[:, k]

                valid_b = batch_indices[mask_k]
                valid_s = idx_k[mask_k]
                valid_g = gain_k[mask_k]

                if valid_b.numel() > 0:
                    y[valid_b, valid_s] = 1.0
                    gain_weights[valid_b, valid_s] = valid_g

        else:
            target_gains = getattr(labels, "candidate_scores", None)
            if target_gains is None:
                raise ValueError(
                    "Could not find 'forward_trace' or 'candidate_scores' in labels."
                )
            if not torch.is_tensor(target_gains):
                target_gains = torch.tensor(target_gains, device=device)
            target_gains = torch.clamp(target_gains, min=0.0).float()
            y = (target_gains > 0).float()
            gain_weights = target_gains

        # 4. Apply log compression to gain weights
        if target_transform == "log":
            gain_weights = torch.log1p(gain_weights)
        elif target_transform == "sqrt":
            gain_weights = torch.sqrt(gain_weights)

        # 5. Normalise gain weights to [0, 1] per sample
        max_vals = gain_weights.max(dim=-1, keepdim=True).values.clamp(min=epsilon)
        gain_weights = gain_weights / max_vals

        # Selected scenarios use gain weights; negatives retain unit weight.
        elem_weights = torch.ones((B, S), device=device, dtype=torch.float32)
        elem_weights = torch.where(y > 0, gain_weights, elem_weights)

        # 7. Weighted BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits_f, y, reduction="none",
        )
        weighted_bce = bce * elem_weights

        if reduction == "mean":
            return weighted_bce.mean()
        else:
            return weighted_bce.sum()

    return loss_fn
