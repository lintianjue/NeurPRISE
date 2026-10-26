from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def score_weighted_kl_loss(
        name: str = "score_weighted_kl_loss",
        gains_attr: str = "gains",
        temperature: float = 5.0,
        reduction: str = "batchmean",
        target_transform: str = "log",
) -> Callable:
    """Build a KL loss that distills stepwise gains into one-shot scenario probabilities.

    Read ``gains_attr`` from the trace, apply ``target_transform`` (log or sqrt),
    and normalize with temperature-scaled softmax. Returns a callable mapping
    model outputs and labels to a loss tensor with the requested reduction.
    """
    kl_div = nn.KLDivLoss(reduction=reduction)

    def loss_fn(out: Any, labels: Any, instance: Any = None) -> torch.Tensor:
        # 1. Extract Logits from Model Output
        if isinstance(out, dict):
            logits = out.get("scores", out.get("logits"))
        else:
            logits = getattr(out, "scores", getattr(out, "logits", None))

        if logits is None:
            raise ValueError("Model output must contain 'scores' or 'logits'.")

        B, S = logits.shape
        device = logits.device
        logits_f = logits.float()

        # 2. Extract Trace Data (Target Construction)
        trace = getattr(labels, "forward_trace", None)

        if trace is not None:
            # Extract fields handling both dict-like and object-like access
            if isinstance(trace, dict):
                selected_idx = trace.get("selected_idx")
                gains = trace.get(gains_attr)
                step_mask = trace.get("step_mask")
            else:
                selected_idx = getattr(trace, "selected_idx")
                gains = getattr(trace, gains_attr)
                step_mask = getattr(trace, "step_mask")

            # Helper to ensure tensor
            def to_tensor(x):
                return x if torch.is_tensor(x) else torch.tensor(x, device=device)

            selected_idx = to_tensor(selected_idx).long()  # [B, K]
            gains = to_tensor(gains).float()  # [B, K]
            step_mask = to_tensor(step_mask).bool()  # [B, K]

            # 3. Create Dense Target Vector [B, S]
            target_gains = torch.zeros((B, S), device=device, dtype=torch.float32)

            # Scatter only active steps so padded entries cannot overwrite valid gains.
            batch_indices = torch.arange(B, device=device)
            K = selected_idx.shape[1]

            for k in range(K):
                # Get data for step k
                idx_k = selected_idx[:, k]  # [B] - The scenario chosen at step k
                gain_k = gains[:, k]  # [B] - The value it added
                mask_k = step_mask[:, k]  # [B] - Is this a valid step?

                # Filter for valid batch items
                valid_b = batch_indices[mask_k]
                valid_s = idx_k[mask_k]
                valid_g = gain_k[mask_k]

                if valid_b.numel() > 0:
                    # Negative padded gains must be masked out before log1p can produce NaNs.
                    if (valid_g < 0).any():
                        bad = valid_g[valid_g < 0]
                        raise ValueError(
                            f"negative marginal gain inside the loss mask at step {k}: "
                            f"{bad.min().item():.4f} (n={bad.numel()}). The restricted "
                            "objective is monotone, so this means step_mask does not match "
                            "selected_idx>=0 for this label file -- do not train on it."
                        )
                    target_gains[valid_b, valid_s] = valid_g.float()

        else:
            # Fallback for sanity checks (e.g. Step 0 scores)
            target_gains = getattr(labels, "candidate_scores", None)
            if target_gains is None:
                raise ValueError("Could not find 'forward_trace' or 'candidate_scores' in labels.")
            if not torch.is_tensor(target_gains):
                target_gains = torch.tensor(target_gains, device=device)
            target_gains = torch.clamp(target_gains, min=0.0).float()

        # 5. Apply Transform (The "Blur" Fix)
        # Compresses the huge gap between Step 1 (500) and Step K (5)
        if target_transform == "log":
            target_gains = torch.log1p(target_gains)
        elif target_transform == "sqrt":
            target_gains = torch.sqrt(target_gains)

        # 6. Normalize Targets -> Probability Distribution
        # Softmax over the SET dimension (S)
        target_dist = F.softmax(target_gains / temperature, dim=-1)

        # 7. Normalize Predictions (Log Softmax for KL)
        pred_log_dist = F.log_softmax(logits_f, dim=-1)

        return kl_div(pred_log_dist, target_dist)

    return loss_fn
