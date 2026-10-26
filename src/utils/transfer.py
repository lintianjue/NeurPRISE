"""Transfer weights between SEL and VC models with aligned input features.

SEL has eight node features; VC adds degree at index 8. Copy shared columns
and zero-fill added columns; dropping trailing columns requires explicit opt-in.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Pull the model tensor dict out of a checkpoint container.

    NeurPRISE `*_best.pt` files are dicts with a `model_state` entry. We also
    tolerate a few common alternative layouts and a raw state_dict.
    """
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            inner = ckpt.get(key)
            if isinstance(inner, dict) and inner and all(
                isinstance(v, torch.Tensor) for v in inner.values()
            ):
                return inner
        # Maybe it is already a raw state_dict (all values are tensors).
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Could not locate a model state_dict inside the checkpoint. "
        f"Top-level type={type(ckpt)}; keys="
        f"{list(ckpt.keys()) if isinstance(ckpt, dict) else 'n/a'}"
    )


def load_pretrained_into(
    model: nn.Module,
    ckpt_path: str,
    *,
    strict_keys: bool = True,
    allow_column_drop: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint weights into a model with compatible feature columns.

    Copy equal-shaped tensors and zero-fill added input columns; narrower inputs
    require ``allow_column_drop=True``. Other shape mismatches raise;
    ``strict_keys=True`` also rejects missing or unexpected keys.
    Updates the model and returns a report of copied, zero-filled and unmatched keys,
    plus dropped-column details only when column dropping is enabled.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = _extract_state_dict(ckpt)
    tgt_sd = model.state_dict()

    src_keys = set(src_sd.keys())
    tgt_keys = set(tgt_sd.keys())
    missing_in_src = sorted(tgt_keys - src_keys)  # target needs, source lacks
    unexpected_in_src = sorted(src_keys - tgt_keys)  # source has, target lacks

    if strict_keys and (missing_in_src or unexpected_in_src):
        raise ValueError(
            "Key set mismatch between source checkpoint and target model.\n"
            f"  missing in source (target expects): {missing_in_src}\n"
            f"  unexpected in source (target lacks): {unexpected_in_src}"
        )

    copied: List[str] = []
    zero_filled: List[Dict[str, Any]] = []
    column_dropped: List[Dict[str, Any]] = []

    new_sd = dict(tgt_sd)  # shallow copy of the ordered mapping
    for name, tgt_tensor in tgt_sd.items():
        if name not in src_sd:
            # Non-strict mode: leave target's own init in place, record it.
            continue
        src_tensor = src_sd[name]
        if tuple(src_tensor.shape) == tuple(tgt_tensor.shape):
            new_sd[name] = src_tensor.clone().to(tgt_tensor.dtype)
            copied.append(name)
            continue

        # Shape mismatch: only tolerated when they agree on every dim except the
        # last, and the target's last dim is >= source's (extra input columns).
        s_shape, t_shape = tuple(src_tensor.shape), tuple(tgt_tensor.shape)
        same_leading = len(s_shape) == len(t_shape) and s_shape[:-1] == t_shape[:-1]
        wider_last = len(s_shape) >= 1 and t_shape[-1] >= s_shape[-1]
        narrower_last = len(s_shape) >= 1 and t_shape[-1] < s_shape[-1]
        if same_leading and narrower_last and allow_column_drop:
            # Rule 2b: keep the aligned leading columns, discard the trailing
            # ones the target has no feature for.
            n_tgt = t_shape[-1]
            trimmed = src_tensor[..., :n_tgt].clone().to(tgt_tensor.dtype)
            new_sd[name] = trimmed
            column_dropped.append(
                {"name": name, "src_shape": s_shape, "tgt_shape": t_shape,
                 "dropped_cols": list(range(n_tgt, s_shape[-1]))}
            )
            continue
        if not (same_leading and wider_last):
            raise ValueError(
                f"Unhandled shape mismatch for '{name}': "
                f"source {s_shape} vs target {t_shape}. "
                "Only trailing-column widening is supported"
                + (
                    "."
                    if allow_column_drop
                    else " (pass allow_column_drop=True to also permit trailing-column "
                    "dropping, which discards learned parameters)."
                )
            )
        n_src = s_shape[-1]
        filled = tgt_tensor.clone()
        filled[..., :n_src] = src_tensor.to(tgt_tensor.dtype)
        filled[..., n_src:] = 0.0  # zero-fill the extra (degree) column(s)
        new_sd[name] = filled
        zero_filled.append(
            {"name": name, "src_shape": s_shape, "tgt_shape": t_shape,
             "zeroed_cols": list(range(n_src, t_shape[-1]))}
        )

    model.load_state_dict(new_sd, strict=True)

    report = {
        "ckpt_path": ckpt_path,
        "num_target_tensors": len(tgt_sd),
        "num_copied": len(copied),
        "num_zero_filled": len(zero_filled),
        "zero_filled": zero_filled,
        "missing_in_src": missing_in_src,
        "unexpected_in_src": unexpected_in_src,
        "src_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }

    # The column-drop fields appear ONLY on the opt-in path: consumers validate
    # this dict's exact key set, so the default report must keep its shape.
    if allow_column_drop:
        report["num_column_dropped"] = len(column_dropped)
        report["column_dropped"] = column_dropped

    logger.info(
        "load_pretrained_into: %d/%d tensors copied 1:1, %d zero-filled, "
        "%d column-dropped from %s (epoch=%s)",
        report["num_copied"], report["num_target_tensors"],
        report["num_zero_filled"], len(column_dropped),
        ckpt_path, report["src_epoch"],
    )
    for zf in zero_filled:
        logger.info(
            "  zero-filled %s: src%s -> tgt%s (zeroed cols %s)",
            zf["name"], zf["src_shape"], zf["tgt_shape"], zf["zeroed_cols"],
        )
    for cd in column_dropped:
        logger.info(
            "  column-dropped %s: src%s -> tgt%s (dropped cols %s)",
            cd["name"], cd["src_shape"], cd["tgt_shape"], cd["dropped_cols"],
        )
    if missing_in_src or unexpected_in_src:
        logger.warning(
            "load_pretrained_into key mismatches (non-strict): missing_in_src=%s unexpected_in_src=%s",
            missing_in_src, unexpected_in_src,
        )
    return report


def freeze_scenario_encoder(model: nn.Module) -> Dict[str, int]:
    """Freeze the scenario_encoder (GNN) params; leave the rest trainable.

    Returns counts so the caller can log/verify exactly what was frozen.
    """
    root = _unwrap_model(model)
    encoder = getattr(root, "encoder", None)
    scen = getattr(encoder, "scenario_encoder", None) if encoder is not None else None
    if scen is None:
        raise AttributeError(
            "Expected model.encoder.scenario_encoder to exist for freezing; "
            f"got encoder={type(encoder)}."
        )
    frozen_params = 0
    frozen_tensors = 0
    for p in scen.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
        frozen_tensors += 1
        frozen_params += p.numel()

    total_tensors = sum(1 for _ in root.parameters())
    trainable_tensors = sum(1 for p in root.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in root.parameters() if p.requires_grad)
    report = {
        "frozen_tensors": frozen_tensors,
        "frozen_params": frozen_params,
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "trainable_params": trainable_params,
    }
    logger.info(
        "freeze_scenario_encoder: froze %d tensors (%d params); "
        "trainable now %d/%d tensors (%d params).",
        frozen_tensors, frozen_params, trainable_tensors, total_tensors,
        trainable_params,
    )
    return report
