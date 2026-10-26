# src/training/batch_utils.py

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass as dataclass_is_dataclass

import torch
from omegaconf import DictConfig

# Adjust this import to wherever you defined Instance / Labels
from src.core.types import Instance, Labels
from src.utils.omega_conf import create_problem_module
import numpy as np


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if dataclass_is_dataclass(x):
        # e.g., ForwardTraceBatch: it contains tensors but doesn't implement .to()
        kwargs = {f.name: to_device(getattr(x, f.name), device) for f in dataclass_fields(type(x))}
        return type(x)(**kwargs)
    if hasattr(x, "to") and callable(getattr(x, "to")):
        return x.to(device)
    return x

def to_device_instance(inst: Instance, device: torch.device) -> Instance:
    """
    Move all tensor values inside Instance.data to the given device.
    Assumes that collate has already converted arrays to tensors.
    """
    return Instance(
        data={k: to_device(v, device) for k, v in inst.data.items()}
    )


def to_device_labels(labels: Labels, device: torch.device) -> Optional[Labels]:
    if labels is None:
        return None
    kwargs = {}
    for f in dataclass_fields(Labels):
        val = getattr(labels, f.name)
        kwargs[f.name] = to_device(val, device)
    return Labels(**kwargs)


def unpack_batch(
    batch: Union[Dict[str, Any], Tuple[Any, Any], Instance],
    device: torch.device,
    problem=None,
    out_for_fallback=None,
) -> Tuple[Instance, Optional[Labels]]:
    """
    Supports:
      - dict:  {"instance": Instance, "label": Labels | None}
      - tuple/list: (Instance, Labels)
      - Instance (no labels)
    """
    inst: Instance
    labels: Optional[Labels] = None

    # 1) Basic structural unpack
    if isinstance(batch, dict):
        inst = batch["instance"]
        labels = batch.get("label", None)
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        inst, labels = batch
    else:
        inst, labels = batch, None

    # 2) Move to device
    inst = to_device_instance(inst, device)
    labels = to_device_labels(labels, device)

    return inst, labels


def get_loader_from_cfg(cfg: DictConfig):
    dm, problem = create_problem_module(cfg)
    eval_cfg = cfg.get("eval", {})
    split = eval_cfg.get("split", "test")

    if split == "train":
        loader = dm.train_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    elif split == "test":
        loader = dm.test_dataloader()
    elif split == "all":
        loader = dm.full_dataloader()
    elif split == 'val_test':
        loader = dm.val_test_dataloader()
    else:
        raise ValueError(f"Unknown eval.split='{split}'")

    return loader, problem, split


def _instance_to_numpy_dict(inst: Instance, idx_in_batch: int) -> Dict[str, Any]:
    """
    Take a batched Instance (tensors with leading batch dim)
    and extract a single-instance dict with numpy arrays.
    """
    data_np: Dict[str, Any] = {}
    for k, v in inst.data.items():
        # v is typically a torch.Tensor with shape [B, ...]
        if torch.is_tensor(v):
            # pick the sample and move to CPU numpy
            v_b = v[idx_in_batch].detach().cpu().numpy()
            data_np[k] = v_b
        else:
            # not a tensor (rare), just keep as-is
            data_np[k] = v
    return data_np


def _make_reduced_instance(full_inst: Instance, sce_idx: np.ndarray) -> Instance:
    """
    Given a batched Instance (but we will slice one sample beforehand),
    construct a *single-instance* reduced-scenario Instance:

    - sce: keep only sce[sce_idx, :]
    - n_scenarios: updated to K
    """
    if sce_idx.ndim != 1:
        raise ValueError(f"Expected sce_idx to be 1D (K,), got shape {sce_idx.shape}")

    # We'll first convert the single sample to numpy dict
    # and then modify 'sce' and 'n_scenarios'.
    # NOTE: this function assumes you've already done per-sample extraction.
    raise RuntimeError(
        "_make_reduced_instance expects per-sample numpy dict; "
        "call _build_reduced_instance() instead."
    )


def _build_reduced_instance_from_np(problem, data_np: Dict[str, Any], sce_idx: np.ndarray) -> Instance:
    """
    Same idea as _make_reduced_instance, but expects a single-instance
    dict of numpy arrays (no batch dimension).

    Returns a new Instance with:
      - sce reduced to sce[sce_idx]
      - n_scenarios updated
    """
    sce_full = problem.get_scenarios(data_np)  # shape [S, N] (or [S, ...])
    if sce_full.ndim < 2:
        raise ValueError(f"Expected 'sce' or 'scenarios' to have ndim >= 2, got shape {sce_full.shape}")

    # Defensive: ensure sce_idx is a 1D numpy array of integers
    sce_idx = np.asarray(sce_idx, dtype=int)
    if sce_idx.ndim != 1:
        raise ValueError(f"sce_idx must be 1D, got {sce_idx.shape}")

    sce_reduced = sce_full[sce_idx]  # [K, N]
    K = sce_reduced.shape[0]

    data_red = dict(data_np)
    if "sce" in data_red:
        data_red["sce"] = sce_reduced
    if "scenarios" in data_red:
        data_red["scenarios"] = sce_reduced
    if "d" in data_red:
        data_red["d"] = sce_reduced
    if "demand" in data_red:
        data_red["demand"] = sce_reduced
    # Update n_scenarios if present
    if "n_scenarios" in data_red:
        data_red["n_scenarios"] = np.array([K], dtype=int)

    return Instance(data=data_red)
