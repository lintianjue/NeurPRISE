import logging
import importlib
import os

import numpy as np
import torch
from torch import nn

def load_problem(module_path: str, kwargs: dict):
    module_name, class_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)

def _find_project_root() -> str:
    """
    Finds the project root by walking up from this file until a directory
    containing either .git or README.md is found. Falls back to the directory
    two levels up from this script if not found.
    """
    current = os.path.abspath(__file__)
    current_dir = os.path.dirname(current)
    while current_dir and current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.git')) or \
           os.path.exists(os.path.join(current_dir, 'README.md')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    # Fallback: project root assumed as repo top-level (script is under scripts/)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _to_scalar(x):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, np.ndarray):
        return float(x.item())
    return float(x)  # last resort


def to_numpy(x):
    """
    Convert various numeric formats (Tensor, list, tuple, float, np.ndarray)
    into a numpy array or numpy scalar.
    Returns None if x is None.
    """

    if x is None:
        return None

    # PyTorch tensor
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return x

    # numpy array or scalars
    if isinstance(x, np.ndarray):
        return x

    # Python list or tuple
    if isinstance(x, (list, tuple)):
        return np.asarray(x)

    # Python scalar
    if isinstance(x, (int, float)):
        return np.asarray(x)

    raise TypeError(f"Unsupported type for to_numpy: {type(x)}")


import pickle
import pprint

def load_pickle(path, limit=10):
    """Load and summarize a pickle file, with safe printing."""
    print(f"\n=== Loaded file: {path} ===")

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("\n--- Type ---")
    print(type(data))

    if isinstance(data, list):
        print(f"\n--- Length: {len(data)} entries ---")

        for idx, item in enumerate(data[:limit]):
            print(f"\n=== Entry {idx} ===")
            _summarize_entry(item)

        if len(data) > limit:
            print(f"... (truncated, showing first {limit} entries only) ...")

    else:
        print("\n--- Preview ---")
        pprint.pprint(_summarize_any(data), width=150)

    print("\n--- End of file summary ---\n")

    return data

def _safe_head(arr, k=50):
    """Return first k elements safely for any type."""
    if arr is None:
        return None
    try:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr.item()
        return arr[:k]
    except Exception:
        return arr

def _print_instance_data(data_dict):
    print("  data:")
    for k, v in data_dict.items():
        print(f"    {k}: {_summarize_any(v)}")

def _summarize_entry(entry: dict):
    """Summarize a single item: instance + label (and label.forward_trace if exists)."""

    # ------------------
    # Instance
    # ------------------
    inst = entry.get("instance")
    if inst is not None:
        nn = len(inst.graph.nodes()) if hasattr(inst, "graph") else "?"
        print(f"instance: {type(inst)}, n_nodes={nn}")
        _print_instance_data(inst.data)

    # ------------------
    # Label
    # ------------------
    lab = entry.get("label")
    if lab is not None:
        print("label:")

        # obj
        if getattr(lab, "obj", None) is not None:
            val = lab.obj.item() if hasattr(lab.obj, "item") else lab.obj
            print(f"  obj: {val}")

        # sce_target_idx
        if hasattr(lab, "sce_target_idx"):
            print(f"  sce_target_idx: {_safe_head(lab.sce_target_idx)}")

        # top K
        if hasattr(lab, "sce_top_idx"):
            print(f"  sce_top_idx:    {_safe_head(lab.sce_top_idx)}")

        # scores (may be None)
        if hasattr(lab, "sce_scores"):
            print(f"  sce_scores:     {_safe_head(lab.sce_scores)}")

        # legacy / optional meta fields
        meta = getattr(lab, "meta", None)
        if isinstance(meta, dict) and "margin" in meta:
            print(f"  margin: {meta['margin']}")

        # ------------------
        # Forward trace
        # ------------------
        fwd = getattr(lab, "forward_trace", None)
        if fwd is not None:
            print("  forward_trace:")

            # ForwardTrace dataclass
            if hasattr(fwd, "__dataclass_fields__"):
                print(f"    K: {fwd.K}")
                print(f"    S: {fwd.S}")
                print(f"    selected_idx: {_safe_head(fwd.selected_idx)}")

                # ---- NEW: dynamic-K / plateau fields ----
                if hasattr(fwd, "stop_step"):
                    print(f"    stop_step: {fwd.stop_step}")

                if hasattr(fwd, "best_scores"):
                    print(f"    best_scores: {_safe_head(fwd.best_scores)}")

                if hasattr(fwd, "gains"):
                    print(f"    gains:       {_safe_head(fwd.gains)}")

                step_mask = getattr(fwd, "step_mask", None)
                if step_mask is not None:
                    # support numpy or torch
                    if hasattr(step_mask, "detach"):
                        step_mask_np = step_mask.detach().cpu().numpy()
                    else:
                        step_mask_np = np.asarray(step_mask)
                    mask_head = _safe_head(step_mask_np.astype(int))
                    print(f"    step_mask:   {mask_head}  (1=used, 0=masked)")

                # ---- candidate decoders traces ----
                cs = getattr(fwd, "candidate_scores", None)
                if cs is not None:
                    T = len(cs)
                    print(f"    candidate_scores: list(len={T})")

                    max_steps = min(10, T)
                    for t in range(max_steps):
                        arr = cs[t]
                        if arr is None:
                            print(f"      step[{t}]: None")
                            continue

                        print(
                            f"      step[{t}]: ndarray, shape={arr.shape}, dtype={arr.dtype}"
                        )
                        print(
                            f"      step[{t}] head: {_safe_head(arr)}"
                        )

                    if T > max_steps:
                        print(f"      ... ({T - max_steps} more steps)")

            # Fallback if trace stored as dict
            elif isinstance(fwd, dict):
                for k, v in fwd.items():
                    if isinstance(v, np.ndarray):
                        print(f"    {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"    forward_trace: unsupported type {type(fwd)}")
            if fwd.step_mask is not None:
                print(f"    effective_K: {np.sum(fwd.step_mask)}")


def _summarize_any(obj):
    """Fallback summarizer."""
    if isinstance(obj, np.ndarray):
        return f"np.ndarray, shape={obj.shape}, dtype={obj.dtype}"
    if isinstance(obj, torch.Tensor):
        return f"torch.Tensor, shape={tuple(obj.shape)}, dtype={obj.dtype}"
    return obj


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    """Return number of parameters in a module."""
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def print_param_counts(model: nn.Module, name: str = "model") -> None:
    """Log parameter counts for a composite model: total and top-level blocks at INFO,
    the per-parameter breakdown at DEBUG."""
    log = logging.getLogger(__name__)
    blocks = ", ".join(f"{child_name}={count_parameters(child):,}"
                       for child_name, child in model.named_children())
    log.info("%s: %s trainable params (%s)", name, f"{count_parameters(model):,}", blocks)
    for pname, param in model.named_parameters():
        log.debug("  %-40s: %s", pname, f"{param.numel():,}")