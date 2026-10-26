from typing import Optional, List
import numpy as np
import torch
from src.utils.batch_utils import _build_reduced_instance_from_np
from src.utils.utils import _to_scalar
from src.core.types import Instance

def _compute_K_from_ratio(S: int, k_ratio: float) -> int:
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")
    K = max(1, int(k_ratio * S))
    return min(K, S)

def _eval_single_reduced_instance(
    problem,
    data_np,
    sce_idx,
    full_obj_val,
    milp_time_limit,
    threads,
    verbose=False,
    mip_gap=None,
    return_solve_evidence: bool = False,
):
    solver_backend = str(getattr(problem, "exact_backend", "bigmilp")).lower()
    full_obj_source = str(getattr(problem, "eval_full_obj_source", "label")).lower()
    if full_obj_source == "solve" or solver_backend == "ccg":
        full_inst = Instance(data=dict(data_np))
        full_labels = problem.solve_exact(
            full_inst,
            time_limit=milp_time_limit,
            threads=threads,
            verbose=verbose,
        )
        full_obj_val = _to_scalar(full_labels.obj)
    else:
        full_obj_val = _to_scalar(full_obj_val)

    reduced_inst = _build_reduced_instance_from_np(problem, data_np, sce_idx)
    labels_red = problem.solve_exact(
        reduced_inst,
        time_limit=milp_time_limit,
        threads=threads,
        verbose=verbose,
        mip_gap=mip_gap,
    )
    red_obj_val = _to_scalar(labels_red.obj)
    regret_val = red_obj_val - full_obj_val
    result = (
        full_obj_val,
        red_obj_val,
        regret_val,
        labels_red.feasible,
        labels_red.obj_bound,
        labels_red.mip_gap,
    )
    if not return_solve_evidence:
        return result
    evidence = {
        "status": labels_red.meta.get("status") if labels_red.meta else None,
        "feasible": labels_red.feasible,
        "obj_bound": labels_red.obj_bound,
        "mip_gap": labels_red.mip_gap,
    }
    return (*result, evidence)

def infer_T_from_labels(labels) -> Optional[int]:
    """
    Returns T (number of steps) for stepwise training/eval.

    Supports:
      - labels.forward_trace.selected_idx: Tensor [B,T] or [T] or np.ndarray
      - labels.sce_top_idx: Tensor [B,K] fallback
    """
    fwd = getattr(labels, "forward_trace", None)
    if fwd is not None:
        sel = getattr(fwd, "selected_idx", None)
        if sel is not None:
            if torch.is_tensor(sel):
                if sel.dim() == 2:
                    return int(sel.size(1))
                if sel.dim() == 1:
                    return int(sel.numel())
            # numpy / list fallback
            try:
                return int(len(sel))
            except Exception:
                pass

    # fallback for non-stepwise labels
    topk = getattr(labels, "sce_top_idx", None)
    if topk is not None and torch.is_tensor(topk) and topk.dim() == 2:
        return int(topk.size(1))

    return None

def _compute_plateau_stop_step(
    best_scores: List[float] | np.ndarray,
    *,
    stop_M: int = 1,
    eps_gain: float = 0.0,
) -> Optional[int]:
    """
    Pure helper for unit tests.

    Returns:
        stop_step (0-indexed) when the plateau condition first triggers,
        or None if no plateau triggers within the provided best_scores.
    """
    if stop_M < 1:
        raise ValueError("stop_M must be >= 1")

    plateau_count = 0
    stop_step: Optional[int] = None

    # idx = 0..T-1 (0-indexed)
    for idx, score in enumerate(best_scores):
        score = float(score)
        if idx == 0:
            gain = score
        else:
            gain = score - float(best_scores[idx - 1])

        if gain <= eps_gain:
            plateau_count += 1
        else:
            plateau_count = 0

        if stop_step is None and plateau_count >= stop_M:
            stop_step = idx  # 0-indexed label matching the trace convention

    return stop_step

def _compute_step_mask(T: int, *, stop_step: int) -> np.ndarray:
    """Return a boolean mask [T] that includes steps before ``stop_step``.

    Steps are zero-based; stop_step=0 masks all steps and stop_step=T keeps all.
    """
    T = int(T)
    stop_step = int(stop_step)

    if T < 0:
        raise ValueError("T must be >= 0")
    if stop_step < 0 or stop_step > T:
        raise ValueError(f"stop_step must be in [0..T], got stop_step={stop_step}, T={T}")

    mask = np.ones((T,), dtype=bool)
    if stop_step < T:
        mask[stop_step:] = False
    return mask
