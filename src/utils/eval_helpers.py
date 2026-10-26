import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, NamedTuple, List

import numpy as np
import torch

from src.core.problem import IProblem
from src.core.types import Labels
from src.utils.batch_utils import unpack_batch, to_device
from src.utils.eval_utils import infer_T_from_labels
from src.utils.metrics_utils import SelectionMetricsAccumulator, StepwiseTeacherMetricsAccumulator

# Optional decoding helpers
try:
    from src.utils.decoding import get_decoding_strategy
except Exception:  # pragma: no cover
    get_decoding_strategy = None

logger = logging.getLogger(__name__)


# --- Data Structures for Composition ---

class BatchConfig(NamedTuple):
    """Encapsulates batch-specific constants resolved before inference."""
    S: int
    K_eff: Optional[int]
    T: int
    is_oracle: bool


class ModelOutput(NamedTuple):
    """Standardized output from model inference."""
    scores: torch.Tensor  # [B, S] - Primary utility scores
    logits: Optional[torch.Tensor]
    logits_steps: Optional[torch.Tensor]  # [B, T, S] - Stepwise raw outputs
    pick_idx_steps: Optional[torch.Tensor]  # [B, T] - Stepwise discrete choices


# --- Stateless Helper Functions ---

def resolve_e2e_out_dir(eval_cfg, run_dir: str | Path | None = None) -> str:
    """Output dir of one end-to-end eval: <base>/<eval.name>/e2e_2ro/k_<k_ratio>.

    Pure: creates nothing. ``<base>`` is ``eval.output_dir`` when absolute; a relative one
    (the shell wrappers pass ``eval/<problem>/<tag>``) sits under ``run_dir``, so the run dir
    keeps the input-tree layout; unset/empty means ``<run_dir>/eval``. Without ``run_dir`` a
    relative ``output_dir`` resolves against the cwd.
    """
    user_out_dir = str(eval_cfg.get("output_dir", "") or "")
    eval_name = str(eval_cfg.get("name", "") or "").strip()
    k_ratio = eval_cfg.get("k_ratio", None)
    if k_ratio is None:
        raise ValueError("eval.k_ratio must be set.")

    if not user_out_dir:
        if run_dir is None:
            raise ValueError("eval.output_dir is unset; run_dir is required to default it.")
        user_out_dir = "eval"
    if run_dir is not None and not os.path.isabs(user_out_dir):
        user_out_dir = os.path.join(str(run_dir), user_out_dir)
    return os.path.join(os.path.abspath(user_out_dir), eval_name, "e2e_2ro", f"k_{k_ratio}")


def _compute_k_eff(S: int, K: Optional[int] = None, k_ratio: Optional[float] = None) -> Optional[int]:
    """Compute effective K based on inputs and scenario count S."""
    if K is not None and k_ratio is not None:
        raise ValueError("Provide at most one of K or k_ratio (not both).")

    if K is not None:
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        return min(int(K), int(S))

    if k_ratio is not None:
        if not (0.0 < float(k_ratio) <= 1.0):
            raise ValueError(f"k_ratio must be in (0,1], got {k_ratio}")
        K_eff = max(1, int(float(k_ratio) * int(S)))
        return min(K_eff, int(S))

    return None


def _extract_scalar_int(value: Any) -> Optional[int]:
    """Extract a scalar int from tensors/arrays/py scalars."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _infer_scenario_count_from_inst(inst: Any) -> Optional[int]:
    """Infer scenario count from a batched Instance."""
    data = getattr(inst, "data", None)
    if isinstance(data, dict):
        if "n_scenarios" in data:
            return _extract_scalar_int(data["n_scenarios"])
        if "sce" in data:
            return int(data["sce"].shape[1])
        if "scenarios" in data:
            return int(data["scenarios"].shape[1])
    return None


def _resolve_batch_config(
        labels: Labels,
        inst: Any,
        K: Optional[int],
        k_ratio: Optional[float],
        debug_force_teacher: bool,
        use_label_ranking: bool
) -> BatchConfig:
    """Determine S, K_eff, T, and oracle mode for the current batch."""
    # 1. Determine S (Number of Scenarios)
    scenario_count: Optional[int] = None
    if isinstance(labels, dict):
        sce_tensor = labels.get("sce_ranking")
        if sce_tensor is None:
            sce_tensor = labels.get("sce_scores")
        if sce_tensor is not None:
            scenario_count = int(sce_tensor.shape[1])
    else:
        if labels is not None:
            sce_tensor = getattr(labels, "sce_ranking", None)
            if sce_tensor is None:
                sce_tensor = getattr(labels, "sce_scores", None)
            if sce_tensor is not None:
                scenario_count = int(sce_tensor.shape[1])
            elif getattr(labels, "forward_trace", None) is not None:
                scenario_count = int(labels.forward_trace.S)

    if scenario_count is None:
        scenario_count = _infer_scenario_count_from_inst(inst)
    if scenario_count is None:
        raise ValueError("Could not infer scenario_count from labels or instance data.")

    # 2. Determine K_eff
    K_eff = _compute_k_eff(scenario_count, K=K, k_ratio=k_ratio)

    # 3. Determine T (Step Count)
    # If debugging stepwise with teacher forcing, T must match teacher path length (K_eff)
    if debug_force_teacher and K_eff:
        T = int(K_eff)
    else:
        T = infer_T_from_labels(labels)

    # Fallback: if T is None (e.g. bigmilp-only labels without forward_trace),
    # use K_eff.  The stepwise decoder takes exactly K steps to select K scenarios.
    if T is None and K_eff is not None:
        T = int(K_eff)

    return BatchConfig(S=scenario_count, K_eff=K_eff, T=T, is_oracle=use_label_ranking)


@contextmanager
def _force_teacher_mode(model: torch.nn.Module):
    """Context manager to temporarily force teacher-forcing behavior in the model."""
    decoder = getattr(model, "decoder", None)
    old_beam = getattr(decoder, "beam_eval", None)
    old_mix = getattr(decoder, "mix_teacher_p", None)

    try:
        if decoder is not None:
            if hasattr(decoder, "beam_eval"):
                decoder.beam_eval = False
            if hasattr(decoder, "mix_teacher_p"):
                decoder.mix_teacher_p = 1.0
        yield
    finally:
        if decoder is not None:
            if old_beam is not None:
                decoder.beam_eval = old_beam
            if old_mix is not None:
                decoder.mix_teacher_p = old_mix


def _run_model_inference(
        model: torch.nn.Module,
        inst: Any,
        labels: Labels,
        cfg: BatchConfig,
        problem: IProblem,
        debug_force_teacher: bool
) -> ModelOutput:
    """Executes model forward pass, optionally applying teacher forcing."""

    # Logic for Teacher Forcing (Debug Mode)
    if debug_force_teacher and cfg.K_eff:
        rank = labels.get("sce_ranking") if isinstance(labels, dict) else getattr(labels, "sce_ranking", None)

        if rank is not None:
            teacher_idx = rank[:, : int(cfg.K_eff)]
            with _force_teacher_mode(model):
                try:
                    out = model(inst, problem=problem, K=cfg.K_eff, T=int(cfg.K_eff), teacher_idx=teacher_idx)
                except TypeError:
                    out = model(inst, problem=problem, K=cfg.K_eff, T=cfg.T)
        else:
            out = model(inst, problem=problem, K=cfg.K_eff, T=cfg.T)
    else:
        # Standard Inference
        out = model(inst, problem=problem, K=cfg.K_eff, T=cfg.T)

    if "scores" not in out:
        raise KeyError("Model output missing required key 'scores'")

    return ModelOutput(
        scores=out["scores"].detach().cpu(),
        logits=out.get("logits", None),
        logits_steps=out.get("logits_steps", None),
        pick_idx_steps=out.get("pick_idx_steps", None)
    )


def _apply_teacher_guided_score_override(output: ModelOutput, T: int) -> torch.Tensor:
    """
    Debug Utility: Overwrites generic scores with 'One-Step-Ahead' predictions.
    Used when we want 'scores' to reflect the model's stepwise teacher-forced choices.
    """
    if output.logits_steps is None:
        return output.scores

    # Argmax at each step T -> [B, T]
    step_preds = torch.argmax(output.logits_steps, dim=-1).detach().cpu()

    B_local, T_local = step_preds.shape
    new_scores = torch.zeros_like(output.scores)

    # Assign descending values [T, T-1, ...] to predicted indices to preserve order
    step_values = torch.arange(T_local, 0, -1, dtype=torch.float).expand(B_local, T_local)
    new_scores.scatter_(1, step_preds, step_values)

    return new_scores


def _process_cheat_swap_worst(labels: Labels, scores: torch.Tensor, logits: Optional[torch.Tensor]) -> None:
    """In-place swap of model's worst prediction with true worst label."""
    worst_idx = getattr(labels, "sce_target_idx", None)
    if worst_idx is None:
        worst_idx = labels["sce_target_idx"]

    if not isinstance(worst_idx, torch.Tensor):
        worst_idx = torch.tensor(worst_idx, dtype=torch.long)

    worst_idx = worst_idx.detach().cpu()
    if worst_idx.dim() > 1:
        worst_idx = worst_idx.squeeze(-1)

    B = scores.shape[0]
    for b in range(B):
        w = int(worst_idx[b])
        j = int(torch.argmax(scores[b]).item())
        if j == w:
            continue

        scores[b, j], scores[b, w] = scores[b, w], scores[b, j]
        if logits is not None:
            logits[b, j], logits[b, w] = logits[b, w], logits[b, j]


def _resolve_selection_indices(
        output: ModelOutput,
        ranking_scores: torch.Tensor,  # Scores used for ranking (may be modified)
        K_eff: Optional[int],
        is_stepwise_strategy: bool
) -> Optional[torch.Tensor]:
    """
    Core Logic: Decides how to pick the Top-K indices.

    Logic Priorities:
    1. Stepwise Rollout: If model ran a rollout (pick_idx_steps), use those choices.
    2. One-Shot/Fallback: Rank the decoders matrix.
    """
    if K_eff is None:
        return None

    # Priority 1: Explicit Stepwise Choices
    # If the model effectively "walked" the graph, use its footprints.
    if output.pick_idx_steps is not None:
        # Slice to K_eff in case T > K
        return output.pick_idx_steps[:, :int(K_eff)].detach().cpu()

    # Priority 2: Score Ranking (One-Shot or Fallback)
    # Using the (potentially swapped/overridden) scores
    _, idx = torch.topk(ranking_scores, k=int(K_eff), dim=1)
    return idx.detach().cpu()


def _get_batch_size(inst: Any, scores: Optional[torch.Tensor]) -> int:
    """Robustly determine batch size from scores or Instance."""
    if scores is not None:
        return int(scores.shape[0])

    # inst is likely an Instance object; try accessing first data element
    if hasattr(inst, "data") and isinstance(inst.data, dict):
        first_val = next(iter(inst.data.values()))
        if hasattr(first_val, "shape"):
            return int(first_val.shape[0])

    # Fallback if inst is a tensor
    if hasattr(inst, "shape"):
        return int(inst.shape[0])

    return 1  # Default fallback


# --- Main Evaluation Function ---

def evaluate_loader(
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        problem: IProblem,
        K: Optional[int] = None,
        k_ratio: Optional[float] = None,
        enable_cheat_swap: bool = False,
        use_label_ranking: bool = False,
        debug_force_teacher: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a model (One-Shot or Stepwise) on a dataloader.

    Returns:
        Dict containing aggregated scores, indices, and metrics.
    """
    model.eval()

    # Strategy Detection
    raw_strategy = str(get_decoding_strategy(model)) if get_decoding_strategy else "one_shot"
    is_stepwise_strategy = "stepwise" in raw_strategy.lower()
    logger.info(f"Decoding Strategy: {raw_strategy}")

    # Accumulators
    sel_metrics_acc = SelectionMetricsAccumulator(K=K, k_ratio=k_ratio)
    tf_metrics_acc = StepwiseTeacherMetricsAccumulator(max_steps=3)

    # Results Containers
    all_results: Dict[str, List[torch.Tensor]] = {"idx": [], "scores": [], "logits": []}

    selection_time_total_sec = 0.0
    num_instances = 0
    warned_override = False

    for batch in dataloader:
        batch = to_device(batch, device)
        inst, labels = unpack_batch(batch, device=device, problem=problem)

        t_start = time.perf_counter()

        # 1. Resolve Batch Configuration (S, K_eff, T)
        cfg = _resolve_batch_config(labels, inst, K, k_ratio, debug_force_teacher, use_label_ranking)

        # 2. Run Inference or Oracle
        if cfg.is_oracle:
            # Oracle Mode: Trust labels directly
            rank = labels.get("sce_ranking") if isinstance(labels, dict) else getattr(labels, "sce_ranking")
            scores = None
            logits = None
            # Oracle indices are just the top ranked items
            idx = rank[:, :int(cfg.K_eff)].detach().cpu() if cfg.K_eff else None
            output = ModelOutput(scores=torch.zeros(1), logits=None, logits_steps=None, pick_idx_steps=None)
            pred_sce_ranking = rank
        else:
            # Model Mode — no_grad saves 25-30% GPU memory at inference
            with torch.no_grad():
                output = _run_model_inference(model, inst, labels, cfg, problem, debug_force_teacher)
            scores = output.scores
            logits = output.logits.detach().cpu() if output.logits is not None else None

            # 3. Handle Stepwise Teacher Metrics
            if output.logits_steps is not None:
                tf_metrics_acc.update(logits_steps=output.logits_steps, labels=labels.forward_trace, tie_eps=0.0)

            # 4. Handle Debug Overrides (Teacher Guided Scores)
            if debug_force_teacher and is_stepwise_strategy and output.logits_steps is not None:
                if not warned_override:
                    logger.warning("DEBUG: Overwriting scores with TEACHER-GUIDED PREDICTIONS.")
                    warned_override = True
                scores = _apply_teacher_guided_score_override(output, cfg.T)

            # 5. Handle Cheat Swap (One-Shot Post-Processing)
            if enable_cheat_swap:
                _process_cheat_swap_worst(labels, scores, logits)

            # 6. Resolve Final Selection Indices
            idx = _resolve_selection_indices(output, scores, cfg.K_eff, is_stepwise_strategy)

            pred_sce_ranking = torch.argsort(scores, dim=1, descending=True)

        selection_time_total_sec += (time.perf_counter() - t_start)

        # --- FIX: Robust Batch Size Extraction ---
        current_batch_size = _get_batch_size(inst, scores)
        num_instances += current_batch_size
        # -----------------------------------------

        if idx is not None: all_results["idx"].append(idx)
        if scores is not None: all_results["scores"].append(scores)
        if logits is not None: all_results["logits"].append(logits)

        # Update Selection Metrics
        sel_metrics_acc.update(pred_sce_ranking, labels, ranking_attr="sce_ranking", K=K)

    # Aggregation
    aggregated = {k: torch.cat(v, dim=0) if v else None for k, v in all_results.items()}

    # Compute Final Metrics
    metrics = sel_metrics_acc.compute()
    metrics.update(tf_metrics_acc.compute())

    selection_time_per_inst = selection_time_total_sec / max(1, num_instances)

    result = {
        "scores": aggregated["scores"],
        "idx": aggregated["idx"],
        "logits": aggregated["logits"],
        "selection_time_total_sec": float(selection_time_total_sec),
        "selection_time_per_instance_sec": float(selection_time_per_inst),
        "avg_inference_time_per_inst": float(selection_time_per_inst),
        "num_instances": int(num_instances),
        "metrics": metrics,
    }

    return result


def _remap_state_dict_prefix(
    state_dict: Dict[str, torch.Tensor],
    old_prefix: str,
    new_prefix: str,
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(old_prefix) for k in state_dict):
        return state_dict
    if new_prefix and any(k.startswith(new_prefix) for k in state_dict):
        return state_dict
    remapped = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            k = new_prefix + k[len(old_prefix):]
        remapped[k] = v
    return remapped


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load model state from checkpoint."""
    if not ckpt_path:
        logger.warning("Checkpoint path is empty; using random initialization.")
        return

    path_obj = Path(ckpt_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path_obj}")

    logger.info(f"Loading checkpoint from: {path_obj}")
    ckpt = torch.load(path_obj, map_location=device)

    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        state_dict = ckpt.get("model_state", ckpt.get("state_dict"))
        state_dict = _remap_state_dict_prefix(state_dict, "module.", "")
        state_dict = _remap_state_dict_prefix(state_dict, "scorer.", "decoder.")
        model.load_state_dict(state_dict, strict=True)
        epoch = ckpt.get("epoch", "NA")
        val_loss = ckpt.get("val_loss", "NA")
        logger.info(f"Loaded epoch {epoch} with val_loss={val_loss}")
    else:
        model.load_state_dict(ckpt, strict=False)
        logger.info("Loaded checkpoint as raw state dict (strict=False).")


