from typing import Dict, Any, Optional, ContextManager
import logging
import os
import math
import statistics
from collections import deque, OrderedDict
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from src.core.metrics import compute
from src.utils.batch_utils import unpack_batch
from src.utils.eval_utils import infer_T_from_labels


logger = logging.getLogger(__name__)


def _atomic_torch_save(obj, path: str) -> None:
    """torch.save through a same-directory temp file + os.replace: a crash mid-write leaves
    the previous file intact and no partial file behind."""
    tmp = f"{path}.partial.{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        metrics,
        callbacks=None,
        device="cuda",
        max_epochs=50,
        log_every_n_steps=50,
        val_every_n_steps=200,
        grad_clip=1.0,
        best_model_path: Optional[str] = None,
        patience: int = 0,
        improvement_threshold: float = 1e-6,
        teacher_mode: str = "none",
        teacher_prob: float = 1.0,
        teacher_key: Optional[str] = "sce_top_idx",
        teacher_in_validation: bool = False,
        grad_debug: bool = False,
        grad_debug_window: int = 50,
        amp_dtype: Optional[str] = "off",
        accumulate_grad_batches: int = 1,
        log_param_update: bool = False,
        is_rank0: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks or []
        self.device = device
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_steps = val_every_n_steps
        self.grad_clip = grad_clip
        self.best_model_path = best_model_path
        self.patience = patience
        self.improvement_threshold = improvement_threshold

        self.teacher_mode = str(teacher_mode)
        if self.teacher_mode not in ("none", "always", "prob"):
            raise ValueError(
                f"teacher_mode must be one of ('none','always','prob'), got {self.teacher_mode!r}"
            )
        self.teacher_prob = float(teacher_prob)
        logger.info(
            f"teacher_mode = {self.teacher_mode}, _use_teacher_forcing() = {self._use_teacher_forcing()}, "
            f"teacher_key = {teacher_key}"
        )
        self.teacher_key = teacher_key
        self.teacher_in_validation = bool(teacher_in_validation)

        # --- grad debug state ---
        self.grad_debug = bool(grad_debug)
        self._grad_norm_hist = deque(maxlen=int(grad_debug_window))
        self._grad_group_param_ids = None  # lazy init
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >= 1.")
        self.log_param_update = bool(log_param_update)
        self._param_update_logged = False
        self.is_rank0 = bool(is_rank0)

        self.amp_dtype = self._normalize_amp_dtype(amp_dtype)
        self.amp_enabled = self.amp_dtype != "off" and self._is_cuda()
        self._amp_torch_dtype: Optional[torch.dtype] = None
        self._grad_scaler: Optional[torch.cuda.amp.GradScaler] = None

        if self.amp_dtype != "off" and not self._is_cuda():
            logger.warning(
                "AMP requested (amp_dtype=%s) but device=%s is not CUDA; AMP disabled.",
                self.amp_dtype,
                self.device,
            )
        if self.amp_enabled:
            if self.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
                logger.warning("CUDA BF16 not supported; falling back to FP16 AMP.")
                self.amp_dtype = "fp16"
            self._amp_torch_dtype = (
                torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
            )
            if self.amp_dtype == "fp16":
                self._grad_scaler = torch.cuda.amp.GradScaler()
            logger.info(
                "AMP enabled: dtype=%s, grad_scaler=%s",
                self.amp_dtype,
                "on" if self._grad_scaler is not None else "off",
            )
        else:
            logger.info("AMP disabled: amp_dtype=%s, device=%s", self.amp_dtype, self.device)

        self.best_val_loss = float("inf")
        self.wait = 0

        # Reuse persistent loaders to avoid leaking worker resources; tear them down explicitly.
        self._train_loader: Optional[Any] = None
        self._val_loader: Optional[Any] = None

        logger.info(
            f"Trainer initialized with device {self.device}, grad_clip {self.grad_clip}, "
            f"amp_enabled={self.amp_enabled}, amp_dtype={self.amp_dtype}, "
            f"accumulate_grad_batches={self.accumulate_grad_batches}"
        )

    def _get_train_loader(self, dm) -> Any:
        """Build the train DataLoader once per `fit()` call and reuse it."""
        if self._train_loader is None:
            self._train_loader = dm.train_dataloader()
        return self._train_loader

    def _get_val_loader(self, dm) -> Any:
        """Build the val DataLoader once and reuse it across validate() calls."""
        if self._val_loader is None:
            self._val_loader = dm.val_dataloader()
        return self._val_loader

    @staticmethod
    def _shutdown_loader(loader: Any) -> None:
        """Explicitly stop a DataLoader's persistent workers/pin-memory thread.

        Relying on `del`/refcounting is not enough: a persistent-worker
        `DataLoader` iterator holds reference cycles that only the cyclic GC
        reclaims, so teardown timing is not deterministic. Call the same
        shutdown hook `DataLoader.__del__` would eventually call, directly
        and immediately, instead of hoping GC runs before FDs/threads pile up.
        """
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            return
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()

    def _teardown_dataloaders(self) -> None:
        """Explicitly tear down and drop any cached train/val DataLoaders."""
        self._shutdown_loader(self._train_loader)
        self._shutdown_loader(self._val_loader)
        self._train_loader = None
        self._val_loader = None

    @staticmethod
    def _normalize_amp_dtype(amp_dtype: Optional[str]) -> str:
        if amp_dtype is None:
            return "off"
        name = str(amp_dtype).strip().lower()
        if name in ("", "none", "off", "false", "0"):
            return "off"
        if name in ("fp16", "float16", "f16"):
            return "fp16"
        if name in ("bf16", "bfloat16"):
            return "bf16"
        raise ValueError("amp_dtype must be one of ('off','fp16','bf16').")

    def _use_teacher_forcing(self) -> bool:
        if self.teacher_mode == "none":
            return False
        if self.teacher_mode == "always":
            return True
        # "prob"
        if self.teacher_prob <= 0.0:
            return False
        if self.teacher_prob >= 1.0:
            return True
        return torch.rand(()) < self.teacher_prob

    def _extract_teacher_idx(
        self,
        labels: Any,
        *,
        T: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        teacher = getattr(labels, self.teacher_key)
        teacher = torch.as_tensor(teacher, device=device).long()

        if teacher.dim() != 2:
            logger.error(
                f"teacher_key {self.teacher_key!r} has invalid shape {teacher.shape}. Return None."
            )
            return None

        # Normalize length to exactly T if possible
        if teacher.size(1) < T:
            logger.error("Not enough steps to teacher-force all T steps; safest is to skip")
            return None
        if teacher.size(1) > T:
            teacher = teacher[:, :T]

        return teacher

    def _infer_grad_groups(self) -> Dict[str, set]:
        """
        Try to split parameters into meaningful groups:
          - score_head
          - encoder
        Prefer explicit module attributes if they exist; otherwise fall back to name patterns.
        """
        groups: Dict[str, set] = {"score_head": set(), "encoder": set()}

        # 1) Attribute-based (most reliable)
        attr_candidates = {
            "score_head": [
                "score_head",
                "decoder",
                "sel_head",
                "selection_head",
                "ranking_head",
                "head",
            ],
            "encoder": [
                "encoder",
                "sce_encoder",
                "det_encoder",
                "backbone",
                "gnn",
                "embedder",
                "scenario_encoder",
                "instance_encoder",
            ],
        }
        for gname, attrs in attr_candidates.items():
            for a in attrs:
                if hasattr(self.model, a):
                    m = getattr(self.model, a)
                    if isinstance(m, nn.Module):
                        for p in m.parameters(recurse=True):
                            groups[gname].add(id(p))

        # 2) Fallback: name-pattern based
        if len(groups["score_head"]) == 0 or len(groups["encoder"]) == 0:
            for n, p in self.model.named_parameters():
                # skip frozen params
                if not p.requires_grad:
                    continue
                ln = n.lower()
                if len(groups["score_head"]) == 0:
                    if (
                        ("decoders" in ln)
                        or ("decod" in ln)
                        or ("sel_head" in ln)
                        or ("ranking_head" in ln)
                    ):
                        groups["score_head"].add(id(p))
                if len(groups["encoder"]) == 0:
                    if ("encoder" in ln) or ("gnn" in ln) or ("embed" in ln) or ("backbone" in ln):
                        groups["encoder"].add(id(p))

        return groups

    @staticmethod
    def _safe_sqsum_grad(p: torch.Tensor) -> torch.Tensor:
        """Return sum(grad^2) as a 0-d tensor on the grad device (avoids per-param CUDA sync)."""
        if p.grad is None:
            return torch.zeros((), device=p.device)
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        return g.pow(2).sum()

    def _compute_grad_stats(self) -> Dict[str, float]:
        """
        Compute grad stats with a single GPU->CPU sync at the end (instead of per parameter).
        """
        total = 0
        none_cnt = 0

        if self._grad_group_param_ids is None:
            self._grad_group_param_ids = self._infer_grad_groups()
        gsets = self._grad_group_param_ids
        score_ids = gsets.get("score_head", set())
        enc_ids = gsets.get("encoder", set())

        # Accumulate on-device scalars (0-d tensors). Initialize lazily once we see first grad.
        sqsum = None
        sqsum_score = None
        sqsum_enc = None

        cnt_score = 0
        cnt_enc = 0

        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            total += 1
            if p.grad is None:
                none_cnt += 1
                continue

            s = self._safe_sqsum_grad(p)

            if sqsum is None:
                sqsum = torch.zeros((), device=s.device)
                sqsum_score = torch.zeros((), device=s.device)
                sqsum_enc = torch.zeros((), device=s.device)

            sqsum = sqsum + s

            pid = id(p)
            if pid in score_ids:
                sqsum_score = sqsum_score + s
                cnt_score += 1
            if pid in enc_ids:
                sqsum_enc = sqsum_enc + s
                cnt_enc += 1

        frac_none = (none_cnt / max(total, 1))

        if sqsum is None:
            global_norm = 0.0
            score_norm = 0.0
            enc_norm = 0.0
        else:
            # Single sync per stat (3 total), not per parameter
            global_norm = float(sqsum.sqrt().item())
            score_norm = float(sqsum_score.sqrt().item())
            enc_norm = float(sqsum_enc.sqrt().item())

        ratio = (enc_norm / (score_norm + 1e-12)) if (score_norm > 0.0 or enc_norm > 0.0) else 0.0
        score_rms = score_norm / math.sqrt(max(cnt_score, 1))
        enc_rms = enc_norm / math.sqrt(max(cnt_enc, 1))
        rms_ratio = (enc_rms / (score_rms + 1e-12)) if (score_rms > 0.0 or enc_rms > 0.0) else 0.0

        return {
            "grad/none_frac": float(frac_none),
            "grad/global_norm": float(global_norm),
            "grad/score_head_norm": float(score_norm),
            "grad/encoder_norm": float(enc_norm),
            "grad/enc_to_head_ratio": float(ratio),
            "grad/score_head_param_cnt_with_grad": float(cnt_score),
            "grad/encoder_param_cnt_with_grad": float(cnt_enc),
            "grad/score_head_rms": float(score_rms),
            "grad/encoder_rms": float(enc_rms),
            "grad/enc_to_head_rms_ratio": float(rms_ratio),
        }

    def _param_global_norm(self) -> float:
        sqsum = None
        for p in self.model.parameters():
            if p is None:
                continue
            s = p.detach().pow(2).sum()
            if sqsum is None:
                sqsum = torch.zeros((), device=s.device)
            sqsum = sqsum + s
        if sqsum is None:
            return 0.0
        return float(sqsum.sqrt().item())



    def _extract_oracle_mask_steps(
            self,
            labels: Any,
            out: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None

        # 2) infer from candidate_scores (+ optional step_mask)
        cand = getattr(labels, "candidate_scores")

        if cand is None or not torch.is_tensor(cand) or cand.dim() != 3:
            return None  # can't infer

        mask = torch.isfinite(cand)  # True where not -inf / not nan

        step_mask = None
        if hasattr(labels, "step_mask"):
            step_mask = getattr(labels, "step_mask")
        elif out is not None:
            if isinstance(out, dict) and "step_mask" in out:
                step_mask = out["step_mask"]
            elif hasattr(out, "step_mask"):
                step_mask = getattr(out, "step_mask")

        if step_mask is not None and torch.is_tensor(step_mask) and step_mask.dim() == 2:
            mask = mask & step_mask.to(torch.bool).unsqueeze(-1)

        return mask

    def _compute_step_mask_stats(self, oracle_mask_steps: torch.Tensor) -> Dict[str, float]:
        """Summarize valid actions at each step without diluting them with stopped samples.

        Returns a metric dict: ``active_frac`` and its alias ``has_any`` use all B samples;
        ``action_frac`` and ``num_valid`` average only active samples.
        Returns an empty dict for missing masks or masks not shaped [B, T, S].
        """
        if oracle_mask_steps is None:
            return {}

        m = oracle_mask_steps.detach()
        if m.dim() != 3:
            return {}

        m = m.bool()
        B, T, S = m.shape
        stats: Dict[str, float] = {}

        for t in range(T):
            mt = m[:, t, :]  # [B,S]
            active_b = mt.any(dim=-1)  # [B]  True if sample has any valid action at step t
            active_frac = active_b.float().mean().item()

            # For compatibility + clarity
            stats[f"mask/active_frac@{t}"] = float(active_frac)
            stats[f"mask/has_any@{t}"] = float(active_frac)

            if active_b.any():
                # Only average action fraction over active samples
                mt_active = mt[active_b]  # [B_active, S]
                stats[f"mask/action_frac@{t}"] = float(mt_active.float().mean().item())
                # Optional: number of valid actions among active samples (more interpretable sometimes)
                stats[f"mask/num_valid@{t}"] = float(mt_active.float().sum(dim=-1).mean().item())
            else:
                # No active samples => keep action_frac at 0 and num_valid at 0
                stats[f"mask/action_frac@{t}"] = 0.0
                stats[f"mask/num_valid@{t}"] = 0.0

        return stats

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        additional_state: Optional[Dict] = None,
    ):
        if not self.is_rank0:
            return
        save_path = self.best_model_path
        if not save_path:
            return
        best_dir = os.path.dirname(save_path)
        os.makedirs(best_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": self._strip_ddp_prefix(self.model.state_dict()),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": float(val_loss),
            "best_val_loss": float(self.best_val_loss),
        }
        if additional_state:
            checkpoint.update(additional_state)

        # 1) Save epoch-specific checkpoint
        _atomic_torch_save(checkpoint, save_path)

        # 2) Also save/overwrite a stable alias for eval
        best_alias_path = os.path.join(best_dir, "best.pt")
        _atomic_torch_save(checkpoint, best_alias_path)

        logger.info(
            f"  ↳ New best model saved to {save_path} (val loss {val_loss:.4f}) "
            f"and aliased as {best_alias_path}"
        )

    def _is_cuda(self) -> bool:
        d = str(self.device)
        return ("cuda" in d) and torch.cuda.is_available()

    def _peak_mem_mb(self) -> str:
        """Peak CUDA memory of this epoch (PyTorch allocator), 'allocated/reserved' in MiB."""
        if not self._is_cuda():
            return "n/a"
        mib = 2 ** 20
        return (f"{torch.cuda.max_memory_allocated(self.device) // mib}/"
                f"{torch.cuda.max_memory_reserved(self.device) // mib}")

    @staticmethod
    def _strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not any(k.startswith("module.") for k in state_dict):
            return state_dict
        remapped = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "", 1)
            remapped[k] = v
        return remapped

    def _autocast(self) -> ContextManager[None]:
        if not self.amp_enabled or self._amp_torch_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_torch_dtype)

    def _ddp_no_sync(self, should_sync: bool) -> ContextManager[None]:
        if should_sync or not hasattr(self.model, "no_sync"):
            return nullcontext()
        return self.model.no_sync()

    @staticmethod
    def _ddp_barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _eval_model(self):
        """
        Use the underlying module for rank0-only validation to avoid DDP collectives
        (e.g., buffer broadcast) when other ranks are not participating.
        """
        if (
            self.is_rank0
            and dist.is_available()
            and dist.is_initialized()
            and hasattr(self.model, "module")
        ):
            return self.model.module
        return self.model

    def _should_step(
        self,
        pending_batches: int,
        batch_idx: int,
        total_batches: Optional[int],
    ) -> bool:
        if pending_batches >= self.accumulate_grad_batches:
            return True
        if total_batches is not None and batch_idx >= total_batches:
            return True
        return False

    def _accum_backward(self, loss: torch.Tensor, *, should_sync: bool) -> None:
        loss_scaled = loss / float(self.accumulate_grad_batches)
        with self._ddp_no_sync(should_sync=should_sync):
            if self._grad_scaler is not None:
                self._grad_scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

    def _optimizer_step(self) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[float]]:
        grad_pre = None
        if self.grad_debug:
            grad_pre = self._compute_grad_stats()
            self._grad_norm_hist.append(grad_pre["grad/global_norm"])

        param_norm_pre = None
        if self.log_param_update and self.is_rank0 and not self._param_update_logged:
            param_norm_pre = self._param_global_norm()

        if self._grad_scaler is not None:
            self._grad_scaler.unscale_(self.optimizer)

        if self.grad_clip is not None:
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
        else:
            pre_clip_norm = None

        grad_post = None
        if self.grad_debug:
            grad_post = self._compute_grad_stats()

        if self._grad_scaler is not None:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if param_norm_pre is not None:
            param_norm_post = self._param_global_norm()
            logger.info(
                "param_update_check: param_norm_pre=%.6f param_norm_post=%.6f delta=%.6f",
                param_norm_pre,
                param_norm_post,
                param_norm_post - param_norm_pre,
            )
            self._param_update_logged = True

        return grad_pre, grad_post, pre_clip_norm

    def _sync(self):
        if self.profile_sync_cuda and self._is_cuda():
            torch.cuda.synchronize()

    def _ms(self, t0, t1) -> float:
        return (t1 - t0) * 1000.0

    def fit_debug(self, dm, problem, loss_fn):
        self.profile_timing = True  # enable/disable timing
        self.profile_every = 50  # log every N steps
        self.profile_sync_cuda = True  # use torch.cuda.synchronize for accurate GPU timing
        state: Dict[str, Any] = {"epoch": 0, "global_step": 0}
        first_epoch = True
        self.model.to(self.device)
        pending_batches = 0

        dl = dm.train_dataloader()
        total_batches = None
        try:
            total_batches = len(dl)
        except TypeError:
            total_batches = None
        dl_iter = iter(dl)
        batch_idx = 0

        while True:
            # --------- 0) measure dataloader fetch time (includes waiting) ----------
            do_profile = getattr(self, "profile_timing", False) and (
                        state["global_step"] % getattr(self, "profile_every", 50) == 0)

            t_fetch0 = time.perf_counter()
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            t_fetch1 = time.perf_counter()

            # --------- 1) unpack / move-to-device time (CPU + H2D copy) ----------
            t_unpack0 = time.perf_counter()
            inst, labels = unpack_batch(
                batch,
                problem=problem,
                device=torch.device(self.device),
                out_for_fallback=None,
            )
            t_unpack1 = time.perf_counter()

            # (Optional) if unpack_batch already moves to GPU, this is where it happens.
            # If inst/labels are still on CPU, you’ll see it later (or a device mismatch error).

            # --------- 2) teacher extraction (often CPU-heavy) ----------
            t_teacher0 = time.perf_counter()
            T = infer_T_from_labels(labels)
            teacher_idx = None
            if self._use_teacher_forcing():
                teacher_idx = self._extract_teacher_idx(labels, T=T, device=torch.device(self.device))
            t_teacher1 = time.perf_counter()

            # --------- 3) forward time (GPU async -> sync if profiling) ----------
            if do_profile:
                self._sync()
            t_fwd0 = time.perf_counter()
            with self._autocast():
                out = self.model(inst, problem=problem, K=None, T=T, teacher_idx=teacher_idx)
                if do_profile:
                    self._sync()
                t_fwd1 = time.perf_counter()

                # --------- 4) loss time (include sync if profiling) ----------
                if do_profile:
                    self._sync()
                t_loss0 = time.perf_counter()
                loss = loss_fn(out, labels, instance=inst)
            if do_profile:
                self._sync()
            t_loss1 = time.perf_counter()

            # --------- 5) loss.item() time (this forces sync!) ----------
            t_item0 = time.perf_counter()
            loss_scalar = float(loss.detach().item())
            t_item1 = time.perf_counter()

            # --------- 6) backward time ----------
            if pending_batches == 0:
                self.optimizer.zero_grad(set_to_none=True)

            if do_profile:
                self._sync()
            t_bwd0 = time.perf_counter()
            batch_idx += 1
            pending_next = pending_batches + 1
            will_step = self._should_step(pending_next, batch_idx, total_batches)
            self._accum_backward(loss, should_sync=will_step)
            if do_profile:
                self._sync()
            t_bwd1 = time.perf_counter()

            # --------- 7) grad stats / clip / step time ----------
            t_opt0 = time.perf_counter()
            pending_batches = pending_next
            should_step = self._should_step(pending_batches, batch_idx, total_batches)

            if should_step:
                grad_pre, grad_post, pre_clip_norm = self._optimizer_step()
                pending_batches = 0
            else:
                grad_pre = None
                grad_post = None
                pre_clip_norm = None

            if do_profile:
                self._sync()
            t_opt1 = time.perf_counter()

            # --------- 8) log breakdown ----------
            if do_profile:
                logger.info(
                    "[Timing ms] fetch=%.1f | unpack=%.1f | teacher=%.1f | fwd=%.1f | loss=%.1f | item=%.1f | bwd=%.1f | opt=%.1f",
                    self._ms(t_fetch0, t_fetch1),
                    self._ms(t_unpack0, t_unpack1),
                    self._ms(t_teacher0, t_teacher1),
                    self._ms(t_fwd0, t_fwd1),
                    self._ms(t_loss0, t_loss1),
                    self._ms(t_item0, t_item1),
                    self._ms(t_bwd0, t_bwd1),
                    self._ms(t_opt0, t_opt1),
                )

            state["global_step"] += 1

        if pending_batches > 0:
            self._optimizer_step()


    def fit(self, dm, problem, loss_fn):
        state: Dict[str, Any] = {"epoch": 0, "global_step": 0}

        self.model.to(self.device)
        epochs_bar = tqdm(
            range(self.max_epochs),
            desc="Epochs",
            leave=True,
            dynamic_ncols=True,
            disable=False,
        )
        first_epoch = True
        try:
            for epoch in epochs_bar:
                state["epoch"] = epoch
                epoch_t0 = time.perf_counter()
                if self._is_cuda():
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.model.train()
                epoch_train_loss = 0.0
                epoch_train_batches = 0
                pending_batches = 0

                train_sampler = getattr(dm, "train_sampler", None)
                if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)

                train_loader = self._get_train_loader(dm)
                total_batches = None
                try:
                    total_batches = len(train_loader)
                except TypeError:
                    total_batches = None

                for batch_idx, batch in enumerate(train_loader, start=1):
                    insts, labels = unpack_batch(
                        batch,
                        problem=problem,
                        device=torch.device(self.device),
                        out_for_fallback=None,
                    )

                    T = infer_T_from_labels(labels)
                    teacher_idx = None
                    if self._use_teacher_forcing():
                        teacher_idx = self._extract_teacher_idx(
                            labels, T=T, device=torch.device(self.device)
                        )
                        empty_teacher = (
                            teacher_idx is None
                            or (torch.is_tensor(teacher_idx) and teacher_idx.numel() == 0)
                        )
                        if empty_teacher:
                            logger.warning(
                                f"[Passing teacher label] teacher_idx is null/empty. epoch={epoch} "
                                f"T={T} label_keys={list(labels.keys()) if isinstance(labels, dict) else type(labels)}"
                            )
                        else:
                            if first_epoch and epoch_train_batches == 0:
                                n = min(10, teacher_idx.size(0))
                                logger.info(
                                    f"[Passing teacher label] teacher_idx first {n} rows:\n{teacher_idx[:n]}"
                                )
                    with self._autocast():
                        out = self.model(insts, problem=problem, K=None, T=T, teacher_idx=teacher_idx)
                        loss = loss_fn(out, labels, instance=insts)
                    epoch_train_loss += loss.item()
                    epoch_train_batches += 1

                    if pending_batches == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    pending_next = pending_batches + 1
                    will_step = self._should_step(pending_next, batch_idx, total_batches)
                    self._accum_backward(loss, should_sync=will_step)
                    pending_batches = pending_next
                    should_step = self._should_step(pending_batches, batch_idx, total_batches)

                    if should_step:
                        grad_pre, grad_post, pre_clip_norm = self._optimizer_step()
                        pending_batches = 0
                    else:
                        grad_pre = None
                        grad_post = None
                        pre_clip_norm = None

                    if self.is_rank0 and state["global_step"] % self.log_every_n_steps == 0:
                        log_payload = {"loss": f"{loss.item():.4f}"}
                        if self.grad_debug and grad_pre is not None and grad_post is not None:
                            hist = list(self._grad_norm_hist)
                            if len(hist) >= 3:
                                log_payload.update(
                                    {
                                        "grad/global_norm_min": float(min(hist)),
                                        "grad/global_norm_med": float(statistics.median(hist)),
                                        "grad/global_norm_max": float(max(hist)),
                                    }
                                )

                            log_payload.update(
                                {
                                    "grad/none_frac": float(grad_pre["grad/none_frac"]),
                                    "grad/global_norm_pre": float(grad_pre["grad/global_norm"]),
                                    "grad/global_norm_post": float(grad_post["grad/global_norm"]),
                                    "grad/score_head_norm_pre": float(
                                        grad_pre["grad/score_head_norm"]
                                    ),
                                    "grad/encoder_norm_pre": float(grad_pre["grad/encoder_norm"]),
                                    "grad/enc_to_head_ratio": float(
                                        grad_pre["grad/enc_to_head_ratio"]
                                    ),
                                    "grad/score_head_param_cnt_with_grad": float(
                                        grad_pre.get("grad/score_head_param_cnt_with_grad", 0.0)),
                                    "grad/encoder_param_cnt_with_grad": float(
                                        grad_pre.get("grad/encoder_param_cnt_with_grad", 0.0)),
                                    "grad/score_head_rms": float(grad_pre.get("grad/score_head_rms", 0.0)),
                                    "grad/encoder_rms": float(grad_pre.get("grad/encoder_rms", 0.0)),
                                    "grad/enc_to_head_rms_ratio": float(grad_pre.get("grad/enc_to_head_rms_ratio", 0.0)),
                                }
                            )
                            if pre_clip_norm is not None:
                                log_payload["grad/clip_return_pre_norm"] = float(pre_clip_norm)

                            if grad_pre["grad/none_frac"] > 0.30:
                                logger.warning(
                                    f"[grad-debug] High none_frac={grad_pre['grad/none_frac']:.3f} "
                                    f"-> potential mask/wiring issue."
                                )
                            if grad_pre["grad/global_norm"] < 1e-10:
                                logger.warning(
                                    "[grad-debug] global grad norm ~0 -> loss mask might be zeroing everything."
                                )
                            if grad_pre["grad/global_norm"] > 1e3:
                                logger.warning(
                                    f"[grad-debug] global grad norm very large ({grad_pre['grad/global_norm']:.2e}) "
                                    f"-> possible explosion/noise."
                                )
                            if (
                                grad_pre["grad/enc_to_head_ratio"] < 1e-3
                                and grad_pre["grad/score_head_norm"] > 0
                            ):
                                logger.warning(
                                    "[grad-debug] encoder grads tiny vs head "
                                    f"(enc/head={grad_pre['grad/enc_to_head_ratio']:.2e}) -> representation may not improve."
                                )
                        oracle_mask_steps = self._extract_oracle_mask_steps(labels.forward_trace, out=out)
                        if oracle_mask_steps is not None:
                            log_payload.update(self._compute_step_mask_stats(oracle_mask_steps))

                        oracle_cost_steps = self._extract_oracle_cost_steps(labels.forward_trace)
                        if oracle_cost_steps is not None:
                            log_payload.update(
                                self._compute_oracle_gap2_stats(
                                    oracle_cost_steps,
                                    valid_mask_steps=oracle_mask_steps,
                                    teacher_idx=teacher_idx,
                                )
                            )

                        logger.info(log_payload)

                    if state["global_step"] % self.val_every_n_steps == 0:
                        self._ddp_barrier()
                        val_loss = self.validate(dm, problem, loss_fn) if self.is_rank0 else None
                        state["val_loss"] = val_loss
                        self._ddp_barrier()
                    state["global_step"] += 1

                if pending_batches > 0:
                    self._optimizer_step()
                    pending_batches = 0

                should_stop = False
                self._ddp_barrier()
                if self.is_rank0:
                    val_loss = self.validate(dm, problem, loss_fn)
                    state["val_loss"] = val_loss
                    avg_train_loss = (
                        epoch_train_loss / epoch_train_batches
                        if epoch_train_batches > 0
                        else float("nan")
                    )
                    logger.info(
                        f"Epoch {epoch}: "
                        f"train_loss = {avg_train_loss:.4f}, "
                        f"val_loss = {val_loss:.4f}, "
                        f"best_val_loss = {self.best_val_loss:.4f}, "
                        f"wait = {self.wait}, "
                        f"epoch_s = {time.perf_counter() - epoch_t0:.1f}, "
                        f"peak_mem_mb = {self._peak_mem_mb()}"
                    )

                    if val_loss is not None and val_loss < self.best_val_loss - self.improvement_threshold:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, val_loss)
                        self.wait = 0
                    else:
                        self.wait += 1
                        if 0 < self.patience <= self.wait:
                            logger.info(
                                f"Early stopping at epoch {epoch} "
                                f"(best val loss {self.best_val_loss:.4f})"
                            )
                            should_stop = True
                self._ddp_barrier()

                if dist.is_available() and dist.is_initialized():
                    stop_tensor = torch.tensor(1 if should_stop else 0, device=self.device)
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
                if should_stop:
                    break
                first_epoch = False
        finally:
            self._teardown_dataloaders()

    @torch.no_grad()
    def validate(self, dm, problem, loss_fn=None):
        if not self.is_rank0 and dist.is_available() and dist.is_initialized():
            return None
        eval_model = self._eval_model()
        eval_model.eval()
        metrics_accum = []
        total_loss = 0.0
        num_batches = 0

        for batch in self._get_val_loader(dm):
            inst, labels = unpack_batch(
                batch,
                problem=problem,
                device=torch.device(self.device),
                out_for_fallback=None,
            )

            T = infer_T_from_labels(labels)
            with self._autocast():
                out = eval_model(inst, problem=problem, K=None, T=T)

            # Compute metrics only if labels available and metric names requested
            try:
                metrics_accum.append(compute(["obj", "feas"], labels))
            except Exception:
                # Keep training even if metrics module expects different shapes/fields
                pass

            if loss_fn is not None:
                with self._autocast():
                    loss = loss_fn(out, labels, instance=inst)
                total_loss += loss.item()
                num_batches += 1

        if metrics_accum:
            keys = metrics_accum[0].keys()
            avg = {k: sum(m[k] for m in metrics_accum) / len(metrics_accum) for k in keys}
            if loss_fn is not None and num_batches > 0:
                avg["loss"] = total_loss / num_batches

        for cb in self.callbacks:
            cb.on_validation_end({"metrics": metrics_accum})

        if loss_fn is not None and num_batches > 0:
            return total_loss / num_batches
        return None

    def _compute_oracle_gap2_stats(
            self,
            oracle_cost_steps: torch.Tensor,
            *,
            valid_mask_steps: Optional[torch.Tensor] = None,  # [B,T,S] bool, True=valid action
            teacher_idx: Optional[torch.Tensor] = None,  # [B,T]
    ) -> Dict[str, float]:
        """
        gap2@t = top1 - top2 (if maximize) OR top2 - top1 (if minimize), averaged over batch.
        This measures separability: small gap => label is noisy/ambiguous (top1/top2 near-tie).
        """
        if oracle_cost_steps is None:
            return {}

        c = oracle_cost_steps.detach().float()
        if c.dim() != 3:
            return {}
        B, T, S = c.shape

        if valid_mask_steps is not None:
            vm = valid_mask_steps.detach().bool()
            if vm.shape == c.shape:
                # set invalid actions to -inf/+inf depending on direction later
                pass
            else:
                vm = None
        else:
            vm = None

        # ---- infer direction (maximize vs minimize) using teacher_idx when possible ----
        maximize = True
        if teacher_idx is not None and teacher_idx.dim() == 2 and teacher_idx.shape[0] == B and teacher_idx.shape[
            1] == T:
            # compare teacher-picked cost to max/min; if closer to max -> maximize
            t = teacher_idx.detach().long().clamp_min(0).clamp_max(S - 1)
            if t.device != c.device:
                t = t.to(device=c.device)
            if t.dtype != torch.long:
                t = t.long()
            picked = c.gather(dim=2, index=t.unsqueeze(-1)).squeeze(-1)  # [B,T]

            if vm is not None:
                neg_inf = torch.finfo(c.dtype).min
                pos_inf = torch.finfo(c.dtype).max
                c_for_max = c.masked_fill(~vm, neg_inf)
                c_for_min = c.masked_fill(~vm, pos_inf)
                cmax = c_for_max.max(dim=2).values
                cmin = c_for_min.min(dim=2).values
            else:
                cmax = c.max(dim=2).values
                cmin = c.min(dim=2).values

            # count which is closer more often
            closer_to_max = (picked - cmax).abs() <= (picked - cmin).abs()
            maximize = bool(closer_to_max.float().mean().item() >= 0.5)

        stats: Dict[str, float] = {}
        stats["oracle/gap2_dir_is_max"] = 1.0 if maximize else 0.0

        # ---- compute per-step gap2 ----
        for tstep in range(T):
            ct = c[:, tstep, :]  # [B,S]
            if vm is not None:
                vmt = vm[:, tstep, :]
            else:
                vmt = None

            if maximize:
                if vmt is not None:
                    ct = ct.masked_fill(~vmt, torch.finfo(ct.dtype).min)
                top2 = torch.topk(ct, k=2, dim=1).values  # [B,2], descending
                gap = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
            else:
                if vmt is not None:
                    ct = ct.masked_fill(~vmt, torch.finfo(ct.dtype).max)
                top2 = torch.topk(-ct, k=2, dim=1).values  # top2 of (-ct) => two smallest of ct
                # top2 is values of (-ct): [-min, -second_min]
                gap = ((-top2[:, 1]) - (-top2[:, 0])).clamp_min(0.0)  # second_min - min

            stats[f"oracle/gap2@{tstep}"] = float(gap.mean().item())

        return stats

    def _extract_oracle_cost_steps(self, labels: Any) -> Optional[torch.Tensor]:
        if labels is None:
            return None

        # 4) fallback: use candidate_scores as oracle per-action scores if that's your label
        cand = labels.candidate_scores
        if cand is not None and torch.is_tensor(cand) and cand.dim() == 3:
            return cand

        return None
