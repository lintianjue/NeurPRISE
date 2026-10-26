import logging

import torch



from dataclasses import dataclass
from typing import Optional, Dict, List, Any

@dataclass
class SelectionMetricsAccumulator:
    K: int = 5
    k_ratio: float = 0.1
    total: int = 0
    correct_top1: float = 0.0

    hit_at_K_sum: float = 0.0
    hit_at_K_den: int = 0

    set_intersection_sum: float = 0.0
    set_intersection_den: float = 0.0



    def update(
        self,
        pred_ranking: torch.Tensor,     # [B, S], model-predicted ranking (indices ordered worst→best)
        labels,                         # batched Labels
        ranking_attr: str = "sce_ranking",
        K: Optional[int] = None,
    ) -> None:
        """
        Update metrics using only predicted and true rankings.

        pred_ranking[b] is assumed to be a permutation of [0..S-1],
        ordered from worst (index 0) to best (index S-1).
        labels.<ranking_attr> should have the same shape and semantics.
        """
        if not hasattr(labels, ranking_attr):
            return

        true_ranking = getattr(labels, ranking_attr)
        if true_ranking is None:
            logging.info(f"true ranking missing from labels, skipping metrics update")
            return
        true_ranking = torch.as_tensor(true_ranking, device=pred_ranking.device)

        if pred_ranking.shape != true_ranking.shape:
            return  # shape mismatch → skip

        B, S = pred_ranking.shape
        self.total += B

        # --- 1) worst-scenario top-1 accuracy ---
        pred_worst = pred_ranking[:, 0]            # [B]
        true_worst = true_ranking[:, 0]            # [B]
        self.correct_top1 += (pred_worst == true_worst).float().sum().item()

        # --- 2) worst-scenario recall@K ---
        if self.k_ratio is not None:
            K_eff = max(1, (int)(self.k_ratio * S))
        else:
            K_eff = min(max(1, K), S)
        pred_topK = pred_ranking[:, :K_eff]        # [B, K_eff]
        true_worst_exp = true_worst.unsqueeze(1)   # [B, 1]
        hit = (pred_topK == true_worst_exp).any(dim=1).float()  # [B]
        self.hit_at_K_sum += hit.sum().item()
        self.hit_at_K_den += B

        # --- 3) set precision/recall/F1 on worst-K set ---
        true_topK = true_ranking[:, :K_eff]        # [B, K_eff]

        # match_matrix[b, i, j] = (pred_topK[b, i] == true_topK[b, j])
        pred_exp = pred_topK.unsqueeze(2)          # [B, K_eff, 1]
        true_exp = true_topK.unsqueeze(1)          # [B, 1, K_eff]
        match_matrix = (pred_exp == true_exp)      # [B, K_eff, K_eff]

        # for each predicted index, check if it appears anywhere in true_topK
        intersection_counts = match_matrix.any(dim=2).float().sum(dim=1)  # [B]

        self.set_intersection_sum += intersection_counts.sum().item()
        self.set_intersection_den += float(B * K_eff)

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"num_instances": float(self.total)}
        if self.total > 0:
            metrics["worst_top1_acc"] = self.correct_top1 / self.total
        if self.hit_at_K_den > 0:
            metrics["worst_recall_at_K"] = self.hit_at_K_sum / self.hit_at_K_den
        if self.set_intersection_den > 0:
            prec_rec = self.set_intersection_sum / self.set_intersection_den
            metrics["set_precision_at_K"] = prec_rec
            metrics["set_recall_at_K"] = prec_rec
            metrics["set_F1_at_K"] = prec_rec
        return metrics


def _rank_desc(x: torch.Tensor) -> torch.Tensor:
    # x: [N]
    # return ranks 0..N-1 where 0 means largest
    return torch.argsort(torch.argsort(-x))

def _spearman_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [N] float, no NaNs
    ra = _rank_desc(a).float()
    rb = _rank_desc(b).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.pow(2).mean().sqrt() * rb.pow(2).mean().sqrt()).clamp_min(1e-12)
    return float((ra * rb).mean() / denom)




@dataclass
class StepwiseTeacherMetricsAccumulator:
    """Accumulate teacher-forced accuracy, tie accuracy, oracle gaps and rank correlation.

    Mask the teacher prefix and include only active rows when ``step_mask`` is given;
    Spearman correlation requires at least three valid actions.
    """
    max_steps: int = 3
    acc_sum: List[float] = None
    tie_acc_sum: List[float] = None
    gap_sum: List[float] = None
    den_sum: List[int] = None

    spearman_sum: List[float] = None
    spearman_den: List[int] = None

    def __post_init__(self):
        self.acc_sum = [0.0] * self.max_steps
        self.tie_acc_sum = [0.0] * self.max_steps
        self.gap_sum = [0.0] * self.max_steps
        self.den_sum = [0] * self.max_steps
        self.spearman_sum = [0.0] * self.max_steps
        self.spearman_den = [0] * self.max_steps

        self.oracle_rank_sum = [0.0] * self.max_steps
        self.oracle_rank_den = [0] * self.max_steps
        self.top2_hit_sum = [0.0] * self.max_steps
        self.top3_hit_sum = [0.0] * self.max_steps

    @staticmethod
    def _inverse_rank_from_scores(scores: torch.Tensor) -> torch.Tensor:
        """
        scores: [B,S] (masked invalid already, e.g. -inf)
        returns inv_rank: [B,S] where 0 is best, 1 is second, ...
        """
        order = torch.argsort(scores, dim=1, descending=True)  # [B,S]
        inv = torch.empty_like(order)
        ar = torch.arange(order.size(1), device=scores.device).unsqueeze(0).expand_as(order)
        inv.scatter_(1, order, ar)
        return inv

    @staticmethod
    def _tie_aware_topk_hit(
            oracle_scores: torch.Tensor,  # [B,S] with -inf invalid
            pred_idx: torch.Tensor,  # [B]
            k: int,
            tie_eps: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns hit [B] bool: pred is in top-k set, tie-aware at boundary.
        """
        topk_vals = torch.topk(oracle_scores, k=k, dim=1).values  # [B,k]
        kth = topk_vals[:, -1]  # [B]
        pred_score = oracle_scores.gather(1, pred_idx.unsqueeze(1)).squeeze(1)  # [B]
        return pred_score >= (kth - float(tie_eps))

    @staticmethod
    def _build_invalid_from_prefix(teacher_idx: torch.Tensor, t: int, S: int) -> torch.Tensor:
        # teacher_idx: [B,T]
        device = teacher_idx.device
        B = teacher_idx.shape[0]
        invalid = torch.zeros((B, S), dtype=torch.bool, device=device)
        if t > 0:
            pref = teacher_idx[:, :t].clamp(0, S - 1)
            invalid.scatter_(1, pref, True)
        return invalid

    @staticmethod
    def _safe_top2_gap(oracle_t: torch.Tensor) -> torch.Tensor:
        # oracle_t: [B,S] with -inf masked invalid
        # If fewer than 2 valid remain, topk will still return something, but could be -inf.
        top2 = torch.topk(oracle_t, k=2, dim=1).values  # [B,2]
        gap = top2[:, 0] - top2[:, 1]
        # if second best is -inf => gap is inf; clamp to 0 in that degenerate case
        gap = torch.where(torch.isfinite(gap), gap, torch.zeros_like(gap))
        return gap

    def update_core(
        self,
        logits_steps: torch.Tensor,                 # [B,T,S]
        teacher_idx: torch.Tensor,                  # [B,T]
        oracle_scores_steps: Optional[torch.Tensor] = None,  # [B,T,S]
        step_mask: Optional[torch.Tensor] = None,   # [B,T] bool
        tie_eps: float = 0.0,
    ) -> None:
        if logits_steps is None:
            return
        B, T, S = logits_steps.shape
        max_t = min(self.max_steps, T)

        teacher_idx = teacher_idx.to(device=logits_steps.device).long().clamp(0, S - 1)
        if oracle_scores_steps is not None:
            oracle_scores_steps = oracle_scores_steps.to(device=logits_steps.device).float()
        if step_mask is not None:
            step_mask = step_mask.to(device=logits_steps.device).bool()

        for t in range(max_t):
            active = torch.ones((B,), dtype=torch.bool, device=logits_steps.device)
            if step_mask is not None:
                active = step_mask[:, t]
            if not active.any():
                continue

            invalid = self._build_invalid_from_prefix(teacher_idx, t, S)
            logits_t = logits_steps[:, t, :].masked_fill(invalid, float("-inf"))
            pred_best = torch.argmax(logits_t, dim=1)

            # exact match vs teacher at step t
            teacher_t = teacher_idx[:, t]
            m = (pred_best == teacher_t) & active
            self.acc_sum[t] += float(m.float().sum().item())
            self.den_sum[t] += int(active.sum().item())

            if oracle_scores_steps is None:
                continue

            oracle_t = oracle_scores_steps[:, t, :].masked_fill(invalid, float("-inf"))

            # oracle rank of model's predicted top-1 (1-based rank)
            inv_rank = self._inverse_rank_from_scores(oracle_t)  # [B,S], 0 best
            pred_rank0 = inv_rank.gather(1, pred_best.unsqueeze(1)).squeeze(1).float() + 1.0  # [B]
            pred_rank0 = pred_rank0.masked_fill(~active, 0.0)
            self.oracle_rank_sum[t] += float(pred_rank0.sum().item())
            self.oracle_rank_den[t] += int(active.sum().item())

            # tie-aware top-2 / top-3 hit
            top2_hit = self._tie_aware_topk_hit(oracle_t, pred_best, k=2, tie_eps=tie_eps) & active
            top3_hit = self._tie_aware_topk_hit(oracle_t, pred_best, k=3, tie_eps=tie_eps) & active
            self.top2_hit_sum[t] += float(top2_hit.float().sum().item())
            self.top3_hit_sum[t] += float(top3_hit.float().sum().item())

            # gap
            gap = self._safe_top2_gap(oracle_t)
            gap = gap.masked_fill(~active, 0.0)
            self.gap_sum[t] += float(gap.sum().item())

            # tie-aware hit: pred in argmax set
            best_val = oracle_t.max(dim=1).values
            is_max = oracle_t >= (best_val.unsqueeze(1) - float(tie_eps))
            tie_hit = is_max.gather(1, pred_best.unsqueeze(1)).squeeze(1) & active
            self.tie_acc_sum[t] += float(tie_hit.float().sum().item())

            # spearman over valid actions (>=3)
            for b in torch.where(active)[0].tolist():
                valid_mask = ~invalid[b]
                a = logits_steps[b, t, valid_mask].float()
                o = oracle_t[b, valid_mask].float()
                if a.numel() >= 3:
                    self.spearman_sum[t] += _spearman_1d(a, o)
                    self.spearman_den[t] += 1

    # Backward-compatible adapter (keeps your current call sites)
    def update(
        self,
        logits_steps: torch.Tensor,  # [B,T,S]
        labels: Any,
        teacher_idx_attr: str = "selected_idx",
        oracle_scores_attr: str = "candidate_scores",
        step_mask_attr: Optional[str] = "step_mask",
        tie_eps: float = 0.0,
    ) -> None:
        if logits_steps is None or not hasattr(labels, teacher_idx_attr):
            return
        teacher_idx = torch.as_tensor(getattr(labels, teacher_idx_attr), device=logits_steps.device)

        oracle_scores_steps = None
        if oracle_scores_attr and hasattr(labels, oracle_scores_attr):
            oracle_scores_steps = torch.as_tensor(getattr(labels, oracle_scores_attr), device=logits_steps.device)

        step_mask = None
        if step_mask_attr and hasattr(labels, step_mask_attr):
            step_mask = torch.as_tensor(getattr(labels, step_mask_attr), device=logits_steps.device)

        self.update_core(
            logits_steps=logits_steps,
            teacher_idx=teacher_idx,
            oracle_scores_steps=oracle_scores_steps,
            step_mask=step_mask,
            tie_eps=tie_eps,
        )

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t in range(self.max_steps):
            den = self.den_sum[t]
            if den <= 0:
                continue
            out[f"tf_step_acc@{t}"] = self.acc_sum[t] / den
            if self.tie_acc_sum[t] > 0.0 or self.gap_sum[t] > 0.0 or self.spearman_den[t] > 0:
                out[f"tf_step_tie_acc@{t}"] = self.tie_acc_sum[t] / den
                out[f"tf_oracle_gap@{t}"] = self.gap_sum[t] / den
                if self.spearman_den[t] > 0:
                    out[f"tf_spearman@{t}"] = self.spearman_sum[t] / self.spearman_den[t]
            if self.oracle_rank_den[t] > 0:
                out[f"tf_oracle_rank_of_pred@{t}"] = self.oracle_rank_sum[t] / self.oracle_rank_den[t]
                out[f"tf_top2_hit@{t}"] = self.top2_hit_sum[t] / self.oracle_rank_den[t]
                out[f"tf_top3_hit@{t}"] = self.top3_hit_sum[t] / self.oracle_rank_den[t]
        return out

