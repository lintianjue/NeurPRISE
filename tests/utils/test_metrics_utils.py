import torch
import pytest

from src.utils.metrics_utils import StepwiseTeacherMetricsAccumulator, SelectionMetricsAccumulator
from src.core.types import Labels


def test_tf_step_acc_and_gap_step0():
    # B=2, T=1, S=4
    logits_steps = torch.tensor([[[0.1, 2.0, 0.0, -1.0]],
                                 [[1.0, 0.0, 0.0, 0.0]]])  # [2,1,4]
    teacher_idx = torch.tensor([[1],
                                [0]])  # perfect matches

    oracle_scores = torch.tensor([[[10.0, 9.0, 0.0, -5.0]],
                                 [[3.0, 1.0, 0.0, -1.0]]])  # [2,1,4]
    # gaps: (10-9)=1, (3-1)=2 => mean gap = 1.5

    acc = StepwiseTeacherMetricsAccumulator(max_steps=1)
    acc.update_core(logits_steps, teacher_idx, oracle_scores_steps=oracle_scores)
    m = acc.compute()

    assert pytest.approx(m["tf_step_acc@0"], 1e-6) == 1.0
    assert pytest.approx(m["tf_oracle_gap@0"], 1e-6) == 1.5


def test_tie_acc_allows_argmax_set():
    # B=1, T=1, S=4
    # model predicts index 2, teacher is index 1
    logits_steps = torch.tensor([[[0.0, 0.1, 5.0, 0.0]]])
    teacher_idx = torch.tensor([[1]])

    # oracle has a tie between 1 and 2
    oracle_scores = torch.tensor([[[0.0, 10.0, 10.0, 0.0]]])

    acc = StepwiseTeacherMetricsAccumulator(max_steps=1)
    acc.update_core(logits_steps, teacher_idx, oracle_scores_steps=oracle_scores, tie_eps=0.0)
    m = acc.compute()

    assert pytest.approx(m["tf_step_acc@0"], 1e-6) == 0.0
    assert pytest.approx(m["tf_step_tie_acc@0"], 1e-6) == 1.0


def test_step_mask_excludes_rows():
    # B=2, T=1, S=3
    logits_steps = torch.tensor([[[0.0, 2.0, 0.0]],
                                 [[0.0, 0.0, 3.0]]])
    teacher_idx = torch.tensor([[1],
                                [1]])  # row1 is "wrong"

    step_mask = torch.tensor([[True],
                              [False]])  # only first row active

    acc = StepwiseTeacherMetricsAccumulator(max_steps=1)
    acc.update_core(logits_steps, teacher_idx, step_mask=step_mask)
    m = acc.compute()

    # only row0 counted: correct => 1.0
    assert pytest.approx(m["tf_step_acc@0"], 1e-6) == 1.0


def test_prefix_invalid_mask_changes_step1_argmax():
    # Here we ensure step1 cannot pick something already in teacher prefix.
    # B=1, T=2, S=4
    teacher_idx = torch.tensor([[3, 0]])  # prefix at t=1 includes {3}

    # At step1 logits would prefer index 3 (already selected), but should be masked out and choose 0
    logits_steps = torch.tensor([[
        [0.0, 0.0, 0.0, 10.0],   # step0: pick 3
        [0.0, 1.0, 2.0, 9.0],    # step1: argmax is 3, but invalid => should pick 2
    ]])

    acc = StepwiseTeacherMetricsAccumulator(max_steps=2)
    acc.update_core(logits_steps, teacher_idx)

    m = acc.compute()
    # step0 pred=3 matches teacher 3
    assert pytest.approx(m["tf_step_acc@0"], 1e-6) == 1.0
    # step1 pred should be 2 (since 3 invalid), but teacher is 0 => mismatch
    assert pytest.approx(m["tf_step_acc@1"], 1e-6) == 0.0


def test_spearman_perfect_positive():
    # B=1, T=1, S=5
    logits_steps = torch.tensor([[[1., 2., 3., 4., 5.]]])
    teacher_idx = torch.tensor([[4]])

    oracle_scores = torch.tensor([[[10., 20., 30., 40., 50.]]])  # same ordering

    acc = StepwiseTeacherMetricsAccumulator(max_steps=1)
    acc.update_core(logits_steps, teacher_idx, oracle_scores_steps=oracle_scores)
    m = acc.compute()

    assert "tf_spearman@0" in m
    assert m["tf_spearman@0"] > 0.99


# ---------------------------------------------------------------------------
# TC-11: test_selection_metrics_accumulator_compute_empty
# ---------------------------------------------------------------------------
def test_selection_metrics_accumulator_compute_empty():
    """Before any updates, compute() returns only num_instances=0."""
    acc = SelectionMetricsAccumulator(K=5, k_ratio=0.1)
    m = acc.compute()
    assert m["num_instances"] == 0.0
    # No accuracy/recall keys before any updates
    assert "worst_top1_acc" not in m
    assert "worst_recall_at_K" not in m


# ---------------------------------------------------------------------------
# TC-12: test_selection_metrics_accumulator_full_match
# ---------------------------------------------------------------------------
def test_selection_metrics_accumulator_full_match():
    """When predicted ranking perfectly matches true ranking, all metrics = 1.0."""
    B, S = 2, 8
    # Predicted ranking: descending from 0..S-1
    pred_ranking = torch.arange(S).unsqueeze(0).expand(B, -1)  # [B, S]
    # True ranking identical
    true_ranking = pred_ranking.clone()

    labels = Labels(sce_ranking=true_ranking)

    acc = SelectionMetricsAccumulator(K=3, k_ratio=0.5)
    acc.update(pred_ranking, labels, ranking_attr="sce_ranking")
    m = acc.compute()

    assert m["num_instances"] == float(B)
    assert m["worst_top1_acc"] == pytest.approx(1.0, abs=1e-6)
    assert m["worst_recall_at_K"] == pytest.approx(1.0, abs=1e-6)
