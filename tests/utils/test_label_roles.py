"""Tests for label role classification and the split invariant."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.label_roles import (
    ROLE_BOTH,
    ROLE_EVAL,
    ROLE_NEITHER,
    ROLE_TRAIN,
    SplitRoleError,
    assert_split_roles,
    instance_fingerprint,
    record_role,
    summarize_roles,
)


def make_record(*, trace=None, x=None, obj=1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "instance": SimpleNamespace(
            data={"adjacency": rng.integers(0, 2, (4, 4)), "det": rng.random(4)}
        ),
        "label": SimpleNamespace(forward_trace=trace, x=x, obj=obj),
    }


def trace_record(seed):
    """SEL/VC/CFLP forward-trace record: obj is the *reduced* value, x absent."""
    return make_record(trace={"steps": [1, 2]}, x=None, obj=3.5, seed=seed)


def eval_record(seed):
    """SEL/VC BigMILP record: optimal first-stage vector present."""
    return make_record(trace=None, x=np.ones(4), seed=seed)


def cflp_eval_record(seed):
    """CFLP BigMILP record: objective-only ground truth, x is None on all 50."""
    return make_record(trace=None, x=None, obj=12.5, seed=seed)


def complete_record(seed):
    return make_record(trace={"steps": [1]}, x=np.ones(4), seed=seed)


class TestRecordRole:
    def test_four_roles(self):
        assert record_role(trace_record(1)) == ROLE_TRAIN
        assert record_role(eval_record(2)) == ROLE_EVAL
        assert record_role(complete_record(3)) == ROLE_BOTH
        assert record_role(make_record(obj=None, seed=4)) == ROLE_NEITHER

    def test_cflp_objective_only_ground_truth_is_evaluable(self):
        """CFLP BigMILP records have x=None on all 50; regret comes from obj.

        Measured on data/cflp/..._m70_n70_s50_..._bigmilp.pkl. An x-based
        definition classifies every CFLP evaluation record as unusable.
        """
        assert record_role(cflp_eval_record(1)) == ROLE_EVAL

    def test_trace_obj_does_not_make_a_record_evaluable(self):
        """Forward-trace records populate obj with the *reduced* value."""
        assert record_role(make_record(trace={"steps": [1]}, x=None, obj=3.5)) == ROLE_TRAIN

    def test_record_with_no_targets_at_all_is_neither(self):
        assert record_role(make_record(trace=None, x=None, obj=None)) == ROLE_NEITHER

    def test_label_may_be_a_mapping(self):
        record = {"instance": SimpleNamespace(data={}), "label": {"forward_trace": {"s": 1}, "x": None}}
        assert record_role(record) == ROLE_TRAIN

    def test_summarize(self):
        records = [trace_record(1), trace_record(2), eval_record(3)]
        counts = summarize_roles(records)
        assert counts[ROLE_TRAIN] == 2
        assert counts[ROLE_EVAL] == 1
        assert counts[ROLE_BOTH] == 0


class TestInstanceFingerprint:
    def test_deterministic_and_content_sensitive(self):
        a, b = trace_record(7), trace_record(7)
        assert instance_fingerprint(a) == instance_fingerprint(b)
        assert instance_fingerprint(trace_record(7)) != instance_fingerprint(trace_record(8))

    def test_ignores_dict_key_order_but_not_values(self):
        arr1, arr2 = np.arange(4), np.arange(4.0)
        r1 = {"instance": SimpleNamespace(data={"a": arr1, "b": arr2}), "label": SimpleNamespace(forward_trace=1, x=None)}
        r2 = {"instance": SimpleNamespace(data={"b": arr2, "a": arr1}), "label": SimpleNamespace(forward_trace=1, x=None)}
        assert instance_fingerprint(r1) == instance_fingerprint(r2)

        r3 = {"instance": SimpleNamespace(data={"a": arr1 + 1, "b": arr2}), "label": SimpleNamespace(forward_trace=1, x=None)}
        assert instance_fingerprint(r1) != instance_fingerprint(r3)

    def test_dtype_change_changes_fingerprint(self):
        r_int = {"instance": SimpleNamespace(data={"a": np.arange(4, dtype=np.int64)}), "label": SimpleNamespace(forward_trace=1, x=None)}
        r_flt = {"instance": SimpleNamespace(data={"a": np.arange(4, dtype=np.float64)}), "label": SimpleNamespace(forward_trace=1, x=None)}
        assert instance_fingerprint(r_int) != instance_fingerprint(r_flt)

    def test_label_does_not_affect_fingerprint(self):
        base = trace_record(11)
        same_instance_other_label = {"instance": base["instance"], "label": SimpleNamespace(forward_trace=None, x=np.zeros(4))}
        assert instance_fingerprint(base) == instance_fingerprint(same_instance_other_label)


class TestAssertSplitRoles:
    """The layout under test is the merged one: 450 trace then 50 eval, split 400/50/50."""

    @staticmethod
    def merged_dataset(n_trace=450, n_eval=50):
        return [trace_record(i) for i in range(n_trace)] + [
            eval_record(10_000 + i) for i in range(n_eval)
        ]

    def test_merged_500_layout_passes(self):
        records = self.merged_dataset()
        counts = assert_split_roles(records, range(0, 400), range(400, 450), range(450, 500))
        assert counts[ROLE_TRAIN] == 450
        assert counts[ROLE_EVAL] == 50

    def test_small_scale_all_complete_passes(self):
        records = [complete_record(i) for i in range(100)]
        assert_split_roles(records, range(0, 80), range(80, 90), range(90, 100))

    def test_eval_record_in_train_raises(self):
        records = self.merged_dataset()
        records[10] = eval_record(99_999)
        with pytest.raises(SplitRoleError, match="train split have no forward trace"):
            assert_split_roles(records, range(0, 400), range(400, 450), range(450, 500))

    def test_eval_record_in_val_raises(self):
        records = self.merged_dataset()
        records[420] = eval_record(99_998)
        with pytest.raises(SplitRoleError, match="val split have no forward trace"):
            assert_split_roles(records, range(0, 400), range(400, 450), range(450, 500))

    def test_boundary_off_by_fifty_raises(self):
        """test_split=0.2 on the merged file: test range swallows 50 trace records."""
        records = self.merged_dataset()
        with pytest.raises(SplitRoleError, match="no true optimum"):
            assert_split_roles(records, range(0, 300), range(300, 400), range(400, 500))

    def test_neither_role_raises(self):
        records = self.merged_dataset()
        records[3] = make_record(trace=None, x=None, obj=None, seed=5)
        with pytest.raises(SplitRoleError, match="neither a forward trace nor a true optimum"):
            assert_split_roles(records, range(0, 400), range(400, 450), range(450, 500))

    def test_pure_trace_file_warns_but_does_not_raise(self, caplog):
        """Today's medium/large path: 450 trace records, inert 45-record test split.

        Must stay reproducible, so this warns rather than failing.
        """
        records = [trace_record(i) for i in range(450)]
        with caplog.at_level(logging.INFO):
            counts = assert_split_roles(records, range(0, 360), range(360, 405), range(405, 450))
        assert counts[ROLE_TRAIN] == 450
        assert any("forward-trace only" in rec.message for rec in caplog.records)

    def test_empty_test_split_is_fine(self):
        """Absolute mode with test_abs_split_number=0."""
        records = [trace_record(i) for i in range(450)]
        assert_split_roles(records, range(0, 405), range(405, 450), range(0, 0))

    def test_context_is_reported(self):
        records = self.merged_dataset()
        records[0] = eval_record(1234)
        with pytest.raises(SplitRoleError, match=r"\[my_file\.pkl\]"):
            assert_split_roles(records, range(0, 400), range(400, 450), range(450, 500), context="my_file.pkl")
