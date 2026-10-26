"""VC->SEL is a trailing-column DROP, and it must stay opt-in.

`load_pretrained_into` already supports the SEL->VC direction: the target is one
column wider and the extra (degree) column is zero-filled.  The reverse pairing
needs the inverse move -- keep SEL's eight aligned columns, discard VC's trailing
degree column -- because SEL has no feature for it.

The reason this is gated behind `allow_column_drop` rather than always on: unlike
zero-filling, dropping a column throws away learned parameters.  A target model
that is accidentally too narrow (a config typo, a divergent `src/` tree) is a bug,
and it must keep raising.  These tests pin both halves.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.utils.transfer import load_pretrained_into


def _ckpt(tmp_path, state, name="src.pt"):
    path = tmp_path / name
    torch.save({"model_state": state, "epoch": 7}, path)
    return str(path)


class _Target(nn.Module):
    """Two first-layer projections plus a width-agnostic tensor, as in the real model."""

    def __init__(self, width: int):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.det_encoder = nn.Sequential(nn.Linear(width, 4))
        self.encoder.scenario_encoder = nn.Module()
        self.encoder.scenario_encoder.emb_mlp = nn.Sequential(nn.Linear(width, 4))
        self.head = nn.Linear(4, 1)


def _source_state(width: int) -> dict:
    torch.manual_seed(0)
    model = _Target(width)
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_narrower_target_drops_the_trailing_column_and_copies_the_rest(tmp_path):
    src = _source_state(9)
    target = _Target(8)
    report = load_pretrained_into(
        target, _ckpt(tmp_path, src), strict_keys=True, allow_column_drop=True
    )

    assert report["num_column_dropped"] == 2
    assert report["num_zero_filled"] == 0
    dropped = {d["name"]: d for d in report["column_dropped"]}
    assert set(dropped) == {
        "encoder.det_encoder.0.weight",
        "encoder.scenario_encoder.emb_mlp.0.weight",
    }
    for entry in dropped.values():
        assert entry["src_shape"] == (4, 9)
        assert entry["tgt_shape"] == (4, 8)
        assert entry["dropped_cols"] == [8]

    # The surviving columns are the SOURCE's, bit for bit -- not the target's init.
    loaded = target.state_dict()
    for name in dropped:
        assert torch.equal(loaded[name], src[name][..., :8])
    # Everything width-agnostic is copied 1:1, and biases are untouched by the trim.
    assert torch.equal(loaded["head.weight"], src["head.weight"])
    assert torch.equal(
        loaded["encoder.det_encoder.0.bias"], src["encoder.det_encoder.0.bias"]
    )


def test_column_drop_is_off_by_default(tmp_path):
    """The bug case: a too-narrow target must still raise, as it always has."""
    src = _source_state(9)
    target = _Target(8)
    with pytest.raises(ValueError) as excinfo:
        load_pretrained_into(target, _ckpt(tmp_path, src), strict_keys=True)
    message = str(excinfo.value)
    assert "Unhandled shape mismatch" in message
    assert "allow_column_drop=True" in message


def test_default_report_keeps_the_frozen_key_set(tmp_path):
    """The default report keeps exactly these eight keys.

    Reports already written were validated against this key set, so adding a
    field unconditionally would invalidate artifacts checked against the old shape.
    """
    src = _source_state(8)
    target = _Target(9)
    report = load_pretrained_into(target, _ckpt(tmp_path, src), strict_keys=True)
    assert set(report) == {
        "ckpt_path", "num_target_tensors", "num_copied", "num_zero_filled",
        "zero_filled", "missing_in_src", "unexpected_in_src", "src_epoch",
    }


def test_widening_still_zero_fills_and_reports_no_drop(tmp_path):
    """The SEL->VC direction is unchanged, including when the new flag is on."""
    src = _source_state(8)
    target = _Target(9)
    report = load_pretrained_into(
        target, _ckpt(tmp_path, src), strict_keys=True, allow_column_drop=True
    )
    assert report["num_zero_filled"] == 2
    assert report["num_column_dropped"] == 0
    loaded = target.state_dict()
    for name in ("encoder.det_encoder.0.weight",
                 "encoder.scenario_encoder.emb_mlp.0.weight"):
        assert torch.equal(loaded[name][..., :8], src[name])
        assert torch.all(loaded[name][..., 8:] == 0.0)
