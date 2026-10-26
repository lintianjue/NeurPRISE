"""Checkpoints are written atomically (src/training/trainer.py::_atomic_torch_save)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import src.training.trainer as trainer_module
from src.training.trainer import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inst, problem=None, K=None, T=None, teacher_idx=None):
        return self.w * inst


def _make_trainer(tmp_path: Path, **extra) -> Trainer:
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        best_model_path=str(tmp_path / "m_best.pt"),
        log_every_n_steps=1000,
        val_every_n_steps=1000,
        amp_dtype="off",
        grad_clip=None,
        is_rank0=True,
        **extra,
    )


def _partial_files(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.iterdir() if ".partial." in p.name]


def test_crash_mid_write_keeps_previous_checkpoint_and_no_partial(tmp_path: Path, monkeypatch) -> None:
    """(M) replace _atomic_torch_save(checkpoint, save_path) by torch.save(checkpoint, save_path)."""
    save_path = tmp_path / "m_best.pt"
    alias_path = tmp_path / "best.pt"
    save_path.write_bytes(b"old-save-bytes")
    alias_path.write_bytes(b"old-alias-bytes")

    def crashing_save(obj, path):
        with open(path, "wb") as f:
            f.write(b"xxx")
        raise RuntimeError("disk full")

    monkeypatch.setattr(trainer_module.torch, "save", crashing_save)

    trainer = _make_trainer(tmp_path)
    with pytest.raises(RuntimeError, match="disk full"):
        trainer._save_checkpoint(epoch=1, val_loss=0.5)

    assert save_path.read_bytes() == b"old-save-bytes"
    assert alias_path.read_bytes() == b"old-alias-bytes"
    assert _partial_files(tmp_path) == []


def test_successful_save_replaces_both_files_and_leaves_no_temp(tmp_path: Path) -> None:
    """(M) delete the os.replace line (files never appear)."""
    trainer = _make_trainer(tmp_path)
    trainer._save_checkpoint(epoch=1, val_loss=0.5)

    save_path = tmp_path / "m_best.pt"
    alias_path = tmp_path / "best.pt"
    for path in (save_path, alias_path):
        assert path.is_file()
        loaded = torch.load(path, weights_only=False)
        assert loaded["epoch"] == 1
    assert _partial_files(tmp_path) == []
