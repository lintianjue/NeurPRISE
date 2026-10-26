import tempfile
from pathlib import Path

import torch

from src.training.trainer import Trainer
import src.training.trainer as trainer_module


class DummyDM:
    def __init__(self, batches):
        self._batches = batches

    def train_dataloader(self):
        return self._batches

    def val_dataloader(self):
        return self._batches[:1]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inst, problem=None, K=None, T=None, teacher_idx=None):
        return self.w * inst


def test_trainer_gradient_accumulation_steps(monkeypatch):
    """Motivation: optimizer steps should happen every N micro-batches."""
    monkeypatch.setattr(trainer_module, "unpack_batch", lambda batch, **_: batch)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)

    class DummyLabels:
        def __init__(self):
            self.forward_trace = None
            self.candidate_scores = None

    batches = [
        (torch.ones(4), DummyLabels()),
        (torch.ones(4), DummyLabels()),
        (torch.ones(4), DummyLabels()),
    ]
    dm = DummyDM(batches)

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    step_count = {"n": 0}
    orig_step = opt.step

    def wrapped_step(*args, **kwargs):
        step_count["n"] += 1
        return orig_step(*args, **kwargs)

    opt.step = wrapped_step

    with tempfile.TemporaryDirectory() as tmp_dir:
        best_path = Path(tmp_dir) / "best.pt"
        trainer = Trainer(
            model,
            opt,
            metrics={},
            device="cpu",
            max_epochs=1,
            best_model_path=str(best_path),
            log_every_n_steps=1000,
            val_every_n_steps=1000,
            amp_dtype="off",
            grad_clip=None,
            accumulate_grad_batches=2,
        )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    trainer.fit(dm, problem=None, loss_fn=loss_fn)

    assert step_count["n"] == 2


def test_should_step_logic(monkeypatch):
    """Motivation: accumulation should step on Nth batch or final batch."""
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1000,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=2,
    )

    assert trainer._should_step(1, batch_idx=1, total_batches=4) is False
    assert trainer._should_step(2, batch_idx=2, total_batches=4) is True
    assert trainer._should_step(1, batch_idx=4, total_batches=4) is True
