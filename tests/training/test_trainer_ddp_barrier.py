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


class DummyLabels:
    def __init__(self):
        self.forward_trace = None
        self.candidate_scores = None


def _setup_ddp_monkeypatch(monkeypatch, barrier_counter):
    monkeypatch.setattr(trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)

    def fake_barrier():
        barrier_counter["n"] += 1

    monkeypatch.setattr(trainer_module.dist, "barrier", fake_barrier)
    monkeypatch.setattr(trainer_module.dist, "broadcast", lambda *args, **kwargs: None)


def test_ddp_barrier_runs_for_non_rank0_validation(monkeypatch):
    """Motivation: non-rank0 workers must wait at validation barriers to avoid DDP desync."""
    monkeypatch.setattr(trainer_module, "unpack_batch", lambda batch, **_: batch)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)

    batches = [
        (torch.ones(4), DummyLabels()),
        (torch.ones(4), DummyLabels()),
    ]
    dm = DummyDM(batches)

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    barrier_counter = {"n": 0}
    _setup_ddp_monkeypatch(monkeypatch, barrier_counter)

    validate_calls = {"n": 0}

    def fake_validate(self, dm, problem, loss_fn=None):
        validate_calls["n"] += 1
        return 0.0

    monkeypatch.setattr(Trainer, "validate", fake_validate)

    trainer = Trainer(
        model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=1,
        is_rank0=False,
    )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    trainer.fit(dm, problem=None, loss_fn=loss_fn)

    assert validate_calls["n"] == 0
    assert barrier_counter["n"] == 6


def test_ddp_barrier_runs_for_rank0_validation(monkeypatch):
    """Motivation: rank0 should run validation while all ranks synchronize via barriers."""
    monkeypatch.setattr(trainer_module, "unpack_batch", lambda batch, **_: batch)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)

    batches = [
        (torch.ones(4), DummyLabels()),
        (torch.ones(4), DummyLabels()),
    ]
    dm = DummyDM(batches)

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    barrier_counter = {"n": 0}
    _setup_ddp_monkeypatch(monkeypatch, barrier_counter)

    validate_calls = {"n": 0}

    def fake_validate(self, dm, problem, loss_fn=None):
        validate_calls["n"] += 1
        return 0.0

    monkeypatch.setattr(Trainer, "validate", fake_validate)

    trainer = Trainer(
        model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=1,
        is_rank0=True,
    )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    trainer.fit(dm, problem=None, loss_fn=loss_fn)

    assert validate_calls["n"] == 3
    assert barrier_counter["n"] == 6
