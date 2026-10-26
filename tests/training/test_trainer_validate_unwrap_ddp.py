import torch

from src.training.trainer import Trainer
import src.training.trainer as trainer_module


class DummyDM:
    def __init__(self, batches):
        self._batches = batches

    def val_dataloader(self):
        return self._batches


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


class FakeDDP:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._eval_called = False
        self._forward_called = False

    def eval(self):
        self._eval_called = True
        return self

    def __call__(self, *args, **kwargs):
        self._forward_called = True
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()


def test_validate_uses_underlying_module_for_rank0_ddp(monkeypatch):
    """Motivation: rank0-only validation should avoid DDP collectives by using module."""
    monkeypatch.setattr(trainer_module, "unpack_batch", lambda batch, **_: batch)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)
    monkeypatch.setattr(trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)

    batches = [(torch.ones(4), DummyLabels())]
    dm = DummyDM(batches)

    model = DummyModel()
    ddp_model = FakeDDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        ddp_model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1000,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=1,
        is_rank0=True,
    )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    loss = trainer.validate(dm, problem=None, loss_fn=loss_fn)

    assert loss is not None
    assert ddp_model._eval_called is False
    assert ddp_model._forward_called is False


def test_validate_avoids_broadcast_on_rank0_ddp(monkeypatch):
    """Motivation: rank0-only validation should not trigger DDP broadcasts."""
    monkeypatch.setattr(trainer_module, "unpack_batch", lambda batch, **_: batch)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)
    monkeypatch.setattr(trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)

    broadcast_calls = {"n": 0}

    def fake_broadcast(*args, **kwargs):
        broadcast_calls["n"] += 1
        return None

    monkeypatch.setattr(trainer_module.dist, "broadcast", fake_broadcast)

    class BroadcastDDP(FakeDDP):
        def __call__(self, *args, **kwargs):
            trainer_module.dist.broadcast(None, src=0)
            return super().__call__(*args, **kwargs)

    batches = [(torch.ones(4), DummyLabels())]
    dm = DummyDM(batches)

    model = DummyModel()
    ddp_model = BroadcastDDP(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        ddp_model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1000,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=1,
        is_rank0=True,
    )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    loss = trainer.validate(dm, problem=None, loss_fn=loss_fn)

    assert loss is not None
    assert broadcast_calls["n"] == 0
    assert ddp_model._forward_called is False
