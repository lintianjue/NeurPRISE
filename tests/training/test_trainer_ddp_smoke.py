import os
import tempfile
import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def _make_init_file() -> str:
    with tempfile.NamedTemporaryFile(prefix="ddp_init_", delete=False) as tmp:
        return tmp.name


def _build_batches() -> list[tuple[torch.Tensor, DummyLabels]]:
    return [
        (torch.ones(4), DummyLabels()),
        (torch.ones(4), DummyLabels()),
    ]


def _ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    trainer_module.unpack_batch = lambda batch, **_: batch
    trainer_module.infer_T_from_labels = lambda labels: 1

    model = DummyModel()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    trainer = Trainer(
        ddp_model,
        opt,
        metrics={},
        device="cpu",
        max_epochs=1,
        log_every_n_steps=1000,
        val_every_n_steps=1,
        amp_dtype="off",
        grad_clip=None,
        accumulate_grad_batches=1,
        is_rank0=(rank == 0),
    )

    def loss_fn(out, labels, instance=None):
        return out.mean()

    dm = DummyDM(_build_batches())
    trainer.fit(dm, problem=None, loss_fn=loss_fn)

    dist.destroy_process_group()


def test_ddp_smoke_run():
    """Motivation: DDP training with mid-epoch validation should complete without hanging."""
    world_size = 2
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    preflight_file = _make_init_file()
    try:
        try:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{preflight_file}",
                rank=0,
                world_size=1,
            )
        except Exception as exc:
            pytest.skip(f"Gloo process group not available in this environment: {exc}")
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
    finally:
        if os.path.exists(preflight_file):
            os.unlink(preflight_file)

    init_file = _make_init_file()
    try:
        mp.spawn(_ddp_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
