"""The current Trainer DataLoader lifecycle, as it applies to CFLP fine-tuning.

Context
-------
Fine-tuning takes the uniform-trained seed-2 CFLP model on the frozen 5% (100-record)
shifted train split.  Its frozen training shape is:

    data.batch_size=32   data.num_workers=4   train.max_epochs=200   train.patience=10

`src/training/trainer.py` carries the loader-leak repair: `fit()` builds one train and
one validation DataLoader per `fit()` call, reuses them across epochs, and tears them
down explicitly in a `finally` block.  The previously pinned runtime rebuilt both loaders
every epoch / every `validate()` call.

**Active for fine-tuning:** the DataLoader reuse/teardown itself.  Because it trains with
    `num_workers=4` (so `persistent_workers=True`), reuse changes how much of the global
    torch RNG stream `fit()` consumes per epoch, which changes the *shuffled train batch
    order from the second epoch onward* relative to the pinned runtime.

Mechanism (measured, see `test_..._builds_one_iterator...` below): with persistent
workers the pinned path constructed a fresh `_MultiProcessingDataLoaderIter` every epoch,
and each construction draws a `_base_seed` from the global generator *in addition to* the
`RandomSampler` permutation draw.  The reused iterator is `_reset()` instead of rebuilt,
so it draws only the permutation.  One fewer global draw per epoch => a different, but
still fully deterministic and reproducible, epoch-2+ shuffle.

These tests therefore do not claim trace-equivalence with the pinned runtime.  They
establish what fine-tuning actually gets: bounded worker lifetime, deterministic teardown,
unchanged epoch-1 order, unchanged validation stream, unchanged early-stopping/checkpoint
selection, a provably inert change on the `num_workers=0` evaluation lane, and exact
run-to-run reproducibility of the new order.

All tests are CPU-only, bounded, and use tiny synthetic data/models: no science
training, no evaluation, no CUDA, no Gurobi.  `pin_memory` is forced to False
(fine-tuning uses True) because allocating page-locked memory initialises a CUDA context; the pin-memory
thread is torn down by the same `_shutdown_workers()` hook these tests exercise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter

import src.training.trainer as trainer_module
from src.training.trainer import Trainer

psutil = pytest.importorskip("psutil")


# --------------------------------------------------------------------------
# Fine-tuning-shaped fixtures: 100 train records / batch 32 / 4 workers, shuffled.
# --------------------------------------------------------------------------

FINETUNE_TRAIN_RECORDS = 100
FINETUNE_VAL_RECORDS = 32
FINETUNE_BATCH_SIZE = 32
FINETUNE_NUM_WORKERS = 4
FINETUNE_SEED = 42
# Validation records are numbered from this offset so a recorded batch identifies its
# own split; `validate()` and `fit()` share the same `unpack_batch` recording hook.
VAL_INDEX_OFFSET = 1000


class DummyLabels:
    def __init__(self) -> None:
        self.forward_trace = None
        self.candidate_scores = None


class IndexDataset(Dataset):
    """Each record is its own index, so a batch is a readable order receipt."""

    def __init__(self, size: int, offset: int = 0) -> None:
        self._size = size
        self._offset = offset

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index + self._offset


def _collate(indices: List[int]) -> Tuple[torch.Tensor, DummyLabels]:
    # Shape (B, 2) so the tiny Linear(2, 1) model below accepts it; column 0
    # carries the record index verbatim.
    column = torch.tensor(indices, dtype=torch.float32).unsqueeze(1)
    return torch.cat([column, column], dim=1), DummyLabels()


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inst, problem=None, K=None, T=None, teacher_idx=None):
        return self.linear(inst)


def _loss_fn(out, labels, instance=None):
    return out.mean()


class FinetuneDataModule:
    """Mirrors `src.training.datamodule.CFLPDataModule`'s loader construction for the
    frozen fine-tuning knobs: train shuffled, validation ordered, both with persistent workers
    whenever `num_workers > 0`."""

    def __init__(self, num_workers: int = FINETUNE_NUM_WORKERS) -> None:
        self.num_workers = num_workers
        self.train_ds = IndexDataset(FINETUNE_TRAIN_RECORDS)
        self.val_ds = IndexDataset(FINETUNE_VAL_RECORDS, offset=VAL_INDEX_OFFSET)
        self.train_calls = 0
        self.val_calls = 0

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=FINETUNE_BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        self.train_calls += 1
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self.val_calls += 1
        return self._loader(self.val_ds, shuffle=False)


def _make_finetune_trainer(model: nn.Module, /, **overrides) -> Trainer:
    """A Trainer carrying the fine-tuning run's material shape."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    defaults: Dict[str, Any] = dict(
        model=model,
        optimizer=optimizer,
        metrics={},
        device="cpu",
        log_every_n_steps=10_000,
        val_every_n_steps=10_000,
        amp_dtype="off",
        grad_clip=None,
        patience=0,
    )
    defaults.update(overrides)
    return Trainer(**defaults)


def _patch_unpack(
    monkeypatch: pytest.MonkeyPatch, sink: List[Tuple[str, List[int]]] | None = None
) -> None:
    """Bypass the CFLP unpacker and optionally record each batch as (split, indices)."""

    def _unpack(batch, **_):
        if sink is not None:
            indices = [int(value) for value in batch[0][:, 0].tolist()]
            if indices and indices[0] >= VAL_INDEX_OFFSET:
                sink.append(("val", [index - VAL_INDEX_OFFSET for index in indices]))
            else:
                sink.append(("train", indices))
        return batch

    monkeypatch.setattr(trainer_module, "unpack_batch", _unpack)
    monkeypatch.setattr(trainer_module, "infer_T_from_labels", lambda labels: 1)


def _install_pinned_rebuild_lifecycle(monkeypatch: pytest.MonkeyPatch, trainer: Trainer) -> None:
    """Restore the pinned runtime's semantics: rebuild both loaders on every use.

    The previous loader is shut down before the next is built.  The pinned runtime did
    *not* do that -- that is the worker leak the repair closes -- but keeping this control
    bounded does not alter the global RNG stream, which is what the order comparison
    below measures.
    """
    state: Dict[str, Any] = {"train": None, "val": None}

    def _rebuild(key: str, factory):
        previous = state[key]
        if previous is not None:
            Trainer._shutdown_loader(previous)
        loader = factory()
        state[key] = loader
        return loader

    monkeypatch.setattr(
        trainer, "_get_train_loader", lambda dm: _rebuild("train", dm.train_dataloader)
    )
    monkeypatch.setattr(
        trainer, "_get_val_loader", lambda dm: _rebuild("val", dm.val_dataloader)
    )


def _train_order_by_epoch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pinned_lifecycle: bool,
    num_workers: int = FINETUNE_NUM_WORKERS,
    max_epochs: int = 4,
    seed: int = FINETUNE_SEED,
) -> List[List[int]]:
    """Run a bounded fine-tuning-shaped `fit()` and return the train record order per epoch."""
    with monkeypatch.context() as patch:
        seen: List[Tuple[str, List[int]]] = []
        _patch_unpack(patch, seen)
        torch.manual_seed(0)
        model = DummyModel()
        trainer = _make_finetune_trainer(model, max_epochs=max_epochs)
        if pinned_lifecycle:
            _install_pinned_rebuild_lifecycle(patch, trainer)
        dm = FinetuneDataModule(num_workers=num_workers)

        torch.manual_seed(seed)
        trainer.fit(dm, problem=None, loss_fn=_loss_fn)

    train_batches = [indices for split, indices in seen if split == "train"]
    batches_per_epoch = -(-FINETUNE_TRAIN_RECORDS // FINETUNE_BATCH_SIZE)
    assert len(train_batches) == batches_per_epoch * max_epochs, (
        f"expected {batches_per_epoch * max_epochs} recorded train batches, "
        f"got {len(train_batches)}"
    )
    return [
        [
            index
            for batch in train_batches[e * batches_per_epoch : (e + 1) * batches_per_epoch]
            for index in batch
        ]
        for e in range(max_epochs)
    ]


# --------------------------------------------------------------------------
# 1) loader lifecycle: built once, torn down explicitly
# --------------------------------------------------------------------------

def test_finetune_loader_lifecycle_active(monkeypatch, tmp_path: Path):
    """The loader is built once per fit() and torn down explicitly; nothing is
    written and no parameter is frozen."""
    _patch_unpack(monkeypatch)
    torch.manual_seed(0)
    model = DummyModel()
    trainer = _make_finetune_trainer(model, max_epochs=5)
    dm = FinetuneDataModule(num_workers=0)

    trainer.fit(dm, problem=None, loss_fn=_loss_fn)

    assert list(tmp_path.iterdir()) == []
    assert all(param.requires_grad for param in model.parameters())

    # Active half: exactly one loader per role, and both released.
    assert dm.train_calls == 1
    assert dm.val_calls == 1
    assert trainer._train_loader is None
    assert trainer._val_loader is None


# --------------------------------------------------------------------------
# 2) mechanism: one iterator per fit() under persistent workers
# --------------------------------------------------------------------------

def test_finetune_reuse_builds_one_dataloader_iterator_for_the_whole_fit(monkeypatch):
    """Motivation: pin the exact mechanism behind the epoch-2+ RNG consequence.  Each
    `_BaseDataLoaderIter` construction draws a `_base_seed` from the global generator;
    the reused iterator is `_reset()` instead of reconstructed, so the current runtime
    performs that draw once per `fit()` instead of once per epoch."""
    max_epochs = 4
    constructions: Dict[str, int] = {"count": 0}
    original_init = _BaseDataLoaderIter.__init__

    def _counting_init(self, loader, *args, **kwargs):
        constructions["count"] += 1
        return original_init(self, loader, *args, **kwargs)

    monkeypatch.setattr(_BaseDataLoaderIter, "__init__", _counting_init)
    _patch_unpack(monkeypatch)
    torch.manual_seed(0)
    model = DummyModel()
    trainer = _make_finetune_trainer(model, max_epochs=max_epochs)
    dm = FinetuneDataModule(num_workers=FINETUNE_NUM_WORKERS)

    torch.manual_seed(FINETUNE_SEED)
    trainer.fit(dm, problem=None, loss_fn=_loss_fn)

    # One train iterator + one val iterator for the entire fit(), not one pair
    # per epoch as the pinned runtime produced.
    assert constructions["count"] == 2, (
        f"expected 2 DataLoader iterator constructions across {max_epochs} epochs, "
        f"got {constructions['count']}"
    )


# --------------------------------------------------------------------------
# 3) the real semantic consequence: epoch-1 identical, epoch-2+ divergent
# --------------------------------------------------------------------------

def test_finetune_shuffled_train_order_matches_pinned_runtime_only_in_the_first_epoch(
    monkeypatch,
):
    """Motivation: state the loader-reuse consequence honestly and measurably instead of
    asserting equivalence in prose.  With fine-tuning's `num_workers=4`, the current runtime
    reproduces the pinned first-epoch shuffle exactly and then diverges from epoch 2
    onward, because it no longer draws a per-epoch worker `_base_seed`."""
    max_epochs = 4
    current = _train_order_by_epoch(monkeypatch, pinned_lifecycle=False, max_epochs=max_epochs)
    pinned = _train_order_by_epoch(monkeypatch, pinned_lifecycle=True, max_epochs=max_epochs)

    # Every epoch is still a full permutation of the frozen 100-record split under
    # both lifecycles -- reuse changes order, never membership or coverage.
    for orders in (current, pinned):
        for order in orders:
            assert sorted(order) == list(range(FINETUNE_TRAIN_RECORDS))

    assert current[0] == pinned[0], "first-epoch shuffle must be unchanged"
    for epoch in range(1, max_epochs):
        assert current[epoch] != pinned[epoch], (
            f"epoch {epoch} order was expected to diverge from the pinned runtime"
        )


def test_finetune_evaluation_lane_num_workers_zero_is_unaffected_by_loader_reuse(monkeypatch):
    """Motivation: bound the blast radius.  The fine-tuning evaluation cells run with
    `num_workers=0`, where `persistent_workers` is False and torch rebuilds the
    single-process iterator on every `__iter__` under both lifecycles.  The change must
    therefore be provably inert there, epoch for epoch."""
    max_epochs = 4
    current = _train_order_by_epoch(
        monkeypatch, pinned_lifecycle=False, num_workers=0, max_epochs=max_epochs
    )
    pinned = _train_order_by_epoch(
        monkeypatch, pinned_lifecycle=True, num_workers=0, max_epochs=max_epochs
    )

    assert current == pinned


# --------------------------------------------------------------------------
# 4) the new order is deterministic and reproducible
# --------------------------------------------------------------------------

def test_finetune_reused_lifecycle_reproduces_the_same_order_for_a_fixed_seed(monkeypatch):
    """Motivation: a changed shuffle is acceptable for fine-tuning only if it is still exactly
    reproducible.  Two independent fits at the same seed must produce identical train
    order in every epoch, including the epochs that diverge from the pinned runtime."""
    max_epochs = 4
    first = _train_order_by_epoch(monkeypatch, pinned_lifecycle=False, max_epochs=max_epochs)
    second = _train_order_by_epoch(monkeypatch, pinned_lifecycle=False, max_epochs=max_epochs)

    assert first == second

    # A different training RNG must still produce a different trajectory, so the
    # equality above is reproducibility rather than a seed-insensitive constant.
    other = _train_order_by_epoch(
        monkeypatch, pinned_lifecycle=False, max_epochs=max_epochs, seed=1
    )
    assert other[0] != first[0]


# --------------------------------------------------------------------------
# 5) the validation stream, which selects the checkpoint, is order-stable
# --------------------------------------------------------------------------

def test_finetune_validation_stream_order_is_identical_under_reuse_and_rebuild(monkeypatch):
    """Motivation: the fine-tuned model's reported checkpoint is chosen by validation loss, so the
    validation stream must not move.  `val_dataloader()` is unshuffled, so reusing its
    persistent iterator must yield the identical record order on every call and match
    the pinned per-call rebuild exactly."""
    max_epochs = 3

    def _val_orders(pinned_lifecycle: bool) -> Tuple[List[List[int]], List[float]]:
        with monkeypatch.context() as patch:
            seen: List[Tuple[str, List[int]]] = []
            _patch_unpack(patch, seen)
            torch.manual_seed(0)
            model = DummyModel()
            trainer = _make_finetune_trainer(model, max_epochs=max_epochs)
            if pinned_lifecycle:
                _install_pinned_rebuild_lifecycle(patch, trainer)
            dm = FinetuneDataModule(num_workers=FINETUNE_NUM_WORKERS)

            orders: List[List[int]] = []
            losses: List[float] = []
            original_validate = trainer.validate

            def _tracking_validate(dm_arg, problem, loss_fn=None):
                mark = len(seen)
                result = original_validate(dm_arg, problem, loss_fn)
                orders.append(
                    [
                        index
                        for split, indices in seen[mark:]
                        if split == "val"
                        for index in indices
                    ]
                )
                losses.append(float(result))
                return result

            patch.setattr(trainer, "validate", _tracking_validate)
            torch.manual_seed(FINETUNE_SEED)
            trainer.fit(dm, problem=None, loss_fn=_loss_fn)
        return orders, losses

    reused_orders, reused_losses = _val_orders(pinned_lifecycle=False)
    rebuilt_orders, rebuilt_losses = _val_orders(pinned_lifecycle=True)

    assert len(reused_orders) == len(rebuilt_orders) > 0
    expected = list(range(FINETUNE_VAL_RECORDS))
    for order in reused_orders:
        assert order == expected, "reused validation iterator changed record order"
    assert reused_orders == rebuilt_orders
    # The first validation call happens before any weight update diverges, so its loss
    # is directly comparable across lifecycles.
    assert reused_losses[0] == pytest.approx(rebuilt_losses[0])


# --------------------------------------------------------------------------
# 6) early stopping / checkpoint selection are untouched by the lifecycle change
# --------------------------------------------------------------------------

def test_finetune_early_stopping_still_fires_and_still_tears_down_the_loaders(monkeypatch):
    """Motivation: fine-tuning runs `patience=10` over up to 200 epochs, so early stopping is
    the expected exit path.  Breaking out of the epoch loop must still reach the
    `finally` teardown -- an early stop must not leak the worker pool -- and the stop
    epoch must not depend on the loader lifecycle."""
    losses = [1.0, 0.9, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    def _run(pinned_lifecycle: bool) -> Tuple[int, FinetuneDataModule, Trainer]:
        with monkeypatch.context() as patch:
            _patch_unpack(patch)
            torch.manual_seed(0)
            model = DummyModel()
            trainer = _make_finetune_trainer(model, max_epochs=len(losses), patience=2)
            if pinned_lifecycle:
                _install_pinned_rebuild_lifecycle(patch, trainer)
            dm = FinetuneDataModule(num_workers=FINETUNE_NUM_WORKERS)

            calls = {"n": 0}

            def _scripted_validate(dm_arg, problem, loss_fn=None):
                value = losses[min(calls["n"], len(losses) - 1)]
                calls["n"] += 1
                return value

            patch.setattr(trainer, "validate", _scripted_validate)
            patch.setattr(trainer, "_save_checkpoint", lambda *a, **k: None)
            torch.manual_seed(FINETUNE_SEED)
            trainer.fit(dm, problem=None, loss_fn=_loss_fn)
            # One scripted validation per epoch, plus the global_step==0 mid-epoch call.
            return calls["n"], dm, trainer

    reused_calls, reused_dm, reused_trainer = _run(pinned_lifecycle=False)
    pinned_calls, _, _ = _run(pinned_lifecycle=True)

    assert reused_calls < len(losses) + 1, "early stopping did not fire"
    assert reused_calls == pinned_calls, "loader lifecycle changed the early-stop epoch"

    # Teardown reached even though fit() exited through `break`.
    assert reused_trainer._train_loader is None
    assert reused_trainer._val_loader is None
    assert reused_dm.train_calls == 1
    # `validate` is scripted here, so the val loader is never requested; the point of
    # this test is the train pool's teardown on the `break` path.
    assert reused_dm.val_calls == 0


# --------------------------------------------------------------------------
# 7) fine-tuning-shaped 4-worker resource behaviour: flat during fit, fully reaped after
# --------------------------------------------------------------------------

def test_finetune_four_worker_pools_stay_flat_and_are_fully_reaped_after_fit(monkeypatch):
    """Motivation: fine-tuning runs up to 200 epochs with `num_workers=4` -- a shape
    that crashes at ~epoch 34 under the pinned per-epoch rebuild.  Prove with
    process-level evidence at that worker count that FDs/threads/children stay flat
    across epochs, and that the explicit teardown returns the child count to baseline
    rather than leaving pools for the GC."""
    if not hasattr(psutil.Process(), "num_fds"):
        pytest.skip("psutil.Process.num_fds is only available on POSIX platforms")

    _patch_unpack(monkeypatch)
    process = psutil.Process()
    baseline_children = len(process.children(recursive=True))

    class ResourceSampler:
        def __init__(self) -> None:
            self.samples: List[Dict[str, int]] = []

        def on_validation_end(self, payload) -> None:
            self.samples.append(
                {
                    "fds": process.num_fds(),
                    "threads": process.num_threads(),
                    "children": len(process.children(recursive=True)),
                }
            )

    sampler = ResourceSampler()
    max_epochs = 12
    torch.manual_seed(0)
    model = DummyModel()
    trainer = _make_finetune_trainer(model, max_epochs=max_epochs, callbacks=[sampler])
    dm = FinetuneDataModule(num_workers=FINETUNE_NUM_WORKERS)

    torch.manual_seed(FINETUNE_SEED)
    trainer.fit(dm, problem=None, loss_fn=_loss_fn)

    assert len(sampler.samples) == max_epochs + 1
    warm = sampler.samples[3:]
    for key in ("fds", "threads", "children"):
        values = [sample[key] for sample in warm]
        assert max(values) - min(values) <= 2, f"{key} grew across epochs (not flat): {values}"

    # Exactly one train pool + one val pool live during the run ...
    assert warm[-1]["children"] - baseline_children == 2 * FINETUNE_NUM_WORKERS
    # ... and both are gone once fit() returns, without waiting for the cyclic GC.
    assert len(process.children(recursive=True)) == baseline_children
    assert trainer._train_loader is None
    assert trainer._val_loader is None


# --------------------------------------------------------------------------
# 8) teardown is safe on the worker-free lane
# --------------------------------------------------------------------------

def test_finetune_teardown_is_a_safe_no_op_without_workers(monkeypatch):
    """Motivation: the evaluation lane and any `num_workers=0` rerun produce
    single-process iterators that have no `_shutdown_workers` hook.  Teardown must
    tolerate that instead of raising at the end of every fit()."""
    _patch_unpack(monkeypatch)
    torch.manual_seed(0)
    model = DummyModel()
    trainer = _make_finetune_trainer(model, max_epochs=2)
    dm = FinetuneDataModule(num_workers=0)

    trainer.fit(dm, problem=None, loss_fn=_loss_fn)

    assert trainer._train_loader is None
    assert trainer._val_loader is None
    # A second teardown on already-released state must also be inert.
    trainer._teardown_dataloaders()
    assert trainer._train_loader is None
