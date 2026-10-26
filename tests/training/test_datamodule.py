import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.types import Instance, Labels
from src.training.datamodule import DataModule
from torch.utils.data import TensorDataset, SequentialSampler


def _build_dm() -> DataModule:
    return DataModule.__new__(DataModule)


class _IdentityProblem:
    @staticmethod
    def collate_fn(batch):
        return batch


def _write_role_records(
    tmp_path: Path,
    *,
    roles: list[str],
) -> tuple[Path, str, str]:
    problem_name = "role_fixture"
    file_name = "labels.pkl"
    data_dir = tmp_path / problem_name
    data_dir.mkdir()
    records = []
    for index, role in enumerate(roles):
        records.append(
            {
                "instance": Instance({"features": np.array([float(index)])}),
                "label": SimpleNamespace(
                    forward_trace={"step": index} if role == "train" else None,
                    x=np.array([index]) if role == "eval" else None,
                    obj=float(index),
                ),
            }
        )
    with (data_dir / file_name).open("wb") as handle:
        pickle.dump(records, handle)
    return tmp_path, problem_name, file_name


def _load_role_fixture(
    tmp_path: Path,
    *,
    roles: list[str],
    role_contract: str = "train_val_test",
) -> DataModule:
    data_path, problem_name, file_name = _write_role_records(tmp_path, roles=roles)
    return DataModule(
        problem=_IdentityProblem(),
        data_path=str(data_path),
        problem_name=problem_name,
        file_name=file_name,
        batch_size=2,
        num_workers=0,
        val_split=0.2,
        test_split=0.2,
        role_contract=role_contract,
    )


def test_eval_only_role_contract_validates_only_held_out_test(tmp_path: Path) -> None:
    # Motivation: a BigMILP file can label all records while evaluation consumes
    # only the held-out tail; non-consumed train/val positions must not be opened.
    dm = _load_role_fixture(
        tmp_path,
        roles=["eval"] * 5,
        role_contract="eval_only",
    )

    assert len(dm.train_ds) == 3
    assert len(dm.val_ds) == 1
    assert len(dm.test_ds) == 1
    assert sum(len(batch) for batch in dm.test_dataloader()) == 1
    with pytest.raises(ValueError, match="no training split"):
        dm.train_dataloader()
    with pytest.raises(ValueError, match="no validation split"):
        dm.val_dataloader()


@pytest.mark.parametrize("bad_role", ["train", "val"])
def test_default_role_contract_rejects_eval_only_record_in_train_or_val(
    tmp_path: Path,
    bad_role: str,
) -> None:
    # Motivation: the eval-only exception must never weaken the default
    # fail-closed invariant that every train and validation label has a trace.
    roles = ["train"] * 5
    roles[0 if bad_role == "train" else 3] = "eval"

    with pytest.raises(ValueError, match=rf"{bad_role} split have no forward trace"):
        _load_role_fixture(tmp_path, roles=roles)


def test_normalize_instances_scales_float_and_preserves_int_types() -> None:
    # Motivation: ensure float features are normalized per-instance and int-like keys stay long.
    instances = [
        Instance(
            {
                "features": np.array([0.0, 2.0], dtype=np.float32),
                "adjacency": np.array([[0, 1], [1, 0]], dtype=np.int64),
                "n_nodes": np.array([2], dtype=np.int64),
            }
        ),
        Instance(
            {
                "features": np.array([1.0, 3.0], dtype=np.float32),
                "adjacency": np.array([[1, 0], [0, 1]], dtype=np.int64),
                "n_nodes": np.array([2], dtype=np.int64),
            }
        ),
    ]

    dm = _build_dm()
    normalized = dm._normalize_instances(instances)

    assert len(normalized) == 2
    assert normalized[0].data["features"].dtype == torch.float32
    assert normalized[0].data["adjacency"].dtype == torch.int64
    assert normalized[0].data["n_nodes"].dtype == torch.int64
    assert normalized[0].data["features"].shape == (2,)
    assert normalized[0].data["adjacency"].shape == (2, 2)

    expected = torch.tensor([0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(normalized[0].data["features"], expected, atol=1e-6)
    assert torch.allclose(normalized[1].data["features"], expected, atol=1e-6)
    assert torch.equal(
        normalized[0].data["adjacency"],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
    )


def test_normalize_instances_constant_float_becomes_zero() -> None:
    # Motivation: avoid NaNs when min == max by checking output is stable.
    instances = [
        Instance({"features": np.array([5.0, 5.0], dtype=np.float32)}),
        Instance({"features": np.array([5.0, 5.0], dtype=np.float32)}),
    ]

    dm = _build_dm()
    normalized = dm._normalize_instances(instances)

    assert torch.allclose(
        normalized[0].data["features"],
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        atol=1e-6,
    )


def test_normalize_instances_empty_returns_same_list() -> None:
    # Motivation: empty input should be a no-op to avoid index errors.
    instances = []
    dm = _build_dm()
    normalized = dm._normalize_instances(instances)

    assert normalized == instances


def test_normalize_instances_dict_payload_preserves_labels() -> None:
    # Motivation: normalization should work for {"instance": Instance, "label": Labels} inputs.
    items = [
        {
            "instance": Instance(
                {
                    "det": np.array([0.0, 2.0], dtype=np.float32),
                    "sce": np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
                }
            ),
            "label": Labels(),
        },
        {
            "instance": Instance(
                {
                    "det": np.array([1.0, 3.0], dtype=np.float32),
                    "sce": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
                }
            ),
            "label": Labels(),
        },
    ]

    dm = _build_dm()
    normalized = dm._normalize_instances(items)

    assert isinstance(normalized[0]["instance"], Instance)
    assert isinstance(normalized[0]["label"], Labels)
    expected_det_0 = torch.tensor([0.0, 2.0 / 7.0], dtype=torch.float32)
    expected_det_1 = torch.tensor([0.0, 2.0 / 7.0], dtype=torch.float32)
    expected_sce_0 = torch.tensor([[1.0 / 7.0, 3.0 / 7.0], [5.0 / 7.0, 1.0]], dtype=torch.float32)
    expected_sce_1 = torch.tensor([[1.0 / 7.0, 3.0 / 7.0], [5.0 / 7.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(normalized[0]["instance"].data["det"], expected_det_0, atol=1e-6)
    assert torch.allclose(normalized[0]["instance"].data["sce"], expected_sce_0, atol=1e-6)
    assert torch.allclose(normalized[1]["instance"].data["det"], expected_det_1, atol=1e-6)
    assert torch.allclose(normalized[1]["instance"].data["sce"], expected_sce_1, atol=1e-6)


def test_resolve_split_counts_absolute_mode_uses_exact_counts() -> None:
    # Motivation: absolute split mode should honor explicit val/test counts.
    n_train, n_val, n_test = DataModule._resolve_split_counts(
        n=450,
        val_split=0.2,
        test_split=0.1,
        split_mode="absolute",
        val_split_number=50,
        test_abs_split_number=0,
    )

    assert (n_train, n_val, n_test) == (400, 50, 0)


def test_resolve_split_counts_absolute_mode_rejects_overflow() -> None:
    # Motivation: protect against invalid splits that would eliminate training data.
    with pytest.raises(ValueError):
        DataModule._resolve_split_counts(
            n=100,
            val_split=0.2,
            test_split=0.1,
            split_mode="absolute",
            val_split_number=60,
            test_abs_split_number=50,
        )


def test_resolve_split_counts_ratio_mode_uses_ratios() -> None:
    # Motivation: ratio split mode should honor fractional splits.
    n_train, n_val, n_test = DataModule._resolve_split_counts(
        n=10,
        val_split=0.2,
        test_split=0.1,
        split_mode="ratio",
        val_split_number=0,
        test_abs_split_number=0,
    )

    assert (n_train, n_val, n_test) == (7, 2, 1)


def test_resolve_split_counts_ratio_mode_applies_min_one() -> None:
    # Motivation: preserve existing behavior where zero ratios still allocate 1.
    n_train, n_val, n_test = DataModule._resolve_split_counts(
        n=5,
        val_split=0.0,
        test_split=0.0,
        split_mode="ratio",
        val_split_number=0,
        test_abs_split_number=0,
    )

    assert (n_train, n_val, n_test) == (3, 1, 1)


def test_train_dataloader_uses_custom_sampler() -> None:
    # Motivation: DDP needs a sampler without shuffle to avoid data duplication.
    dm = _build_dm()
    dm.batch_size = 2
    dm.num_workers = 0

    class DummyProblem:
        @staticmethod
        def collate_fn(batch):
            return batch

    dm.problem = DummyProblem()
    dm.train_ds = TensorDataset(torch.arange(4))
    sampler = SequentialSampler(dm.train_ds)
    dm.set_train_sampler(sampler)

    loader = dm.train_dataloader()
    assert loader.sampler is sampler
