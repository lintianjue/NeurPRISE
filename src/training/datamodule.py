from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, Sampler
from src.core.types import Instance
from src.core.problem import IProblem
from src.utils.label_roles import assert_split_roles
import pickle
import logging
from pathlib import Path
from typing import Callable, Any
import sys


ROLE_CONTRACT_TRAIN_VAL_TEST = "train_val_test"
ROLE_CONTRACT_EVAL_ONLY = "eval_only"
_ROLE_CONTRACTS = {ROLE_CONTRACT_TRAIN_VAL_TEST, ROLE_CONTRACT_EVAL_ONLY}


def _register_numpy_pickle_shims() -> None:
    """Provide shims for numpy internal module paths used by older pickles."""
    try:
        import numpy._core.numeric  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.numeric as _np_numeric
        sys.modules["numpy._core.numeric"] = _np_numeric
    try:
        import numpy._core.multiarray  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.multiarray as _np_multiarray
        sys.modules["numpy._core.multiarray"] = _np_multiarray

class InstanceDataset(Dataset):
    """Generic dataset that wraps a list of Instance objects.

    If `transform` is provided, it is applied ONCE in __init__ to all instances
    (eager preprocessing), not per-sample in __getitem__.
    """

    def __init__(self, instances: List[Any], transform: Optional[Callable[[Any], Any]] = None):
        if transform is not None:
            # Eagerly transform everything once
            self.instances = [transform(inst) for inst in instances]
            self.transform = None
        else:
            self.instances = instances
            self.transform = None

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class DataModule:
    """Generic DataModule that works with any problem implementing IProblem.
    """

    def __init__(
        self,
        problem: IProblem,
        data_path: str,
        problem_name: str,
        file_name: str,
        batch_size: int,
        num_workers: int,
        val_split: float = 0.2,
        test_split: float = 0.0,
        split_mode: str = "ratio",
        val_split_number: int = 0,
        test_abs_split_number: int = 0,
        train_split_number: int = 0,
        normalize: bool = False,
        normalize_mode: str = "minmax",
        role_contract: str = ROLE_CONTRACT_TRAIN_VAL_TEST,
    ):

        self.problem = problem
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.train_sampler: Sampler | None = None
        if role_contract not in _ROLE_CONTRACTS:
            raise ValueError(
                f"Unknown role contract '{role_contract}'. Expected one of "
                f"{sorted(_ROLE_CONTRACTS)}."
            )
        self.role_contract = role_contract

        # Load pickle file
        data_file = f"{data_path}/{problem_name}/{file_name}"
        self.data_file = str(Path(data_file).resolve())
        logging.info("Loading data file: %s", data_file)
        with open(data_file, "rb") as f:
            _register_numpy_pickle_shims()
            instance_list = pickle.load(f)

        logging.info(f"Loaded {len(instance_list)} instances from {data_file}")

        # Optional: Convert to tensors and normalize
        if normalize:
            instance_list = self._normalize_instances(instance_list)

        self.full_ds = InstanceDataset(instance_list)

        # --- Simple hold-out split: train / val / (optional) test ---
        n = len(self.full_ds)
        n_train, n_val, n_test = self._resolve_split_counts(
            n=n,
            val_split=val_split,
            test_split=test_split,
            split_mode=split_mode,
            val_split_number=val_split_number,
            test_abs_split_number=test_abs_split_number,
            train_split_number=train_split_number,
        )

        # Test set anchored at the end so it stays fixed across data budgets.
        # When n_train + n_val + n_test < n, instances in the gap are unused.
        test_start = n - n_test

        # Validate label roles before splitting: exact-only records cannot train the model.
        test_indices = (
            range(test_start, test_start + n_test) if n_test > 0 else range(0, 0)
        )
        if role_contract == ROLE_CONTRACT_EVAL_ONLY:
            # Keep the frozen test tail; exact-only labels cannot feed training loaders.
            assert_split_roles(
                instance_list,
                range(0, 0),
                range(0, 0),
                test_indices,
                context=data_file,
            )
        else:
            assert_split_roles(
                instance_list,
                range(0, n_train),
                range(n_train, n_train + n_val),
                test_indices,
                context=data_file,
            )

        splits = [
            Subset(self.full_ds, range(0, n_train)),
            Subset(self.full_ds, range(n_train, n_train + n_val)),
            Subset(self.full_ds, range(test_start, test_start + n_test)),
        ]
        self.train_ds = splits[0]
        self.val_ds = splits[1]
        self.test_ds = splits[2] if n_test > 0 else None

        if self.test_ds is not None:
            self.val_test_merged = ConcatDataset([self.val_ds, self.test_ds])
            logging.info(
                f"Loaded {len(self.train_ds)} training, {len(self.val_ds)} "
                f"validation, and {len(self.test_ds)} test instances."
            )
        else:
            # If test doesn't exist, merged set is just val
            self.val_test_merged = self.val_ds
            logging.info(
                f"Loaded {len(self.train_ds)} training and {len(self.val_ds)} "
                f"validation instances (no test set)."
            )

    @staticmethod
    def _resolve_split_counts(
        *,
        n: int,
        val_split: float,
        test_split: float,
        split_mode: str,
        val_split_number: int,
        test_abs_split_number: int,
        train_split_number: int = 0,
    ) -> tuple[int, int, int]:
        if split_mode == "ratio":
            n_test = max(1, int(n * test_split))
            n_val = max(1, int(n * val_split))
        elif split_mode == "absolute":
            if val_split_number < 0 or test_abs_split_number < 0:
                raise ValueError(
                    "Absolute split numbers must be >= 0. "
                    f"Got val_split_number={val_split_number}, "
                    f"test_abs_split_number={test_abs_split_number}."
                )
            n_val = int(val_split_number)
            n_test = int(test_abs_split_number)
        else:
            raise ValueError(
                "split_mode must be 'ratio' or 'absolute'. "
                f"Got split_mode={split_mode}."
            )

        if train_split_number > 0:
            n_train = int(train_split_number)
            if n_train + n_val + n_test > n:
                raise ValueError(
                    f"Explicit splits exceed dataset size: "
                    f"train={n_train} + val={n_val} + test={n_test} = "
                    f"{n_train + n_val + n_test} > {n}."
                )
        else:
            n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                f"Invalid split: train size {n_train} <= 0. "
                f"Check split_mode={split_mode}, val_split={val_split}, "
                f"test_split={test_split}, val_split_number={val_split_number}, "
                f"test_abs_split_number={test_abs_split_number}, "
                f"train_split_number={train_split_number}, n={n}."
            )
        return n_train, n_val, n_test


    def _normalize_instances(self, instances: List[Any]) -> List[Any]:
        """Convert numpy arrays to tensors and optionally normalize."""
        if not instances:
            return instances

        first = instances[0]
        if isinstance(first, Instance):
            return self._normalize_instance_list(instances)
        if isinstance(first, dict) and isinstance(first.get("instance"), Instance):
            normalized_instances = self._normalize_instance_list(
                [item["instance"] for item in instances]
            )
            normalized_items = []
            for item, inst in zip(instances, normalized_instances):
                new_item = dict(item)
                new_item["instance"] = inst
                normalized_items.append(new_item)
            return normalized_items

        raise TypeError(
            "Expected list of Instance or dicts with an Instance under 'instance'. "
            f"Got {type(first)}."
        )

    def _normalize_instance_list(self, instances: List[Instance]) -> List[Instance]:
        normalized_instances = []
        mode = getattr(self, "normalize_mode", "minmax")

        for inst in instances:
            normalized_instances.append(
                self._normalize_instance_shared_pairs(inst, mode=mode)
            )

        return normalized_instances

    @staticmethod
    def _normalize_instance_shared_pairs(
        inst: Instance, *, mode: str = "minmax"
    ) -> Instance:
        """
        Normalize each tensor independently, except when deterministic/scenario
        tensors represent the same quantity (e.g., det & sce). Those are normalized
        together to preserve their relative scale.
        """
        if mode not in {"minmax", "scale"}:
            raise ValueError(f"Unknown normalize mode '{mode}'. Expected 'minmax' or 'scale'.")
        new_data = dict(inst.data)

        # Shared-scale group: det + sce (same cost type across stages)
        if "det" in new_data and "sce" in new_data:
            det_t = torch.as_tensor(new_data["det"]).float()
            sce_t = torch.as_tensor(new_data["sce"]).float()
            combined = torch.cat([det_t.reshape(-1), sce_t.reshape(-1)], dim=0)
            if mode == "minmax":
                t_min = combined.min()
                t_max = combined.max()
                denom = t_max - t_min + 1e-8
                new_data["det"] = (det_t - t_min) / denom
                new_data["sce"] = (sce_t - t_min) / denom
            else:
                denom = combined.abs().max() + 1e-8
                new_data["det"] = det_t / denom
                new_data["sce"] = sce_t / denom

        # Per-tensor normalization for remaining float tensors
        for key, value in new_data.items():
            if key in {"det", "sce"}:
                continue
            tensor = torch.as_tensor(value)
            if tensor.is_floating_point():
                tensor = tensor.float()
                if mode == "minmax":
                    t_min = tensor.min()
                    t_max = tensor.max()
                    tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
                else:
                    denom = tensor.abs().max() + 1e-8
                    tensor = tensor / denom
                new_data[key] = tensor
            else:
                new_data[key] = tensor

        return Instance(new_data)

    def train_dataloader(self) -> DataLoader:
        if getattr(self, "role_contract", ROLE_CONTRACT_TRAIN_VAL_TEST) == ROLE_CONTRACT_EVAL_ONLY:
            raise ValueError("The eval-only role contract has no training split.")
        logging.info(f"Training DataLoader batch_size {self.batch_size}, num_workers {self.num_workers}")
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        if self.train_sampler is not None:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=self.problem.collate_fn,
                pin_memory=True,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def train_eval_dataloader(self) -> DataLoader:
        """Return a DataLoader for evaluating training instances in their original order.

        Disable shuffling so saved selection rows remain aligned with instance rows.
        """
        if getattr(self, "role_contract", ROLE_CONTRACT_TRAIN_VAL_TEST) == ROLE_CONTRACT_EVAL_ONLY:
            raise ValueError("The eval-only role contract has no training split.")
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        if getattr(self, "role_contract", ROLE_CONTRACT_TRAIN_VAL_TEST) == ROLE_CONTRACT_EVAL_ONLY:
            raise ValueError("The eval-only role contract has no validation split.")
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise ValueError("No test dataset loaded. Set test_split>0 or call set_splits().")
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    # to evaluate the heuristic on the full dataset
    def full_dataloader(self) -> DataLoader:
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def val_test_dataloader(self) -> DataLoader:
        prefetch_factor = 4 if self.num_workers > 0 else None
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.val_test_merged,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.problem.collate_fn,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def set_splits(
        self,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: Optional[List[int]] = None,
    ) -> None:
        """Replace train/validation/test subsets with explicit indices.

        Update the DataModule in place; omit ``test_indices`` to disable the test subset.
        """
        self.train_ds = Subset(self.full_ds, train_indices)
        self.val_ds = Subset(self.full_ds, val_indices)
        self.test_ds = Subset(self.full_ds, test_indices) if test_indices is not None else None

    def set_train_sampler(self, sampler: Sampler | None) -> None:
        self.train_sampler = sampler
