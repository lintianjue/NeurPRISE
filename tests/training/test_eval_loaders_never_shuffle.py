"""Only training may shuffle; every other pass must not.

WHAT THIS GUARDS.  `scripts/04_05_eval_ml_2ro_e2e.py` saves the model's per-instance
scenario selections as `<split>_idx.npy` and saves NO instance ids beside them, so the
array is meaningful only if the loader that produced it walked the dataset in order.
An evaluation on the training split that reuses `train_dataloader()`, which shuffles,
yields an artefact of exactly the right shape, every index in range, nothing raised, and
an overlap with the teacher's selections at chance (8.7% measured; 4 of 50 at random is
8.0%).  Read at face value it says the model is random on its own training data.

WHY IT IS STRUCTURAL.  A grep for "shuffle=False" passes on a comment and fails on a
rename.  This walks the AST: every method whose name ends in `dataloader` must construct
its DataLoader with an explicit `shuffle` keyword, and that keyword must be False for
every method except `train_dataloader`.  `train_dataloader` is allowed both branches --
it has a sampler path that must pass shuffle=False, and a plain path that must shuffle --
so it is asserted to contain at least one True rather than exempted entirely.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

DM = pathlib.Path(__file__).resolve().parents[2] / "src" / "training" / "datamodule.py"
MAY_SHUFFLE = {"train_dataloader"}


def _loader_shuffle_args(fn: ast.FunctionDef) -> list[bool | None]:
    """Every `DataLoader(...)` built inside `fn`, as its literal `shuffle=` value."""
    out: list[bool | None] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != "DataLoader":
            continue
        kw = next((k for k in node.keywords if k.arg == "shuffle"), None)
        if kw is None:
            out.append(None)                      # not passed -> defaults False, but be loud
        elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
            out.append(kw.value.value)
        else:
            out.append(None)                      # computed -> not statically checkable
    return out


def _dataloader_methods() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(DM.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and any(
                   isinstance(b, ast.FunctionDef) and b.name == "train_dataloader"
                   for b in n.body))
    return {b.name: b for b in cls.body
            if isinstance(b, ast.FunctionDef) and b.name.endswith("dataloader")}


def test_every_eval_loader_is_sequential():
    methods = _dataloader_methods()
    assert methods, "no *_dataloader methods found -- the AST walk is looking in the wrong place"
    for name, fn in sorted(methods.items()):
        if name in MAY_SHUFFLE:
            continue
        vals = _loader_shuffle_args(fn)
        assert vals, f"{name} builds no DataLoader; this test can no longer see what it does"
        assert all(v is False for v in vals), (
            f"{name} builds a DataLoader with shuffle={vals}. Evaluation reads artefacts "
            f"back by position and saves no instance ids, so anything but a sequential "
            f"walk silently misaligns them.")


def test_train_dataloader_still_shuffles():
    """The converse, so this file cannot be satisfied by turning shuffling off everywhere."""
    vals = _loader_shuffle_args(_dataloader_methods()["train_dataloader"])
    assert True in vals, (
        "train_dataloader no longer shuffles anywhere. Training on a fixed instance order "
        "is a different experiment, not a fix for the alignment problem.")


def test_train_eval_dataloader_exists():
    """The supported way to read the training split for evaluation."""
    assert "train_eval_dataloader" in _dataloader_methods()


def test_eval_script_does_not_borrow_the_training_loader():
    """The call site, not just the loader: the defect was in the eval script."""
    src = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "04_05_eval_ml_2ro_e2e.py").read_text()
    tree = ast.parse(src)
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "train_dataloader" not in calls, (
        "the evaluation script calls train_dataloader(), which shuffles. Use "
        "train_eval_dataloader() -- see its docstring for what that silently costs.")
    assert "train_eval_dataloader" in calls
