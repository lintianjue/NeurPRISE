"""Repo-level fixtures for producer-anchored controls.

A *producer* is the code that writes a path, a file or a field; a *consumer* is the
code that has to know what it was called.  When the two are written independently,
the natural control -- a test that builds the expected name the same way the consumer
builds it -- shares the consumer's bug and passes (a `k_0.20` vs `k_0.2` directory
mismatch goes unnoticed that way).

The fix in kind is to never type the expectation: import the *real* producer and let it
compute it.  These fixtures make that a one-liner for any ladder.  Most repo producers
are `scripts/NN_*.py` entry points that are not importable as `scripts.04_05_...`
(the module name starts with a digit) and are expensive to import (torch,
torch_geometric), so they are loaded by file location and cached for the session.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

import os as _os

# --- run_io launcher variables ------------------------------------------------------
# setdefault, so a caller's own values win: an input root pointed at the repo
# (read-only reads still work) and a fresh, per-session writable run root.
_os.environ.setdefault("NEURPRISE_INPUT_ROOT", str(Path(__file__).resolve().parents[1]))
_os.environ.setdefault("NEURPRISE_RUN_ROOT", tempfile.mkdtemp(prefix="neurprise_pytest_run_root_"))


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "scripts" / "04_05_eval_ml_2ro_e2e.py"


def load_producer(relative_path: str | Path) -> ModuleType:
    """Import a repo script by file location, regardless of its package layout.

    `relative_path` is resolved against the repository root.  Modules are cached under
    a name derived from the path, so a session imports each producer at most once.
    """
    path = (REPO_ROOT / relative_path).resolve()
    module_name = "producer_" + path.stem.replace(".", "_").replace("-", "_")
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        pytest.fail(f"cannot load the producer at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def producer_module() -> Callable[[str | Path], ModuleType]:
    """Factory: `producer_module("scripts/foo.py")` -> the imported module."""
    return load_producer


@pytest.fixture(scope="session")
def producer_resolve_out_dir() -> Callable[[Any], str]:
    """The evaluator's own `_resolve_out_dir`, imported from the real script.

    `scripts/04_05_eval_ml_2ro_e2e.py` (via `src.utils.eval_helpers.resolve_e2e_out_dir`)
    is the authority for every
    `<root>/<name>/e2e_2ro/k_<ratio>` directory in the repo.  Consumers that spell that
    layout themselves must be tested against this, not against a hand-written string.
    """
    return load_producer(EVALUATOR_PATH)._resolve_out_dir


@pytest.fixture(scope="session")
def producer_eval_dir(producer_resolve_out_dir) -> Callable[..., Path]:
    """Drive the real producer with the real Hydra override token for one k ratio.

    `producer_eval_dir(root, name, "0.20")` returns the directory the evaluator would
    create for `eval.output_dir=<root> eval.name=<name> eval.k_ratio=0.20`, parsed by
    Hydra's own override parser rather than by `float()` -- so the test cannot agree with
    the consumer by accident.  The producer returns a path and creates nothing;
    aim `root` at `tmp_path` regardless.
    """
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    def _dir(root: Path | str, name: str, k_ratio: str) -> Path:
        parsed = OverridesParser.create().parse_overrides([f"eval.k_ratio={k_ratio}"])[0].value()
        return Path(producer_resolve_out_dir(OmegaConf.create(
            {"output_dir": str(root), "name": name, "k_ratio": parsed}
        )))

    return _dir
