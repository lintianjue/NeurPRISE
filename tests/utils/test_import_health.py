"""The tree must be importable.

A multi-file edit to shared ``src/`` modules that lands non-atomically surfaces
somewhere expensive and unrelated --

* a three-way cycle ``label_utils -> batch_utils -> omega_conf -> label_utils`` makes
  all three unimportable, found only when a regret evaluation dies at import;
* a training launch dies importing a symbol which existed when the run started
  and has been moved by the time it imports.

Neither is detectable by importing inside an already-warm pytest process: by then
``sys.modules`` is populated and a cycle resolves silently. Each check therefore
runs in a **fresh subprocess**.

Keep this fast and dependency-free. It is meant to be the first thing anyone runs
after touching ``src/``.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Modules that participate in the known cycle, plus the leaf the helper was moved to.
CORE_MODULES = [
    "src.utils.batch_utils",
    "src.utils.omega_conf",
    "src.utils.label_utils",
    "src.utils.vertex_cover.vertex_cover_utils",
    "src.utils.label_roles",
    "src.training.datamodule",
]


def _run(args, timeout=300):
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    )


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_imports_in_a_fresh_interpreter(module):
    # Motivation: a cycle is invisible once sys.modules is warm. One process per module.
    proc = _run([sys.executable, "-c", f"import {module}"])
    assert proc.returncode == 0, (
        f"`import {module}` failed in a fresh interpreter. The tree is broken "
        f"until this is fixed.\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_training_entry_point_still_starts():
    # Motivation: an ImportError in 03_train.py's import block kills a training launch.
    # --help exercises the whole import block without running anything.
    proc = _run([sys.executable, "scripts/03_train.py", "--help"])
    assert proc.returncode == 0, (
        "scripts/03_train.py --help failed; the training entry point cannot start.\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
