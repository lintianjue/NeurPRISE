"""The sidecar format has exactly one definition (`src/utils/run_metadata.py`), and
every re-export -- `scripts/pipeline/write_run_sidecar.py`, `scripts/pipeline/verify_ckpt_provenance.py`
-- points at the same objects.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.pipeline.write_run_sidecar as write_run_sidecar
import src.utils.run_metadata as run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_ckpt(tmp_path: Path, name: str = "model_best.pt", nbytes: int = 256) -> Path:
    p = tmp_path / name
    p.write_bytes(os.urandom(nbytes))
    return p


def test_old_and_new_paths_build_identical_fields(tmp_path):
    """Kill: a second `build_fields` defined in write_run_sidecar.py."""
    ckpt = _make_ckpt(tmp_path)
    label = tmp_path / "label.pkl"
    label.write_bytes(b"some label bytes")

    old = write_run_sidecar.build_fields(ckpt, arm="x", label_path=label)
    new = run_metadata.build_fields(ckpt, arm="x", label_path=label)
    old.pop("sidecar_written_at", None)
    new.pop("sidecar_written_at", None)
    assert old == new


def test_field_names_and_null_arm(tmp_path):
    """Kill: renaming any key / wrapping the arm assignment in `if arm is not None`."""
    ckpt = _make_ckpt(tmp_path)
    label = tmp_path / "label.pkl"
    label.write_bytes(b"labelbytes")

    fields = run_metadata.build_fields(ckpt, arm=None, label_path=label)
    ck = fields["checkpoint"]
    assert set(ck.keys()) >= {"best_pt", "best_pt_md5", "best_pt_bytes", "exists", "arm"}
    assert "arm" in ck and ck["arm"] is None
    assert ck["best_pt_md5"] == hashlib.md5(ckpt.read_bytes()).hexdigest()

    lab = fields["label"]
    assert set(lab.keys()) >= {"label_path", "label_path_resolved", "label_md5"}


def test_missing_checkpoint_writes_nothing(tmp_path):
    """Kill: dropping the `is_file()` check (falls through to a FileNotFoundError from
    inside build_fields, caught by the generic except -- ok is still False, but the
    message no longer names "does not exist")."""
    ckpt = tmp_path / "does_not_exist.pt"
    ok, msg = run_metadata.write_and_verify(ckpt)
    assert ok is False
    assert "does not exist" in msg
    assert not run_metadata.sidecar_path(ckpt).exists()


def test_existing_sidecar_is_not_overwritten(tmp_path, monkeypatch):
    """Kill: removing the `side.exists()` refusal. The `os.link` exclusivity in the write
    path is also a safety net for a REGULAR pre-existing sidecar, so the return value
    alone does not discriminate; assert the short-circuit -- `build_fields` (which reads
    and hashes the checkpoint) must never run -- which only the early check guarantees."""
    ckpt = _make_ckpt(tmp_path)
    side = run_metadata.sidecar_path(ckpt)
    side.write_text("PRE-EXISTING", encoding="utf-8")

    def must_not_be_called(*a, **kw):
        raise AssertionError("build_fields ran despite an existing sidecar")

    monkeypatch.setattr(run_metadata, "build_fields", must_not_be_called)

    ok, msg = run_metadata.write_and_verify(ckpt)
    assert ok is False
    assert "overwrite" in msg
    assert side.read_text(encoding="utf-8") == "PRE-EXISTING"

    # dangling symlink case: exists() is False here, only is_symlink() catches it
    ckpt2 = _make_ckpt(tmp_path, name="model2_best.pt")
    side2 = run_metadata.sidecar_path(ckpt2)
    side2.symlink_to(tmp_path / "nowhere.json")
    ok2, msg2 = run_metadata.write_and_verify(ckpt2)
    assert ok2 is False


def test_checkpoint_changed_during_write(tmp_path, monkeypatch):
    """Kill: removing the recompute-after-write digest check."""
    ckpt = _make_ckpt(tmp_path)
    real_md5 = run_metadata.file_md5
    calls = {"n": 0}

    def flaky_md5(path, chunk=1 << 20):
        calls["n"] += 1
        if calls["n"] >= 2:
            return "0" * 32
        return real_md5(path, chunk)

    monkeypatch.setattr(run_metadata, "file_md5", flaky_md5)
    ok, msg = run_metadata.write_and_verify(ckpt)
    assert ok is False
    assert "changed" in msg


def test_internal_exception_is_returned_not_raised(tmp_path, monkeypatch):
    """Kill: removing the outer try/except in write_and_verify."""
    ckpt = _make_ckpt(tmp_path)

    def boom(*a, **kw):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(run_metadata.json, "dump", boom)
    ok, msg = run_metadata.write_and_verify(ckpt)
    assert ok is False
    assert "ValueError" in msg
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")]
    assert leftover == []


def test_cli_exit_codes(tmp_path):
    """Kill: dropping the sys.path insert (ImportError -> rc 1), or `return 0`."""
    ckpt = _make_ckpt(tmp_path)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    r1 = subprocess.run(
        [sys.executable, "scripts/pipeline/write_run_sidecar.py", str(ckpt), "--arm", "a"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    assert run_metadata.sidecar_path(ckpt).exists()

    r2 = subprocess.run(
        [sys.executable, "scripts/pipeline/write_run_sidecar.py", str(ckpt), "--arm", "a"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert r2.returncode == 1


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_path_loaded_modules_expose_the_names():
    """Kill: a local redefinition of these names in either script."""
    wrs = _load_by_path("_t_wrs", "scripts/pipeline/write_run_sidecar.py")
    vcp = _load_by_path("_t_vcp", "scripts/pipeline/verify_ckpt_provenance.py")
    for mod in (wrs, vcp):
        assert mod.SIDECAR_SUFFIX is run_metadata.SIDECAR_SUFFIX
        assert mod.sidecar_path is run_metadata.sidecar_path
        assert mod.file_md5 is run_metadata.file_md5


def test_single_definition():
    """Kill: reintroducing a copy of SIDECAR_SUFFIX or file_md5/_file_md5 anywhere."""
    modules = [
        "scripts/pipeline/write_run_sidecar.py",
        "scripts/pipeline/verify_ckpt_provenance.py",
        "src/utils/run_metadata.py",
    ]
    suffix_assigns = []
    md5_defs = []
    for relpath in modules:
        tree = ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"), filename=relpath)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SIDECAR_SUFFIX":
                        suffix_assigns.append(relpath)
            if isinstance(node, ast.FunctionDef) and node.name in ("file_md5", "_file_md5"):
                md5_defs.append((relpath, node.name))

    assert suffix_assigns == ["src/utils/run_metadata.py"]
    assert md5_defs == [("src/utils/run_metadata.py", "file_md5")]
