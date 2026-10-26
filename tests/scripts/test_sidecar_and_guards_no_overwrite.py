"""Write-once provenance: `write_run_sidecar.write_and_verify` may not
overwrite an existing artifact.

A second emission for the same checkpoint (crash-retry, a re-run pointed at the same cell,
two workers racing) must be refused rather than silently clobbering a sidecar something else
may already be trusting. Each test below names the mutation that would turn it red.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.write_run_sidecar import sidecar_path, write_and_verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_ckpt(tmp_path: Path, name: str = "model.pt") -> Path:
    ckpt = tmp_path / name
    ckpt.write_bytes(b"fake checkpoint bytes")
    return ckpt


def test_sidecar_written_once_second_call_refused(tmp_path):
    """Mutation: restore `os.replace(tmp, side)` and drop the pre-write exists check."""
    ckpt = _make_ckpt(tmp_path)
    ok1, msg1 = write_and_verify(ckpt, arm="exact")
    assert ok1, msg1
    side = sidecar_path(ckpt)
    assert side.is_file()
    body = json.loads(side.read_text())
    assert body["checkpoint"]["best_pt_md5"]
    before = _sha256(side)

    ok2, msg2 = write_and_verify(ckpt, arm="exact")
    assert ok2 is False
    assert "refusing to overwrite" in msg2

    assert _sha256(side) == before


def test_no_temp_file_left_behind(tmp_path, monkeypatch):
    """Mutation: remove the `finally` cleanup of the temp file."""
    ckpt = _make_ckpt(tmp_path)
    ok1, msg1 = write_and_verify(ckpt, arm="exact")
    assert ok1, msg1
    ok2, _ = write_and_verify(ckpt, arm="exact")
    assert ok2 is False
    assert not list(tmp_path.glob(".*.tmp"))

    # a write that fails mid-json.dump must not leave a sidecar or a temp file either
    ckpt2 = _make_ckpt(tmp_path, "model2.pt")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(json, "dump", _boom)
    ok3, msg3 = write_and_verify(ckpt2, arm="exact")
    assert ok3 is False
    assert not sidecar_path(ckpt2).exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_dangling_symlink_sidecar_refused(tmp_path):
    """Mutation: drop `or side.is_symlink()`.

    `Path.exists()` follows symlinks, so a symlink whose target DOES exist would already be
    caught by the plain `exists()` check -- that would not exercise the added clause. A
    DANGLING symlink (target absent) makes `exists()` return False while `is_symlink()`
    returns True, so only the new clause catches it.
    """
    ckpt = _make_ckpt(tmp_path)
    side = sidecar_path(ckpt)
    target = tmp_path / "nonexistent_target.txt"
    side.symlink_to(target)
    assert not side.exists()          # dangling: exists() is False
    assert side.is_symlink()

    ok, msg = write_and_verify(ckpt, arm="exact")
    assert ok is False
    assert "refusing to overwrite" in msg

    # nothing was written through the dangling link, and the link itself is untouched
    assert not target.exists()
    assert Path(os.readlink(side)) == target


def test_emit_loudly_returns_false_and_prints_banner_on_second_call(tmp_path, capsys):
    """Mutation: make write_and_verify return True when the sidecar already exists."""
    from scripts.pipeline.write_run_sidecar import emit_loudly

    ckpt = _make_ckpt(tmp_path)
    assert emit_loudly(ckpt, arm="exact") is True
    capsys.readouterr()

    ok = emit_loudly(ckpt, arm="exact")
    captured = capsys.readouterr()
    assert ok is False
    assert "PROVENANCE SIDECAR NOT WRITTEN" in captured.err
    assert "refusing to overwrite" in captured.err
