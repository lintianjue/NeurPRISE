"""Single definition of the `<checkpoint>.run.json` provenance sidecar format.

`SIDECAR_SUFFIX`, `sidecar_path`, `file_md5`, `build_fields`, `write_and_verify` and
`emit_loudly` are defined once here; the writer, verifier and adapter reader all import
them. `scripts/pipeline/write_run_sidecar.py` is its CLI.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_SUFFIX = ".run.json"


def sidecar_path(ckpt: Path) -> Path:
    return Path(str(ckpt) + SIDECAR_SUFFIX)


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_fields(ckpt: Path, *, arm=None, label_path=None, extra=None) -> dict:
    """Build the metadata dict for a checkpoint sidecar.

    Hash this exact file into ``checkpoint.best_pt_md5`` and record its path, size
    and arm; aliases may have different bytes and need their own sidecars.
    Returns the merged checkpoint, optional label and provenance fields.
    """
    fields = dict(extra or {})
    ck = dict(fields.get("checkpoint") or {})
    ck["best_pt"] = str(ckpt)
    ck["best_pt_md5"] = file_md5(ckpt)
    ck["best_pt_bytes"] = ckpt.stat().st_size
    ck["exists"] = True
    # `arm` is recorded even when None. A null arm is an honest "not declared", and the
    # consumer fails closed against it when it expects a specific arm -- the correct
    # outcome, not a gap.
    ck["arm"] = arm
    fields["checkpoint"] = ck

    if label_path is not None:
        lab = dict(fields.get("label") or {})
        lp = Path(label_path)
        resolved = lp.resolve() if lp.exists() else None
        lab["label_path"] = str(label_path)
        lab["label_path_resolved"] = str(resolved) if resolved else None
        lab["label_md5"] = file_md5(resolved) if resolved else None
        fields["label"] = lab

    fields.setdefault("sidecar_written_at", datetime.now(timezone.utc).isoformat())
    fields.setdefault("sidecar_written_by", os.path.basename(sys.argv[0]) or "write_run_sidecar")
    fields.setdefault("host", socket.gethostname())
    return fields


def write_and_verify(ckpt, *, arm=None, label_path=None, extra=None) -> tuple[bool, str]:
    """Write `<ckpt>.run.json`, then read it back and re-verify. Never raises.

    Returns (ok, message). The caller decides how loudly to fail; this function's
    contract is that it does not take a finished training run down with it.
    """
    try:
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            return False, (f"refusing to write a sidecar for a checkpoint that does not "
                           f"exist: {ckpt}. The sidecar must be written AFTER the "
                           f"checkpoint, or its digest describes nothing")

        side = sidecar_path(ckpt)
        if side.exists() or side.is_symlink():
            return False, (f"refusing to overwrite existing sidecar {side}; provenance "
                           f"sidecars are write-once")

        fields = build_fields(ckpt, arm=arm, label_path=label_path, extra=extra)

        # Write via an exclusively-created temp file in the same directory, then publish
        # with a hard link (which fails if the target already exists), so a crash mid-write
        # cannot leave a truncated sidecar and a concurrent writer cannot clobber one.
        tmp = side.with_name(f".{side.name}.{os.getpid()}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(fields, fh, indent=2, sort_keys=True, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, side)
            except FileExistsError:
                return False, (f"refusing to overwrite existing sidecar {side}; "
                               f"provenance sidecars are write-once")
            except OSError as exc:
                return False, f"sidecar emission failed (hard link unsupported: {exc})"
        finally:
            if tmp.exists():
                tmp.unlink()

        # --- verify by reading back, not by trusting the write ---
        with open(side, "r", encoding="utf-8") as fh:
            readback = json.load(fh)
        rb = readback.get("checkpoint") or {}
        want = fields["checkpoint"]["best_pt_md5"]
        if rb.get("best_pt_md5") != want:
            return False, (f"sidecar read-back mismatch: wrote {want}, read "
                           f"{rb.get('best_pt_md5')}")
        recomputed = file_md5(ckpt)
        if recomputed != rb.get("best_pt_md5"):
            return False, (f"checkpoint changed while the sidecar was being written: "
                           f"sidecar {rb.get('best_pt_md5')}, file now {recomputed}")
        if rb.get("arm") != arm:
            return False, f"sidecar read-back arm mismatch: wrote {arm!r}, read {rb.get('arm')!r}"
        if Path(str(rb.get("best_pt"))).name != ckpt.name:
            return False, (f"sidecar read-back subject mismatch: describes "
                           f"{rb.get('best_pt')!r}, sits beside {ckpt.name!r}")

        return True, (f"{side.name} written and verified: checkpoint.best_pt_md5={want} "
                      f"arm={arm!r}")
    except Exception as exc:
        return False, f"sidecar emission failed ({type(exc).__name__}: {exc})"


def emit_loudly(ckpt, *, arm=None, label_path=None, extra=None, logger=None) -> bool:
    """`write_and_verify` plus the loud half. Returns ok; never raises."""
    ok, msg = write_and_verify(ckpt, arm=arm, label_path=label_path, extra=extra)
    line = f"[sidecar] {'OK' if ok else 'FAILED'} {msg}"
    if logger is not None:
        (logger.info if ok else logger.error)(line)
    if not ok:
        print("=" * 78, file=sys.stderr)
        print(f"[sidecar] PROVENANCE SIDECAR NOT WRITTEN: {msg}", file=sys.stderr)
        print("[sidecar] The checkpoint is safe on disk. It will NOT pass "
              "scripts/pipeline/verify_ckpt_provenance.py until a sidecar exists.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
    else:
        print(line, flush=True)
    return ok
