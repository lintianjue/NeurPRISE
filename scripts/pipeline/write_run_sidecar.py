#!/usr/bin/env python3
"""Emitter for the `<checkpoint>.run.json` provenance sidecar.

Pairs with `scripts/pipeline/verify_ckpt_provenance.py`, which refuses to evaluate a checkpoint
without one.

WHAT IT RECORDS, AND WHY EACH FIELD IS LOAD-BEARING
---------------------------------------------------
`ckpt_md5`   the digest of the finished checkpoint. Answers *"are these the bytes the
             sidecar describes"* — catches a file copied, truncated or replaced.
`arm`        which experimental arm this checkpoint IS. Answers *"is this the `tl2s`
             checkpoint"*, which `ckpt_md5` cannot; a filename alone is not evidence of
             the arm. A sidecar recording bytes but not identity would let us prove a
             file is intact while still not knowing what it is.
`label_md5`  which supervision produced it — the arm's *content*, not just its name, so a
             mislabelled arm is catchable even if the `arm` string is right.

FAILURE MODE: LOUD AND LATE, BY CONSTRUCTION
--------------------------------------------
This runs inside training, so a bug here costs a **completed run**, not a preflight.
Therefore:

  * it is called **after** the checkpoint exists, never before or during;
  * it never raises into the training process — every path returns a bool;
  * it **verifies its own output by reading it back** and re-comparing the digest, so a
    sidecar that was written but is unreadable or wrong is reported, not assumed;
  * on failure the caller prints loudly and may exit non-zero, but the model is already
    safely on disk by then.
  * An existing sidecar is never overwritten; a second emission for the same checkpoint
    returns False.

A sidecar this emitter cannot verify is worse than none, because the consumer would
accept it.

Usage as a library (what 03_train.py does):
    from scripts.pipeline.write_run_sidecar import write_and_verify
    ok, msg = write_and_verify(ckpt_path, extra={...}, arm=..., label_path=...)

Usage as a CLI (backfill, or an emitter for a path this module does not own):
    python3 scripts/pipeline/write_run_sidecar.py CKPT --arm tl2s --label-path data/....pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.run_metadata import (  # noqa: E402
    SIDECAR_SUFFIX, sidecar_path, file_md5, build_fields, write_and_verify, emit_loudly,
)

__all__ = ["SIDECAR_SUFFIX", "sidecar_path", "file_md5", "build_fields", "write_and_verify", "emit_loudly", "main"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--arm", default=None, help="the experimental arm this checkpoint IS")
    ap.add_argument("--label-path", default=None, help="supervision file used to train it")
    ap.add_argument("--extra-json", default=None,
                    help="path to a JSON object merged into the sidecar (e.g. an existing "
                         "run record)")
    args = ap.parse_args()

    extra = None
    if args.extra_json:
        with open(args.extra_json, "r", encoding="utf-8") as fh:
            extra = json.load(fh)

    ok = emit_loudly(args.ckpt, arm=args.arm, label_path=args.label_path, extra=extra)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
