#!/usr/bin/env python3
"""Merge a forward-trace label file and a BigMILP label file into one dataset.

Produces the single 500-record file that lets the DataModule's index-anchored
80/10/10 land as 400 train / 50 val / 50 test with the correct label type in
every slot -- the arrangement the paper describes. Background:
``src/utils/label_roles.py``.

Two guarantees this tool is built around, both verified rather than asserted in
a comment:

**Order is preserved exactly.** Output is trace records in source order, then
BigMILP records in source order. After writing, the file is re-read from disk
and every record's instance fingerprint is compared element-for-element against
the sources. A merge that silently reordered would be caught here, not by a
confusing training curve three days later.

**Inputs are never modified.** Both are opened read-only, and their md5 sums are
taken before and after the whole run and compared. The output path is refused if
it already exists unless --force is passed, and the write is atomic (temp file
then rename) so an interrupted run cannot leave a half-file that looks valid.

Usage
-----
    python scripts/pipeline/merge_label_roles.py \
        --trace data/vertex_cover/label_..._forward_trace.pkl \
        --eval  data/vertex_cover/label_..._bigmilp.pkl \
        --out   data/vertex_cover/label_..._merged500.pkl

    # what would happen, touching nothing:
    python scripts/pipeline/merge_label_roles.py --trace A --eval B --out C --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.label_roles import (  # noqa: E402
    ROLE_EVAL,
    ROLE_TRAIN,
    instance_fingerprint,
    record_role,
    summarize_roles,
)

log = logging.getLogger("merge_label_roles")


def file_md5(path: str | os.PathLike) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: str | os.PathLike) -> list:
    with open(path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise TypeError(f"{path}: expected a list of records, got {type(records).__name__}")
    if not records:
        raise ValueError(f"{path}: file is empty")
    return records


def check_uniform_role(records: list, expected: str, path: str) -> None:
    offenders = [i for i, r in enumerate(records) if record_role(r) != expected]
    if offenders:
        got = {i: record_role(records[i]) for i in offenders[:5]}
        raise ValueError(
            f"{path}: expected every record to have role {expected!r}, but "
            f"{len(offenders)} do not. First offenders (index -> role): {got}. "
            f"Role histogram: {summarize_roles(records)}"
        )


def merge(trace_path: str, eval_path: str, out_path: str, *, force: bool, dry_run: bool) -> dict:
    out = Path(out_path)
    if out.exists() and not force and not dry_run:
        raise FileExistsError(
            f"{out_path} already exists. Refusing to overwrite a label file; pass "
            f"--force only if you are certain, or choose a new name."
        )

    md5_trace_before, md5_eval_before = file_md5(trace_path), file_md5(eval_path)
    log.info("input md5  trace=%s  eval=%s", md5_trace_before, md5_eval_before)

    trace_records = load_records(trace_path)
    eval_records = load_records(eval_path)
    log.info("loaded %d trace records, %d eval records", len(trace_records), len(eval_records))

    # 1. every record must carry exactly the role its file claims
    check_uniform_role(trace_records, ROLE_TRAIN, trace_path)
    check_uniform_role(eval_records, ROLE_EVAL, eval_path)

    # 2. the two files must describe disjoint instances -- otherwise the merged
    #    "test" set is not held out at all
    fp_trace = [instance_fingerprint(r) for r in trace_records]
    fp_eval = [instance_fingerprint(r) for r in eval_records]
    overlap = set(fp_trace) & set(fp_eval)
    if overlap:
        raise ValueError(
            f"{len(overlap)} instance(s) appear in BOTH files. Merging would put the "
            f"same instance in training and in the held-out test set."
        )
    for name, fps in (("trace", fp_trace), ("eval", fp_eval)):
        if len(set(fps)) != len(fps):
            raise ValueError(f"{name} file contains duplicate instances ({len(fps) - len(set(fps))} repeats)")

    merged = list(trace_records) + list(eval_records)
    boundary = len(trace_records)

    # 3. in-memory order check before anything is written
    fp_merged = [instance_fingerprint(r) for r in merged]
    if fp_merged != fp_trace + fp_eval:
        raise AssertionError("in-memory merge reordered records; refusing to write")

    manifest = {
        "trace_file": str(trace_path),
        "eval_file": str(eval_path),
        "out_file": str(out_path),
        "n_trace": len(trace_records),
        "n_eval": len(eval_records),
        "n_total": len(merged),
        "boundary_index": boundary,
        "md5_trace_in": md5_trace_before,
        "md5_eval_in": md5_eval_before,
        "roles_merged": summarize_roles(merged),
    }

    if dry_run:
        log.info("dry run: would write %d records to %s", len(merged), out_path)
        manifest["dry_run"] = True
        return manifest

    # 4. atomic write
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)

    # 5. read back from disk and verify order and roles survived serialization
    reread = load_records(out)
    if len(reread) != len(merged):
        raise AssertionError(f"read-back length {len(reread)} != {len(merged)}")
    fp_reread = [instance_fingerprint(r) for r in reread]
    if fp_reread != fp_trace + fp_eval:
        bad = next(i for i, (a, b) in enumerate(zip(fp_reread, fp_trace + fp_eval)) if a != b)
        raise AssertionError(f"read-back order differs from source at index {bad}")
    for i in range(boundary):
        if record_role(reread[i]) != ROLE_TRAIN:
            raise AssertionError(f"read-back index {i} is not a training record")
    for i in range(boundary, len(reread)):
        if record_role(reread[i]) != ROLE_EVAL:
            raise AssertionError(f"read-back index {i} is not an evaluation record")

    # 6. inputs must be byte-identical to how we found them
    md5_trace_after, md5_eval_after = file_md5(trace_path), file_md5(eval_path)
    if (md5_trace_after, md5_eval_after) != (md5_trace_before, md5_eval_before):
        raise AssertionError(
            "an input file changed during the merge -- "
            f"trace {md5_trace_before}->{md5_trace_after}, eval {md5_eval_before}->{md5_eval_after}"
        )

    manifest["md5_out"] = file_md5(out)
    manifest["md5_trace_out"] = md5_trace_after
    manifest["md5_eval_out"] = md5_eval_after

    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    log.info("wrote %s (%d records, boundary %d)", out_path, len(merged), boundary)
    log.info("manifest %s", manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True, help="forward-trace label pickle (training records)")
    parser.add_argument("--eval", required=True, dest="eval_path", help="BigMILP label pickle (evaluation records)")
    parser.add_argument("--out", required=True, help="output pickle; must not already exist")
    parser.add_argument("--force", action="store_true", help="permit overwriting an existing output")
    parser.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    manifest = merge(args.trace, args.eval_path, args.out, force=args.force, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
