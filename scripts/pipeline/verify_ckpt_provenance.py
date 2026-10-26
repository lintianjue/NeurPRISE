#!/usr/bin/env python3
"""Consumer-side provenance gate: a checkpoint may not be evaluated unless a sidecar
beside it records its digest, and that digest matches the bytes about to be loaded.

WHY THIS IS AT THE CONSUMER AND NOT IN THE TRAINING RUNNER
----------------------------------------------------------
A checkpoint can be produced outside the pipeline and placed by hand into a directory
no script writes to — no log, no Hydra record, no provenance file — so its
arm-to-checkpoint mapping rests on the filename. **Hardening the runner cannot catch
that**, because the failure mode is a checkpoint arriving in the expected path without
passing through the runner at all. Only the consumer sees what is actually loaded.

Echoing a digest without comparing it is worse than no check: the digest lands in the
log and reads as verification to anyone scanning it.

FAIL CLOSED — every one of these is FATAL
-----------------------------------------
  * checkpoint absent
  * sidecar absent                      <- must not be a warning
  * sidecar unreadable or not valid JSON
  * `checkpoint.best_pt_md5` absent, null or empty
  * `checkpoint.best_pt_md5` present and mismatched
  * the sidecar's digest describes a DIFFERENT file than the one being loaded
  * any unexpected error while checking

Treating a missing field as "nothing to compare, carry on" would make the gate
decorative, so absence is failure, not a skip.

NO BYPASS FLAG. A checkpoint without a sidecar fails this, and that is correct — its
honest state *is* "no provenance". Evaluating such a checkpoint is a decision to be
taken explicitly, not a flag to be passed; a `--skip-provenance` would be used the first time the check is inconvenient, which is
exactly when it is working.

Usage:
    python3 scripts/pipeline/verify_ckpt_provenance.py CKPT [CKPT ...]
    python3 scripts/pipeline/verify_ckpt_provenance.py --quiet CKPT     # only on failure
Exit 0 iff every checkpoint passed. Exit 4 on any failure (the runner's FATAL code).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.run_metadata import SIDECAR_SUFFIX, sidecar_path, file_md5  # noqa: E402

# The digest field is `checkpoint.best_pt_md5`, NOT a new `ckpt_md5`. It already exists in
# the training-side run records; a second name for the same quantity would fork the
# contract this gate exists to hold. `DIGEST_PATH` is quoted verbatim in every failure
# message so a reader can see which field was looked for.
DIGEST_PATH = ("checkpoint", "best_pt_md5")
SUBJECT_PATH = ("checkpoint", "best_pt")     # which file the digest describes
ARM_PATH = ("checkpoint", "arm")
EXIT_FAIL = 4


def _dig(meta: dict, path: tuple[str, ...]):
    """Fetch a nested field, returning (found, value)."""
    node = meta
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _fmt(path: tuple[str, ...]) -> str:
    return ".".join(path)


def check_one(ckpt: Path, expect_arm: str | None = None) -> tuple[bool, str]:
    """Return (ok, message). Any error path returns False -- never raises past here.

    `expect_arm` is the arm the CALLER is loading this checkpoint for. When given,
    `checkpoint.arm` must be present and equal. This is the half the digest cannot do:
    the digest proves the file is intact, not that it is the `tl2s` model. A filename is
    a claim, not evidence.
    """
    try:
        if not ckpt.is_file():
            return False, f"checkpoint absent: {ckpt}"

        side = sidecar_path(ckpt)
        if not side.is_file():
            # Still fail closed -- but say WHICH kind of absence this is. There is a second
            # sidecar convention (`<output_tag>.run.json` in the cell directory), and
            # reporting "no provenance" for a checkpoint that has provenance under a
            # different contract sends the operator hunting for the wrong thing. The gate does not accept it: two contracts is the defect, and
            # blessing one from the consumer would entrench the fork.
            # Only name sidecars that are ORPHANED -- i.e. whose implied subject file does
            # not exist, which is the signature of the tag-named convention. Listing every
            # sibling would flag a directory of well-formed per-checkpoint sidecars as a
            # convention clash, which is the opposite of helpful.
            others = sorted(
                p.name for p in ckpt.parent.glob("*" + SIDECAR_SUFFIX)
                if not Path(str(p)[: -len(SIDECAR_SUFFIX)]).is_file())
            hint = ""
            if others:
                hint = (f" NOTE: this directory does contain {others} — a different sidecar "
                        f"convention. That is not accepted here: the contract is "
                        f"'<checkpoint>{SIDECAR_SUFFIX} carrying {_fmt(DIGEST_PATH)}', and the "
                        f"two conventions need reconciling rather than the consumer "
                        f"accepting both.")
            return False, (
                f"no provenance sidecar: expected {side.name} beside the checkpoint. "
                f"A checkpoint with no recorded origin cannot be evaluated; if this is a "
                f"legacy hand-placed file, that is a decision to take explicitly, not a "
                f"flag to pass.{hint}")

        try:
            with open(side, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except json.JSONDecodeError as exc:
            return False, f"sidecar is not valid JSON ({side.name}): {exc}"
        except OSError as exc:
            return False, f"sidecar unreadable ({side.name}): {exc}"

        if not isinstance(meta, dict):
            return False, f"sidecar {side.name} is a {type(meta).__name__}, expected an object"

        found, recorded = _dig(meta, DIGEST_PATH)
        if not found:
            return False, (
                f"sidecar {side.name} has no {_fmt(DIGEST_PATH)!r} field. This is the case "
                f"that reproduces the defect being replaced: a missing digest is NOT "
                f"'nothing to compare', it is an unverifiable checkpoint")
        if recorded is None or not str(recorded).strip():
            return False, (
                f"sidecar {side.name} has an empty {_fmt(DIGEST_PATH)!r}. NOTE: an empty "
                f"'checkpoint' block usually means the emitter's population step failed, not "
                f"that provenance was declined — check the sidecar's 'config' block for an "
                f"error before treating this checkpoint as unprovenanced")

        # WHICH FILE does the recorded digest describe? `best_pt_md5` is the digest of the
        # file named in `checkpoint.best_pt`, which is NOT always the file being loaded:
        # the trainer writes both `<longname>_best.pt` and a `best.pt` alias, the two are
        # separate serialisations of the SAME weights, and their file digests differ.
        # Measured on one pair: identical model_state, epoch and val_loss; file
        # sizes 1,718,373 vs 1,669,316. Comparing a `best.pt` digest against the long-named
        # file is a guaranteed false mismatch, so say so rather than cry corruption.
        _, subject = _dig(meta, SUBJECT_PATH)
        if subject is not None:
            subj_name = Path(str(subject)).name
            if subj_name and subj_name != ckpt.name:
                sibling = ckpt.parent / subj_name
                extra = ""
                if sibling.is_file() and file_md5(sibling).lower() == str(recorded).strip().lower():
                    extra = (f" The digest DOES match {subj_name} in the same directory, so "
                             f"this is a sidecar describing a different file, not a corrupt "
                             f"checkpoint.")
                return False, (
                    f"sidecar {side.name} describes {subj_name!r} "
                    f"({_fmt(SUBJECT_PATH)}), but the file being loaded is {ckpt.name!r}. "
                    f"A digest that describes a different file verifies nothing about this "
                    f"one.{extra}")

        actual = file_md5(ckpt)
        if str(recorded).strip().lower() != actual.lower():
            return False, (
                f"DIGEST MISMATCH for {ckpt.name}: sidecar records {recorded}, file is "
                f"{actual}. The bytes about to be loaded are not the bytes the sidecar "
                f"describes")

        # Identity, not just integrity. Only checked when the caller states an
        # expectation -- a consumer that does not know which arm it wants cannot be
        # protected here, and pretending otherwise would be the decorative kind of check.
        if expect_arm is not None:
            arm_found, got = _dig(meta, ARM_PATH)
            if not arm_found:
                return False, (
                    f"sidecar {side.name} records no {_fmt(ARM_PATH)!r}, but this is being "
                    f"loaded as {expect_arm!r}. Integrity without identity: the digest "
                    f"proves the file is intact, not that it is the {expect_arm!r} model")
            if got is None or not str(got).strip():
                return False, (
                    f"sidecar {side.name} has a null/empty {_fmt(ARM_PATH)!r}, but this "
                    f"checkpoint is being loaded as {expect_arm!r}")
            if str(got).strip() != str(expect_arm).strip():
                return False, (
                    f"ARM MISMATCH for {ckpt.name}: sidecar says this is the {got!r} "
                    f"checkpoint, but it is being loaded as {expect_arm!r}. This is the "
                    f"failure the filename-based mapping cannot see")

        arm_note = f" arm={_dig(meta, ARM_PATH)[1]!r}" if expect_arm is not None else ""
        return True, f"{ckpt.name} md5 {actual} matches {side.name}{arm_note}"
    except Exception as exc:  # fail closed on anything unforeseen
        return False, f"provenance check errored on {ckpt} ({type(exc).__name__}: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+", type=Path,
                    help="checkpoint paths, or ARM=PATH to also assert the arm")
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args()

    # `ARM=PATH` lets a caller state, per checkpoint, which arm it believes it is loading.
    pairs: list[tuple[Path, str | None]] = []
    for raw in args.ckpts:
        s = str(raw)
        if "=" in s and not Path(s).exists():
            arm, _, path = s.partition("=")
            pairs.append((Path(path), arm))
        else:
            pairs.append((Path(s), None))

    failures = 0
    for ckpt, expect_arm in pairs:
        ok, msg = check_one(ckpt, expect_arm=expect_arm)
        if ok:
            if not args.quiet:
                print(f"[provenance] OK   {msg}")
        else:
            failures += 1
            print(f"[provenance] FATAL {msg}", file=sys.stderr)

    if failures:
        print(f"[provenance] {failures} of {len(args.ckpts)} checkpoint(s) failed; refusing "
              f"to evaluate.", file=sys.stderr)
        return EXIT_FAIL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
