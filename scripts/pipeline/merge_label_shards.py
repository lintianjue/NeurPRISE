#!/usr/bin/env python
"""Recombine the per-shard outputs of a parallel `02_prise_label.py` run.

Shards are produced by `label.prise.index_start/index_end` and land next to the
canonical label file as `..._shard{start}-{end}.pkl`. This script concatenates
them in index order and refuses to write a merged file that is short.

A short shard is not cosmetic: `02_prise_label.py` `continue`s past an instance
whose forward heuristic raised, so a silently truncated merge would train on a
different set than the one the split arithmetic assumes.

Usage:
  python scripts/pipeline/merge_label_shards.py \
      --label-path data/selection/label_prise_seed42_cnt2500_n20_s50_p10_scen_uniform.pkl \
      --expect-start 0 --expect-end 2250
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

# Unpickling a shard rebuilds src.core.types.{Instance,Labels}, so `src` MUST resolve to THIS
# repo. Run as `python scripts/pipeline/merge_label_shards.py ...` and sys.path[0] is scripts/, not the
# repo root -- and an editable install of another project that ships a top-level `src` registers
# an import finder that then claims `src`. Same guard as scripts/pipeline/label_monotonicity_check.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SHARD_RE = re.compile(r"_shard(\d+)-(\d+)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-path", required=True,
                    help="Canonical (unsharded) label path to write.")
    ap.add_argument("--expect-start", type=int, required=True)
    ap.add_argument("--expect-end", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = Path(args.label_path)
    shards = []
    for p in target.parent.glob(f"{target.stem}_shard*{target.suffix}"):
        m = SHARD_RE.search(p.stem)
        if m:
            shards.append((int(m.group(1)), int(m.group(2)), p))
    shards.sort()

    if not shards:
        print(f"ERROR: no shards matching {target.stem}_shard*{target.suffix}",
              file=sys.stderr)
        return 2

    merged = []
    cursor = args.expect_start
    for start, end, path in shards:
        if start != cursor:
            print(f"ERROR: gap or overlap — expected shard starting at {cursor}, "
                  f"found {path.name}", file=sys.stderr)
            return 2
        with open(path, "rb") as f:
            pairs = pickle.load(f)
        if len(pairs) != end - start:
            print(f"ERROR: {path.name} holds {len(pairs)} pairs, expected "
                  f"{end - start}. Instances were dropped (see the shard log for "
                  f"'Forward heuristic failed'); re-run this shard.", file=sys.stderr)
            return 2
        merged.extend(pairs)
        cursor = end

    if cursor != args.expect_end:
        print(f"ERROR: shards cover [{args.expect_start}, {cursor}), expected end "
              f"{args.expect_end}.", file=sys.stderr)
        return 2

    print(f"{len(shards)} shards, {len(merged)} instance+label pairs, "
          f"covering [{args.expect_start}, {args.expect_end}).")
    if args.dry_run:
        print("--dry-run: not writing.")
        return 0

    with open(target, "xb") as f:  # never replace an existing label file
        pickle.dump(merged, f)
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
