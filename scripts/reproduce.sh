#!/usr/bin/env bash
# Reproduce the paper's numbers. Run from anywhere; see doc/REPRODUCE.md.
#
#   scripts/reproduce.sh download            fetch checkpoints, instances, labels and required
#                                            evaluation inputs into $NEURPRISE_INPUT_ROOT
#   scripts/reproduce.sh eval [--table N ...] [--threads N] [--device cuda]
#                                            (--device: NeurPRISE scoring; the paper's timings use cuda)
#                                            rerun NeurPRISE from released checkpoints
#                                            (hours per table, days for all)
#   scripts/reproduce.sh data  <args>        regenerate instances  (scripts/01_create_instance.sh)
#   scripts/reproduce.sh label <args>        regenerate labels     (scripts/02_label.sh)
#   scripts/reproduce.sh train <args>        retrain a model       (scripts/03_train.sh)
#
# Options (before the mode): --dry-run prints the commands instead of running them.
# Environment: NEURPRISE_INPUT_ROOT (default: repo root), NEURPRISE_RUN_ROOT (default:
# <repo>/runs),
# HF_REPO (Hugging Face repository for `download`, default SanLouis/NeurPRISE).
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
MODE="${1:-}"
shift || true

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NEURPRISE_INPUT_ROOT="${NEURPRISE_INPUT_ROOT:-$REPO}"
export NEURPRISE_RUN_ROOT="${NEURPRISE_RUN_ROOT:-$REPO/runs}"
# Solver threads are set per command; BLAS defaults to 4 threads.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

case "$MODE" in
  download)
    run python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[2], local_dir=sys.argv[1], allow_patterns=['checkpoints/**/*.pt', 'data/**', 'eval_regret/full_objs_cache.json'])
" "$NEURPRISE_INPUT_ROOT" "${HF_REPO:-SanLouis/NeurPRISE}"
    ;;
  eval)
    mkdir -p "$NEURPRISE_RUN_ROOT"
    TABLES=() RUNS_ARGS=()
    while [[ "$#" -gt 0 ]]; do
      case "$1" in
        --table)
          if [[ "$#" -lt 2 || "$2" == --* ]]; then
            echo "eval: --table requires at least one table number" >&2; exit 2
          fi
          shift
          ;;
        --threads)
          if [[ "$#" -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
            echo "eval: --threads requires a positive integer" >&2; exit 2
          fi
          RUNS_ARGS+=(--threads "$2")
          shift 2
          ;;
        --device)
          if [[ "$#" -lt 2 || "$2" == --* ]]; then
            echo "eval: --device requires a value (cpu, cuda, cuda:N)" >&2; exit 2
          fi
          RUNS_ARGS+=(--device "$2")
          shift 2
          ;;
        --*) echo "eval: unknown option $1" >&2; exit 2 ;;
        *) TABLES+=("$1"); shift ;;
      esac
    done
    RUNS_ARGS+=(${TABLES[@]+--table "${TABLES[@]}"})
    SCRIPT="$NEURPRISE_RUN_ROOT/reproduce_eval.sh"
    python repro/runs.py ${RUNS_ARGS[@]+"${RUNS_ARGS[@]}"} > "$SCRIPT.tmp"
    mv "$SCRIPT.tmp" "$SCRIPT"
    echo "[$(date '+%F %T')] $(grep -c '^python' "$SCRIPT") commands -> $SCRIPT"
    if [[ "$DRY_RUN" -eq 1 ]]; then cat "$SCRIPT"; exit 0; fi
    bash "$SCRIPT"
    echo "[eval] Experiments completed. Results: $NEURPRISE_RUN_ROOT/repro/"
    ;;
  data)  run scripts/01_create_instance.sh "$@" ;;
  label) run scripts/02_label.sh "$@" ;;
  train) run scripts/03_train.sh "$@" ;;
  *)
    sed -n '2,/^set /{/^#/p}' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
