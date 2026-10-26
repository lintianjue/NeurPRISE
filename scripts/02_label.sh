#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_LIB="${SCRIPT_DIR}/script_lib.sh"

if [[ -f "${SCRIPT_LIB}" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_LIB}"
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  ./scripts/02_label.sh [PROBLEM] [N_NODES] [N_SCENARIOS] [LABEL_SCRIPT] [N_INSTANCE] [K_RATIO] [SOLVER_THREADS] [BIG_MILP_TIME_LIMIT] [PRISE_TIME_LIMIT] [PRISE_LABEL_OUTPUT] [PROCESS_FIRST_PCT] [PROCESS_LAST_PCT]
  ./scripts/02_label.sh cflp|cflp_ebp|cflp_ebp_mm|cflp_ebp_normal [N_FACILITIES] [N_CUSTOMERS] [N_SCENARIOS] [LABEL_SCRIPT] [N_INSTANCE] [K_RATIO] [SOLVER_THREADS] [BIG_MILP_TIME_LIMIT] [PRISE_TIME_LIMIT] [PRISE_LABEL_OUTPUT] [PROCESS_FIRST_PCT] [PROCESS_LAST_PCT]

Args (positional):
  PROBLEM           Problem name (default: vertex_cover)
  N_NODES           Number of nodes (default: 8)
  N_SCENARIOS       Number of scenarios (default: 10)
  LABEL_SCRIPT      Script suffix for scripts/02_<suffix>.py (default: prise_label)
  N_INSTANCE        Number of instances to label (default: 100)
  K_RATIO           Ratio for label.prise.k_ratio (default: 0.4)
  SOLVER_THREADS    Solver threads (default: 12)
  BIG_MILP_TIME_LIMIT  (default: 60)
  PRISE_TIME_LIMIT     (default: 10)
  PRISE_LABEL_OUTPUT   (default: both)
  PROCESS_FIRST_PCT    Only process first X% instances (default: unset)
  PROCESS_LAST_PCT     Only process last  X% instances (default: unset)
                   Note: PROCESS_FIRST_PCT and PROCESS_LAST_PCT are mutually exclusive.
  Env: MIP_GAP         Solver optimality tolerance (default: 1e-4)

Notes:
  - This runs: python scripts/02_${LABEL_SCRIPT}.py
  - K_RATIO typically should be reduced as N_SCENARIOS increases.
  - For PRISE output_mode=bigmilp|both, a timing summary is saved to:
      eval/<problem>/s<seed>_i<instances>_n<n_nodes>_s<n_scenarios>/bigmilp.json
      eval/<problem>/s<seed>_i<instances>_m<n_facilities>_n<n_customers>_s<n_scenarios>/bigmilp.json

Examples:
  ./scripts/02_label.sh
  ./scripts/02_label.sh vertex_cover 8 10 prise_label 100 0.4 12 60 10 both 50
  ./scripts/02_label.sh vertex_cover 20 50 prise_label 500 0.2 16 60 10 both "" 10
  ./scripts/02_label.sh cflp 30 30 10 prise_label 100 0.2 16 60 10 both
  ./scripts/02_label.sh cflp_ebp 30 30 10 prise_label 100 0.2 16 60 10 both
EOF
}

# Help flags
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Positional args with defaults
# Problems whose shape is (n_facilities, n_customers) instead of n_nodes.
_is_cflp_family() {
  case "${1}" in
    cflp|cflp_ebp|cflp_ebp_mm|cflp_ebp_normal) return 0 ;;
    *) return 1 ;;
  esac
}

PROBLEM="${1:-cflp}"
INTERACTIVE="${INTERACTIVE:-0}"
MIP_GAP="${MIP_GAP:-1e-4}"

if [[ "${INTERACTIVE}" == "1" && "$(type -t prompt_with_default)" == "function" ]]; then
  PROBLEM="$(prompt_with_default "PROBLEM" "${PROBLEM}")"
  if _is_cflp_family "${PROBLEM}"; then
    N_FACILITIES="$(prompt_with_default "N_FACILITIES" "${2:-30}")"
    N_CUSTOMERS="$(prompt_with_default "N_CUSTOMERS" "${3:-30}")"
    N_SCENARIOS="$(prompt_with_default "N_SCENARIOS" "${4:-10}")"
    LABEL_SCRIPTS="$(prompt_with_default "LABEL_SCRIPT" "${5:-prise_label}")"
    N_INSTANCE="$(prompt_with_default "N_INSTANCE" "${6:-100}")"
    K_RATIO="$(prompt_with_default "K_RATIO" "${7:-0.16}")"
    SOLVER_THREADS="$(prompt_with_default "SOLVER_THREADS" "${8:-12}")"
    BIG_MILP_TIME_LIMIT="$(prompt_with_default "BIG_MILP_TIME_LIMIT" "${9:-60}")"
    PRISE_TIME_LIMIT="$(prompt_with_default "PRISE_TIME_LIMIT" "${10:-10}")"
    PRISE_LABEL_OUTPUT="$(prompt_with_default "PRISE_LABEL_OUTPUT" "${11:-both}")"
    PROCESS_FIRST_PCT="$(prompt_with_default "PROCESS_FIRST_PCT" "${12:-}")"
    PROCESS_LAST_PCT="$(prompt_with_default "PROCESS_LAST_PCT" "${13:-}")"
  else
    N_NODES="$(prompt_with_default "N_NODES" "${2:-8}")"
    N_SCENARIOS="$(prompt_with_default "N_SCENARIOS" "${3:-10}")"
    LABEL_SCRIPTS="$(prompt_with_default "LABEL_SCRIPT" "${4:-prise_label}")"
    N_INSTANCE="$(prompt_with_default "N_INSTANCE" "${5:-100}")"
    K_RATIO="$(prompt_with_default "K_RATIO" "${6:-0.16}")"
    SOLVER_THREADS="$(prompt_with_default "SOLVER_THREADS" "${7:-12}")"
    BIG_MILP_TIME_LIMIT="$(prompt_with_default "BIG_MILP_TIME_LIMIT" "${8:-60}")"
    PRISE_TIME_LIMIT="$(prompt_with_default "PRISE_TIME_LIMIT" "${9:-10}")"
    PRISE_LABEL_OUTPUT="$(prompt_with_default "PRISE_LABEL_OUTPUT" "${10:-both}")"
    PROCESS_FIRST_PCT="$(prompt_with_default "PROCESS_FIRST_PCT" "${11:-}")"
    PROCESS_LAST_PCT="$(prompt_with_default "PROCESS_LAST_PCT" "${12:-}")"
  fi
fi

if _is_cflp_family "${PROBLEM}"; then
  N_FACILITIES="${N_FACILITIES:-${2:-30}}"
  N_CUSTOMERS="${N_CUSTOMERS:-${3:-30}}"
  N_SCENARIOS="${N_SCENARIOS:-${4:-50}}"
  LABEL_SCRIPTS="${LABEL_SCRIPTS:-${5:-prise_label}}"
  N_INSTANCE="${N_INSTANCE:-${6:-100}}"
  K_RATIO="${K_RATIO:-${7:-0.16}}"      # K_ratio should be reduced as N_SCENARIOS increases
  SOLVER_THREADS="${SOLVER_THREADS:-${8:-12}}"
  BIG_MILP_TIME_LIMIT="${BIG_MILP_TIME_LIMIT:-${9:-60}}"
  PRISE_TIME_LIMIT="${PRISE_TIME_LIMIT:-${10:-10}}"
  PRISE_LABEL_OUTPUT="${PRISE_LABEL_OUTPUT:-${11:-both}}"
  PROCESS_FIRST_PCT="${PROCESS_FIRST_PCT:-${12:-}}"
  PROCESS_LAST_PCT="${PROCESS_LAST_PCT:-${13:-}}"
else
  N_NODES="${N_NODES:-${2:-8}}"
  N_SCENARIOS="${N_SCENARIOS:-${3:-10}}"
  LABEL_SCRIPTS="${LABEL_SCRIPTS:-${4:-prise_label}}"
  N_INSTANCE="${N_INSTANCE:-${5:-100}}"
  K_RATIO="${K_RATIO:-${6:-0.16}}"      # K_ratio should be reduced as N_SCENARIOS increases
  SOLVER_THREADS="${SOLVER_THREADS:-${7:-12}}"
  BIG_MILP_TIME_LIMIT="${BIG_MILP_TIME_LIMIT:-${8:-60}}"
  PRISE_TIME_LIMIT="${PRISE_TIME_LIMIT:-${9:-10}}"
  PRISE_LABEL_OUTPUT="${PRISE_LABEL_OUTPUT:-${10:-both}}"
  PROCESS_FIRST_PCT="${PROCESS_FIRST_PCT:-${11:-}}"
  PROCESS_LAST_PCT="${PROCESS_LAST_PCT:-${12:-}}"
fi

# Basic validation
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; }  # float or scientific

if _is_cflp_family "${PROBLEM}"; then
  if ! is_int "$N_FACILITIES" || ! is_int "$N_CUSTOMERS" || ! is_int "$N_SCENARIOS" || ! is_int "$N_INSTANCE" || ! is_int "$SOLVER_THREADS"; then
    usage
    die "N_FACILITIES, N_CUSTOMERS, N_SCENARIOS, N_INSTANCE, SOLVER_THREADS must be non-negative integers."
  fi
else
  if ! is_int "$N_NODES" || ! is_int "$N_SCENARIOS" || ! is_int "$N_INSTANCE" || ! is_int "$SOLVER_THREADS"; then
    usage
    die "N_NODES, N_SCENARIOS, N_INSTANCE, SOLVER_THREADS must be non-negative integers."
  fi
fi

if ! is_num "$K_RATIO"; then
  usage
  die "K_RATIO must be a number (e.g., 0.4)."
fi
if ! is_num "$MIP_GAP"; then
  usage
  die "MIP_GAP must be a number (e.g., 1e-4)."
fi

if [[ -n "${PROCESS_FIRST_PCT}" && -n "${PROCESS_LAST_PCT}" ]]; then
  usage
  die "PROCESS_FIRST_PCT and PROCESS_LAST_PCT are mutually exclusive. Set only one."
fi

if [[ -n "${PROCESS_FIRST_PCT}" ]] && ! is_num "${PROCESS_FIRST_PCT}"; then
  usage
  die "PROCESS_FIRST_PCT must be a number (e.g., 50)."
fi

if [[ -n "${PROCESS_LAST_PCT}" ]] && ! is_num "${PROCESS_LAST_PCT}"; then
  usage
  die "PROCESS_LAST_PCT must be a number (e.g., 10)."
fi

SCRIPT_PATH="scripts/02_${LABEL_SCRIPTS}.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  usage
  die "cannot find ${SCRIPT_PATH}"
fi

if _is_cflp_family "${PROBLEM}"; then
  echo "Running: ${SCRIPT_PATH}"
  echo "  problem=${PROBLEM} n_facilities=${N_FACILITIES} n_customers=${N_CUSTOMERS} n_scenarios=${N_SCENARIOS} instance_cnt=${N_INSTANCE} k_ratio=${K_RATIO} mip_gap=${MIP_GAP} threads=${SOLVER_THREADS}"
else
  P_SELECT=$(( N_NODES / 2 ))
  echo "Running: ${SCRIPT_PATH}"
  echo "  problem=${PROBLEM} n_nodes=${N_NODES} n_scenarios=${N_SCENARIOS} instance_cnt=${N_INSTANCE} p_select=${P_SELECT} k_ratio=${K_RATIO} mip_gap=${MIP_GAP} threads=${SOLVER_THREADS}"
fi

EXTRA_LABEL_ARGS=()
if [[ -n "${PROCESS_FIRST_PCT}" ]]; then
  EXTRA_LABEL_ARGS+=( "label.prise.process_first_pct=${PROCESS_FIRST_PCT}" )
fi
if [[ -n "${PROCESS_LAST_PCT}" ]]; then
  EXTRA_LABEL_ARGS+=( "label.prise.process_last_pct=${PROCESS_LAST_PCT}" )
fi

if _is_cflp_family "${PROBLEM}"; then
  N_FACILITIES="${N_FACILITIES:-${N_NODES}}"
  N_CUSTOMERS="${N_CUSTOMERS:-${N_CUSTOMERS:-}}"
  if [[ -z "${N_CUSTOMERS}" ]]; then
    die "Set N_CUSTOMERS for cflp labeling (env or CLI)."
  fi

    python "${SCRIPT_PATH}" \
      problem="${PROBLEM}" \
      problem.instance.n_facilities="${N_FACILITIES}" \
      problem.instance.n_customers="${N_CUSTOMERS}" \
      problem.instance.n_scenarios="${N_SCENARIOS}" \
      solver.threads="${SOLVER_THREADS}" \
      solver.mip_gap="${MIP_GAP}" \
      solver.time_limit="${BIG_MILP_TIME_LIMIT}" \
      solver.prise_time_limit="${PRISE_TIME_LIMIT}" \
      data.instance_cnt="${N_INSTANCE}" \
      label.prise.k_ratio="${K_RATIO}" \
      label.prise.output="${PRISE_LABEL_OUTPUT}" \
      "${EXTRA_LABEL_ARGS[@]}"
else
    python "${SCRIPT_PATH}" \
      problem="${PROBLEM}" \
      problem.instance.n_nodes="${N_NODES}" \
      problem.instance.n_scenarios="${N_SCENARIOS}" \
      problem.instance.p_select="${P_SELECT}" \
      solver.threads="${SOLVER_THREADS}" \
      solver.mip_gap="${MIP_GAP}" \
      solver.time_limit="${BIG_MILP_TIME_LIMIT}" \
      solver.prise_time_limit="${PRISE_TIME_LIMIT}" \
      data.instance_cnt="${N_INSTANCE}" \
      label.prise.k_ratio="${K_RATIO}" \
      label.prise.output="${PRISE_LABEL_OUTPUT}" \
      "${EXTRA_LABEL_ARGS[@]}"
fi
