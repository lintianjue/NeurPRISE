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
  ./scripts/01_create_instance.sh [PROBLEM] [N_NODES] [N_SCENARIOS] [N_INSTANCE]
  ./scripts/01_create_instance.sh cflp|cflp_ebp|cflp_ebp_mm|cflp_ebp_normal [N_FACILITIES] [N_CUSTOMERS] [N_SCENARIOS] [N_INSTANCE]

Args (positional):
  PROBLEM      Problem name (default: vertex_cover)
  N_NODES      Number of nodes (default: 8)
  N_SCENARIOS  Number of scenarios (default: 10)
  N_INSTANCE   Number of instances to generate (default: 200)
  N_FACILITIES Number of facilities for cflp variants (default: 30)
  N_CUSTOMERS  Number of customers for cflp variants (default: 30)

Examples:
  ./scripts/01_create_instance.sh
  ./scripts/01_create_instance.sh vertex_cover 20 50 500
  ./scripts/01_create_instance.sh selection 50 100 1000
  ./scripts/01_create_instance.sh cflp 30 30 10 200
  ./scripts/01_create_instance.sh cflp_ebp 30 30 10 200
  ./scripts/01_create_instance.sh cflp_ebp_mm 30 30 10 200
  ./scripts/01_create_instance.sh cflp_ebp_normal 30 30 10 200

Notes:
  - Set INTERACTIVE=1 to be prompted per-problem for the right dimensions.
EOF
}

# Help flags
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Positional args with defaults
# The CFLP family, in ONE place.  It was spelled out inline at 2 sites here
# and 5 in 02_label.sh; adding a variant meant editing all seven, and the
# one that got missed was the bug.  Membership is a case, not a substring.
_is_cflp_family() {
  case "${1}" in
    cflp|cflp_ebp|cflp_ebp_mm|cflp_ebp_normal) return 0 ;;
    *) return 1 ;;
  esac
}

PROBLEM="${1:-vertex_cover}"
INTERACTIVE="${INTERACTIVE:-0}"

# Basic integer validation
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

if [[ "${INTERACTIVE}" == "1" && "$(type -t prompt_with_default)" == "function" ]]; then
  PROBLEM="$(prompt_with_default "PROBLEM" "${PROBLEM}")"
  if _is_cflp_family "${PROBLEM}"; then
    N_FACILITIES="$(prompt_with_default "N_FACILITIES" "${2:-30}")"
    N_CUSTOMERS="$(prompt_with_default "N_CUSTOMERS" "${3:-30}")"
    N_SCENARIOS="$(prompt_with_default "N_SCENARIOS" "${4:-10}")"
    N_INSTANCE="$(prompt_with_default "N_INSTANCE" "${5:-200}")"
  else
    N_NODES="$(prompt_with_default "N_NODES" "${2:-8}")"
    N_SCENARIOS="$(prompt_with_default "N_SCENARIOS" "${3:-10}")"
    N_INSTANCE="$(prompt_with_default "N_INSTANCE" "${4:-200}")"
  fi
fi

if _is_cflp_family "${PROBLEM}"; then
  N_FACILITIES="${N_FACILITIES:-${2:-30}}"
  N_CUSTOMERS="${N_CUSTOMERS:-${3:-30}}"
  N_SCENARIOS="${N_SCENARIOS:-${4:-10}}"
  N_INSTANCE="${N_INSTANCE:-${5:-200}}"

  if ! is_int "${N_FACILITIES}" || ! is_int "${N_CUSTOMERS}" || ! is_int "${N_SCENARIOS}" || ! is_int "${N_INSTANCE}"; then
    usage
    die "N_FACILITIES, N_CUSTOMERS, N_SCENARIOS, and N_INSTANCE must be non-negative integers."
  fi

  echo "Running: create_instance problem=${PROBLEM} n_facilities=${N_FACILITIES} n_customers=${N_CUSTOMERS} n_scenarios=${N_SCENARIOS} instance_cnt=${N_INSTANCE}"
    python "scripts/01_create_inst_spec.py" \
      problem="${PROBLEM}" \
      problem.instance.n_facilities="${N_FACILITIES}" \
      problem.instance.n_customers="${N_CUSTOMERS}" \
      problem.instance.n_scenarios="${N_SCENARIOS}" \
      data.instance_cnt="${N_INSTANCE}"
else
  N_NODES="${N_NODES:-${2:-8}}"
  N_SCENARIOS="${N_SCENARIOS:-${3:-10}}"
  N_INSTANCE="${N_INSTANCE:-${4:-200}}"

  if ! is_int "${N_NODES}" || ! is_int "${N_SCENARIOS}" || ! is_int "${N_INSTANCE}"; then
    usage
    die "N_NODES, N_SCENARIOS, and N_INSTANCE must be non-negative integers."
  fi

  P_SELECT=$(( N_NODES / 2 ))

  echo "Running: create_instance problem=${PROBLEM} n_nodes=${N_NODES} n_scenarios=${N_SCENARIOS} instance_cnt=${N_INSTANCE} p_select=${P_SELECT}"

  if [[ "${PROBLEM}" == "selection" || "${PROBLEM}" == "selection_mm" || "${PROBLEM}" == "selection_normal" ]]; then
      python "scripts/01_create_inst_spec.py" \
        problem="${PROBLEM}" \
        problem.instance.n_nodes="${N_NODES}" \
        problem.instance.n_scenarios="${N_SCENARIOS}" \
        problem.instance.p_select="${P_SELECT}" \
        data.instance_cnt="${N_INSTANCE}"
  else
      python "scripts/01_create_inst_spec.py" \
        problem="${PROBLEM}" \
        problem.instance.n_nodes="${N_NODES}" \
        problem.instance.n_scenarios="${N_SCENARIOS}" \
        problem.instance.p_select=10 \
        data.instance_cnt="${N_INSTANCE}"
  fi
fi
