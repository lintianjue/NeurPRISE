#!/usr/bin/env bash
set -euo pipefail

# Thin, user-friendly wrapper around the Python GPU-pool scheduler.
# Prefer this entrypoint for day-to-day use.

abs_path() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/03_train_parallel_gpu_pool.sh [options]

Options:
  --gpus "0,2,3"           Comma-separated physical GPU ids to use
  --n_gpus <int>           GPUs per job (pool size must be >= N)
  --n_nodes <int>          (default: 20)
  --n_customers <int>      (default: unset; required for cflp variants)
  --n_scenarios <int>      (default: 50)
  --n_instances <int>      (default: 2500)
  --batch <int>            (default: 32)
  --lr <float>             Override train.optimizer.lr
  --ddp_trace              Enable DDP collective tracing (train.ddp_trace=true)
  --extra "<overrides>"    Extra Hydra overrides passed to training runs
  --train_config <name>    Hydra train config name (default: train)
  --log_dir <path>         Directory for per-run logs
  --dist_name <name>       Scenario name override (default: uniform)
  --exp "<spec>"           Add one experiment spec (repeatable)
                           Format: PROBLEM|MODEL|LOSS  or  MODEL|LOSS
  --dry_run                Print selected configs and expanded commands, then exit
  -h, --help               Show this help

Examples:
  scripts/03_train_parallel_gpu_pool.sh
  scripts/03_train_parallel_gpu_pool.sh --gpus "0,2,3" --n_nodes 20 --n_scenarios 50 --n_instances 2500 --batch 32
  N_GPUS=2 scripts/03_train_parallel_gpu_pool.sh --n_nodes 20 --n_scenarios 50 --n_instances 2500 --batch 32
  scripts/03_train_parallel_gpu_pool.sh --extra "train.epochs=50 seed=42"
  scripts/03_train_parallel_gpu_pool.sh --train_config train
  scripts/03_train_parallel_gpu_pool.sh --ddp_trace
  scripts/03_train_parallel_gpu_pool.sh --exp "vertex_cover|det_source_sce_mean|score_weighted_loss_flat"
EOF
}

die() { echo "Error: $*" >&2; exit 2; }

# Allow running from anywhere (repo root not required)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="${SCRIPT_DIR}/03_train_parallel_gpu_pool.py"
SCRIPT_LIB="${SCRIPT_DIR}/script_lib.sh"

if [[ -f "${SCRIPT_LIB}" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_LIB}"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${PY_RUNNER}" ]]; then
  die "Python runner not found: ${PY_RUNNER}"
fi

# Defaults (kept here for quick discoverability; Python runner also has defaults)
GPUS="0,3"
GPUS_FROM_ARG="0"
N_GPUS="${N_GPUS:-}"
N_NODES="20"
N_CUSTOMERS=""
N_SCENARIOS="50"
N_INSTANCES="2500"
BATCH="32"
LR="0.0006"
EXTRA=""
LOG_DIR="logs/train_parallel"
EXPS=()
DRY_RUN="0"
DDP_TRACE="0"
GPU_AUTO_SELECTED=""
DETECTED_BUSY=""
DETECTED_BUSY_NAMES=""
FALLBACK_USED="0"

INTERACTIVE="${INTERACTIVE:-0}"
CONFIRM="${CONFIRM:-0}"
VERBOSE="${VERBOSE:-0}"
AUTO_GPU="${AUTO_GPU:-1}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROBLEM_DIR="${ROOT_DIR}/configs/problem"
MODEL_DIR="${ROOT_DIR}/configs/model"
LOSS_DIR="${ROOT_DIR}/configs/losses"
DIST_NAME="${DIST_NAME:-uniform}"
SEED="${SEED:-42}"
TRAIN_CONFIG="train"
DEFAULT_PROBLEM="vertex_cover"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_NAME_FORMAT="exp_{i}_${RUN_ID}_p-{problem}_m-{model}_l-{loss}.log"

list_yaml_basenames() {
  local dir="${1:?dir required}"
  find "${dir}" -maxdepth 1 -type f -name "*.yaml" -printf "%f\n" | sed 's/\.yaml$//' | sort
}

print_indexed_list() {
  local title="${1:?title required}"
  shift
  local -a items=( "$@" )
  local i=0
  echo "${title}:"
  for item in "${items[@]}"; do
    i=$((i+1))
    printf '  %d) %s\n' "${i}" "${item}"
  done
}

format_cuda_devices() {
  local csv="${1:-}"
  if [[ -z "${csv}" ]]; then
    printf '%s' ""
    return 0
  fi
  printf 'cuda:%s' "${csv//,/ ,cuda:}" | tr -d ' '
}

exp_to_plm() {
  local spec="${1:?spec required}"
  local default_problem="${2:?default required}"
  local -n out_problem="$3"
  local -n out_model="$4"
  local -n out_loss="$5"
  local -a parts=()
  IFS='|' read -r -a parts <<< "${spec}"
  if [[ "${#parts[@]}" -eq 3 ]]; then
    out_problem="${parts[0]}"
    out_model="${parts[1]}"
    out_loss="${parts[2]}"
    return 0
  fi
  if [[ "${#parts[@]}" -eq 2 ]]; then
    out_problem="${default_problem}"
    out_model="${parts[0]}"
    out_loss="${parts[1]}"
    return 0
  fi
  return 1
}

format_log_name() {
  local idx="${1:?idx required}"
  local problem="${2:?problem required}"
  local model="${3:?model required}"
  local loss="${4:?loss required}"
  local name="${LOG_NAME_FORMAT}"
  name="${name//\{i\}/${idx}}"
  name="${name//\{problem\}/${problem}}"
  name="${name//\{model\}/${model}}"
  name="${name//\{loss\}/${loss}}"
  name="${name//\{run_id\}/${RUN_ID}}"
  printf '%s' "${name}"
}

prompt_with_default() {
  local prompt="${1:?prompt required}"
  local default_value="${2:-}"
  local input=""
  read -r -p "${prompt} [${default_value}]: " input
  if [[ -z "${input}" ]]; then
    printf '%s' "${default_value}"
  else
    printf '%s' "${input}"
  fi
}

parse_gpu_selection() {
  local selection="${1:?selection required}"
  local token=""
  local start=""
  local end=""
  local -a parts=()
  local -a out=()

  selection="$(echo "${selection}" | tr -d ' ')"
  if [[ -z "${selection}" ]]; then
    echo "Error: empty GPU selection." >&2
    return 1
  fi
  if [[ "${selection,,}" == "all" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "Error: 'all' requires nvidia-smi to list GPUs." >&2
      return 1
    fi
    mapfile -t parts < <(nvidia-smi --list-gpus | sed -n 's/^GPU \([0-9]\+\):.*/\1/p')
    if [[ "${#parts[@]}" -eq 0 ]]; then
      echo "Error: no GPUs reported by nvidia-smi." >&2
      return 1
    fi
    printf '%s' "$(IFS=,; echo "${parts[*]}")"
    return 0
  fi
  if [[ "${selection}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${selection}"
    return 0
  fi
  if [[ "${selection}" =~ ^[0-9]+-[0-9]+$ ]]; then
    start="${selection%-*}"
    end="${selection#*-}"
    if [[ "${start}" -gt "${end}" ]]; then
      echo "Error: invalid GPU range: ${selection}." >&2
      return 1
    fi
    for token in $(seq "${start}" "${end}"); do
      out+=( "${token}" )
    done
    printf '%s' "$(IFS=,; echo "${out[*]}")"
    return 0
  fi
  IFS=',' read -r -a parts <<< "${selection}"
  for token in "${parts[@]}"; do
    if [[ ! "${token}" =~ ^[0-9]+$ ]]; then
      echo "Error: invalid GPU id '${token}'." >&2
      return 1
    fi
    out+=( "${token}" )
  done
  printf '%s' "$(IFS=,; echo "${out[*]}")"
  return 0
}

parse_selection() {
  local selection="${1:?selection required}"
  local max_idx="${2:?max required}"
  local -n out_arr="$3"
  local token=""
  local start=""
  local end=""
  local i=0
  local -a parts=()
  local -a selected=()

  selection="$(echo "${selection}" | tr -d ' ')"
  if [[ -z "${selection}" ]]; then
    echo "Error: empty selection." >&2
    return 1
  fi
  if [[ "${selection,,}" == "all" || "${selection}" == "0" ]]; then
    for i in $(seq 1 "${max_idx}"); do
      selected+=( "${i}" )
    done
    out_arr=( "${selected[@]}" )
    return 0
  fi

  IFS=',' read -r -a parts <<< "${selection}"
  for token in "${parts[@]}"; do
    if [[ -z "${token}" ]]; then
      continue
    fi
    if [[ "${token,,}" == "all" || "${token}" == "0" ]]; then
      for i in $(seq 1 "${max_idx}"); do
        selected+=( "${i}" )
      done
      out_arr=( "${selected[@]}" )
      return 0
    elif [[ "${token}" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${token%-*}"
      end="${token#*-}"
      if [[ "${start}" -le 0 || "${end}" -le 0 || "${start}" -gt "${end}" || "${end}" -gt "${max_idx}" ]]; then
        echo "Error: range out of bounds: ${token} (1-${max_idx})." >&2
        return 1
      fi
      for i in $(seq "${start}" "${end}"); do
        selected+=( "${i}" )
      done
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      if [[ "${token}" -le 0 || "${token}" -gt "${max_idx}" ]]; then
        echo "Error: index out of bounds: ${token} (1-${max_idx})." >&2
        return 1
      fi
      selected+=( "${token}" )
    else
      echo "Error: invalid token '${token}' (use int, a-b, 0, or all)." >&2
      return 1
    fi
  done

  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "Error: no valid indices selected." >&2
    return 1
  fi

  out_arr=( "${selected[@]}" )
  return 0
}

expand_exps_from_lists() {
  local -n probs="$1"
  local -n models="$2"
  local -n losses="$3"
  local p=""
  local m=""
  local l=""
  for p in "${probs[@]}"; do
    for m in "${models[@]}"; do
      for l in "${losses[@]}"; do
        EXPS+=( "${p}|${m}|${l}" )
      done
    done
  done
}

append_extra_override() {
  local override="${1:?override required}"
  if [[ -z "${EXTRA}" ]]; then
    EXTRA="${override}"
  else
    EXTRA="${EXTRA} ${override}"
  fi
}

prompt_index_value() {
  local label="${1:?label required}"
  local default_value="${2:-}"
  local -n options="$3"
  local max="${#options[@]}"
  local input=""
  local idx=""
  while true; do
    read -r -p "${label} index (1-${max}, Enter=${default_value}): " input
    if [[ -z "${input}" ]]; then
      printf '%s' "${default_value}"
      return 0
    fi
    if [[ "${input}" =~ ^[0-9]+$ ]] && [[ "${input}" -ge 1 && "${input}" -le "${max}" ]]; then
      idx="${input}"
      printf '%s' "${options[$((idx-1))]}"
      return 0
    fi
    echo "Error: invalid ${label} index." >&2
  done
}

collect_data_options() {
  local problem="${1:?problem required}"
  local data_dir="${ROOT_DIR}/data/${problem}"
  local filename=""
  if [[ ! -d "${data_dir}" ]]; then
    return 0
  fi
  while IFS= read -r filename; do
    if [[ "${filename}" =~ ^label_prise_seed([0-9]+)_cnt([0-9]+)_s([0-9]+)_n([0-9]+)_scen_(.+)\.pkl$ ]]; then
      SEED_SET["${BASH_REMATCH[1]}"]=1
      INSTANCE_SET["${BASH_REMATCH[2]}"]=1
      SCENARIO_SET["${BASH_REMATCH[3]}"]=1
      NODE_SET["${BASH_REMATCH[4]}"]=1
      DESC_SET["${BASH_REMATCH[5]}"]=1
    fi
  done < <(find "${data_dir}" -maxdepth 1 -type f -name "label_prise_seed*_cnt*_s*_n*_scen_*.pkl" -printf "%f\n")
}

set_to_sorted_list_numeric() {
  local -n set="$1"
  local -n out="$2"
  if [[ "${#set[@]}" -eq 0 ]]; then
    out=()
    return 0
  fi
  mapfile -t out < <(printf '%s\n' "${!set[@]}" | sort -n)
}

set_to_sorted_list_alpha() {
  local -n set="$1"
  local -n out="$2"
  if [[ "${#set[@]}" -eq 0 ]]; then
    out=()
    return 0
  fi
  mapfile -t out < <(printf '%s\n' "${!set[@]}" | sort)
}

summarize_exps() {
  local -a exp_parts=()
  local -A problem_set=()
  local -A model_set=()
  local -A loss_set=()
  local exp=""
  local p=""
  local m=""
  local l=""

  for exp in "${EXPS[@]}"; do
    IFS='|' read -r -a exp_parts <<< "${exp}"
    if [[ "${#exp_parts[@]}" -eq 3 ]]; then
      p="${exp_parts[0]}"
      m="${exp_parts[1]}"
      l="${exp_parts[2]}"
      problem_set["${p}"]=1
    elif [[ "${#exp_parts[@]}" -eq 2 ]]; then
      m="${exp_parts[0]}"
      l="${exp_parts[1]}"
    else
      continue
    fi
    model_set["${m}"]=1
    loss_set["${l}"]=1
  done

  SELECTED_PROBLEMS=()
  SELECTED_MODELS=()
  SELECTED_LOSSES=()
  for p in "${!problem_set[@]}"; do
    SELECTED_PROBLEMS+=( "${p}" )
  done
  for m in "${!model_set[@]}"; do
    SELECTED_MODELS+=( "${m}" )
  done
  for l in "${!loss_set[@]}"; do
    SELECTED_LOSSES+=( "${l}" )
  done
}

# Parse args (simple wrapper; forwards to Python runner)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2"; GPUS_FROM_ARG="1"; shift 2 ;;
    --n_gpus)      N_GPUS="$2"; shift 2 ;;
    --n_nodes)     N_NODES="$2"; shift 2 ;;
    --n_customers) N_CUSTOMERS="$2"; shift 2 ;;
    --n_scenarios) N_SCENARIOS="$2"; shift 2 ;;
    --n_instances) N_INSTANCES="$2"; shift 2 ;;
    --batch)       BATCH="$2"; shift 2 ;;
    --lr)          LR="$2"; shift 2 ;;
    --ddp_trace)   DDP_TRACE="1"; shift 1 ;;
    --extra)       EXTRA="$2"; shift 2 ;;
    --train_config|--config_name) TRAIN_CONFIG="$2"; shift 2 ;;
    --log_dir)     LOG_DIR="$2"; shift 2 ;;
    --dist_name)   DIST_NAME="$2"; shift 2 ;;
    --exp)         EXPS+=( "$2" ); shift 2 ;;
    --dry_run)     DRY_RUN="1"; shift 1 ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${GPUS_FROM_ARG}" = "0" && "${AUTO_GPU}" = "1" ]]; then
  if GPUS="$(detect_free_gpus)"; then
    GPU_AUTO_SELECTED="1"
  else
    echo "No free GPUs detected. Set AUTO_GPU=0 or specify GPUs explicitly." >&2
    exit 1
  fi
fi

if [[ -n "${N_GPUS}" ]]; then
  if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]]; then
    echo "Error: --n_gpus must be a positive integer, got '${N_GPUS}'." >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found; cannot validate N_GPUS=${N_GPUS}." >&2
    exit 1
  fi
  mapfile -t _avail_gpus < <(nvidia-smi --list-gpus | sed -n 's/^GPU \([0-9]\+\):.*/\1/p')
  if [[ "${#_avail_gpus[@]}" -lt "${N_GPUS}" ]]; then
    echo "Error: requested N_GPUS=${N_GPUS}, but only ${#_avail_gpus[@]} GPU(s) available." >&2
    exit 1
  fi
  if ! GPUS="$(parse_gpu_selection "${GPUS}")"; then
    exit 1
  fi
  IFS=',' read -r -a _sel_gpus <<< "${GPUS}"
  if [[ "${#_sel_gpus[@]}" -lt "${N_GPUS}" ]]; then
    echo "Error: N_GPUS=${N_GPUS} requires at least ${N_GPUS} GPU ids; got ${#_sel_gpus[@]} (${GPUS})." >&2
    exit 1
  fi
fi

if [[ "${INTERACTIVE}" = "1" && "${GPUS_FROM_ARG}" = "0" && "${AUTO_GPU}" = "0" ]]; then
  while true; do
    read -r -p "Select CUDA devices (e.g., 0,3). Leave blank for default: " gpu_input
    if [[ -z "${gpu_input}" ]]; then
      break
    fi
    if GPUS="$(parse_gpu_selection "${gpu_input}")"; then
      break
    fi
  done
fi

if [[ "${#EXPS[@]}" -eq 0 && "${INTERACTIVE}" = "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU Status (nvidia-smi) ==="
    nvidia-smi || true
    echo "==============================="
  else
    echo "Note: nvidia-smi not found; skipping GPU status."
  fi

  mapfile -t PROBLEMS < <(list_yaml_basenames "${PROBLEM_DIR}")
  mapfile -t MODELS < <(list_yaml_basenames "${MODEL_DIR}")
  mapfile -t LOSSES < <(list_yaml_basenames "${LOSS_DIR}")
  mapfile -t TRAIN_CONFIGS < <(list_yaml_basenames "${ROOT_DIR}/configs" | sed -n 's/^\(train[^[:space:]]*\)$/\1/p')

  if [[ "${#PROBLEMS[@]}" -eq 0 || "${#MODELS[@]}" -eq 0 || "${#LOSSES[@]}" -eq 0 ]]; then
    echo "Error: could not list configs under ${PROBLEM_DIR}, ${MODEL_DIR}, ${LOSS_DIR}." >&2
    exit 1
  fi

  print_indexed_list "Problems" "${PROBLEMS[@]}"
  print_indexed_list "Models" "${MODELS[@]}"
  print_indexed_list "Losses" "${LOSSES[@]}"
  echo "Enter selections as: problem_sel,model_sel,loss_sel"
  echo "Examples: 1,2,3 | 1,0,3 | 0,0,0 | 2,1-3,0 | all,2,0"
  echo "Use 0 or all for ALL in a category."

  while true; do
    read -r -p "Selection: " selection_line
    selection_line="$(echo "${selection_line}" | tr -d ' ')"
    problem_sel="${selection_line%%,*}"
    rest="${selection_line#*,}"
    model_sel="${rest%,*}"
    loss_sel="${selection_line##*,}"
    if [[ -z "${problem_sel}" || -z "${model_sel}" || -z "${loss_sel}" || "${selection_line}" != *","*","* ]]; then
      echo "Error: expected format '<problem_sel>,<model_sel>,<loss_sel>'." >&2
      continue
    fi

    if ! parse_selection "${problem_sel}" "${#PROBLEMS[@]}" PROBLEM_IDXS; then
      continue
    fi
    if ! parse_selection "${model_sel}" "${#MODELS[@]}" MODEL_IDXS; then
      continue
    fi
    if ! parse_selection "${loss_sel}" "${#LOSSES[@]}" LOSS_IDXS; then
      continue
    fi

    SELECTED_PROBLEMS=()
    SELECTED_MODELS=()
    SELECTED_LOSSES=()
    for idx in "${PROBLEM_IDXS[@]}"; do
      SELECTED_PROBLEMS+=( "${PROBLEMS[$((idx-1))]}" )
    done
    for idx in "${MODEL_IDXS[@]}"; do
      SELECTED_MODELS+=( "${MODELS[$((idx-1))]}" )
    done
    for idx in "${LOSS_IDXS[@]}"; do
      SELECTED_LOSSES+=( "${LOSSES[$((idx-1))]}" )
    done

    expand_exps_from_lists SELECTED_PROBLEMS SELECTED_MODELS SELECTED_LOSSES
    break
  done

  DATA_DESC="${DATA_DESC:-${DIST_NAME}}"
  declare -A SEED_SET=()
  declare -A NODE_SET=()
  declare -A SCENARIO_SET=()
  declare -A INSTANCE_SET=()
  declare -A DESC_SET=()
  for problem in "${SELECTED_PROBLEMS[@]}"; do
    collect_data_options "${problem}"
  done

  set_to_sorted_list_numeric SEED_SET SEED_OPTIONS
  set_to_sorted_list_numeric NODE_SET NODE_OPTIONS
  set_to_sorted_list_numeric SCENARIO_SET SCENARIO_OPTIONS
  set_to_sorted_list_numeric INSTANCE_SET INSTANCE_OPTIONS
  set_to_sorted_list_alpha DESC_SET DESC_OPTIONS

  if [[ "${#SEED_OPTIONS[@]}" -gt 0 ]]; then
    print_indexed_list "Seeds (from data files)" "${SEED_OPTIONS[@]}"
    SEED="$(prompt_index_value "Seed" "${SEED}" SEED_OPTIONS)"
  else
    SEED="$(prompt_with_default "SEED" "${SEED}")"
  fi

  if [[ "${#NODE_OPTIONS[@]}" -gt 0 ]]; then
    print_indexed_list "Nodes (from data files)" "${NODE_OPTIONS[@]}"
    N_NODES="$(prompt_index_value "Nodes" "${N_NODES}" NODE_OPTIONS)"
  else
    N_NODES="$(prompt_with_default "N_NODES" "${N_NODES}")"
  fi
  NEED_CUSTOMERS="0"
  for problem in "${SELECTED_PROBLEMS[@]}"; do
    if [[ "${problem}" == "cflp" || "${problem}" == "cflp_ebp" ]]; then
      NEED_CUSTOMERS="1"
      break
    fi
  done
  if [[ "${NEED_CUSTOMERS}" = "1" ]]; then
    if [[ -z "${N_CUSTOMERS}" ]]; then
      N_CUSTOMERS="${N_NODES}"
    fi
    N_CUSTOMERS="$(prompt_with_default "N_CUSTOMERS" "${N_CUSTOMERS}")"
  else
    N_CUSTOMERS=""
  fi

  if [[ "${#SCENARIO_OPTIONS[@]}" -gt 0 ]]; then
    print_indexed_list "Scenarios (from data files)" "${SCENARIO_OPTIONS[@]}"
    N_SCENARIOS="$(prompt_index_value "Scenarios" "${N_SCENARIOS}" SCENARIO_OPTIONS)"
  else
    N_SCENARIOS="$(prompt_with_default "N_SCENARIOS" "${N_SCENARIOS}")"
  fi

  if [[ "${#INSTANCE_OPTIONS[@]}" -gt 0 ]]; then
    print_indexed_list "Instances (from data files)" "${INSTANCE_OPTIONS[@]}"
    N_INSTANCES="$(prompt_index_value "Instances" "${N_INSTANCES}" INSTANCE_OPTIONS)"
  else
    N_INSTANCES="$(prompt_with_default "N_INSTANCES" "${N_INSTANCES}")"
  fi

  if [[ "${#DESC_OPTIONS[@]}" -gt 0 ]]; then
    print_indexed_list "Data desc (from data files)" "${DESC_OPTIONS[@]}"
    DATA_DESC="$(prompt_index_value "Data desc" "${DATA_DESC}" DESC_OPTIONS)"
  else
    DATA_DESC="$(prompt_with_default "DATA_DESC" "${DATA_DESC}")"
  fi

  if [[ "${#TRAIN_CONFIGS[@]}" -gt 0 ]]; then
    print_indexed_list "Train configs" "${TRAIN_CONFIGS[@]}"
    TRAIN_CONFIG="$(prompt_index_value "Train config" "${TRAIN_CONFIG}" TRAIN_CONFIGS)"
  else
    TRAIN_CONFIG="$(prompt_with_default "TRAIN_CONFIG" "${TRAIN_CONFIG}")"
  fi

  BATCH="$(prompt_with_default "BATCH" "${BATCH}")"
  LR="$(prompt_with_default "LR (blank to keep config)" "${LR}")"
  DIST_NAME="${DATA_DESC}"
  append_extra_override "seed=${SEED}"
fi

if [[ -n "${LR}" ]]; then
  append_extra_override "train.optimizer.lr=${LR}"
fi
if [[ "${DDP_TRACE}" = "1" ]]; then
  append_extra_override "+train.ddp_trace=true"
fi

if [[ "${INTERACTIVE}" = "1" && "${DIST_NAME}" == *forward_trace* ]]; then
  SPLIT_MODE="$(prompt_with_default "DataModule split mode (ratio|absolute)" "ratio")"
  case "${SPLIT_MODE}" in
    ratio)
      append_extra_override "data.split_mode=ratio"
      ;;
    absolute)
      VAL_SPLIT_NUMBER=""
      TEST_ABS_SPLIT_NUMBER=""
      while true; do
        VAL_SPLIT_NUMBER="$(prompt_with_default "Val split number (absolute)" "${VAL_SPLIT_NUMBER:-0}")"
        if [[ "${VAL_SPLIT_NUMBER}" =~ ^[0-9]+$ ]]; then
          break
        fi
        echo "Error: val split number must be a non-negative integer." >&2
      done
      while true; do
        TEST_ABS_SPLIT_NUMBER="$(prompt_with_default "Test split number (absolute)" "${TEST_ABS_SPLIT_NUMBER:-0}")"
        if [[ "${TEST_ABS_SPLIT_NUMBER}" =~ ^[0-9]+$ ]]; then
          break
        fi
        echo "Error: test split number must be a non-negative integer." >&2
      done
      append_extra_override "data.split_mode=absolute"
      append_extra_override "data.val_split_number=${VAL_SPLIT_NUMBER}"
      append_extra_override "data.test_abs_split_number=${TEST_ABS_SPLIT_NUMBER}"
      ;;
    *)
      echo "Error: split mode must be 'ratio' or 'absolute'." >&2
      exit 1
      ;;
  esac
fi

if [[ "${VERBOSE}" = "1" || "${INTERACTIVE}" = "1" ]]; then
  if [[ "${#EXPS[@]}" -gt 0 && "${#SELECTED_MODELS[@]}" -eq 0 && "${#SELECTED_LOSSES[@]}" -eq 0 && "${#SELECTED_PROBLEMS[@]}" -eq 0 ]]; then
    summarize_exps
  fi
  echo "=== Resolved overrides ==="
  if [[ "${#SELECTED_PROBLEMS[@]}" -gt 0 ]]; then
    echo "problem(s)=${SELECTED_PROBLEMS[*]}"
  else
    echo "problem(s)=(default)"
  fi
  echo "n_nodes=${N_NODES}, n_customers=${N_CUSTOMERS}, n_scenarios=${N_SCENARIOS}, n_instances=${N_INSTANCES}"
  echo "scenario_name=${DIST_NAME}, seed=${SEED}"
  if [[ "${#SELECTED_MODELS[@]}" -gt 0 ]]; then
    echo "model(s)=${SELECTED_MODELS[*]}"
  else
    echo "model(s)=(default/exp)"
  fi
  if [[ "${#SELECTED_LOSSES[@]}" -gt 0 ]]; then
    echo "losses=${SELECTED_LOSSES[*]}"
  else
    echo "losses=(default/exp)"
  fi
  if [[ -n "${GPU_AUTO_SELECTED:-}" ]]; then
    if [[ "${FALLBACK_USED}" = "1" ]]; then
      echo "gpu_detect=fallback_table"
    fi
    if [[ -n "${DETECTED_BUSY}" ]]; then
      echo "busy_gpus=${DETECTED_BUSY} (processes=${DETECTED_BUSY_NAMES})"
    else
      echo "busy_gpus=(none)"
    fi
    echo "free_gpus=${GPUS}"
  fi
  echo "train_config=${TRAIN_CONFIG}"
  if [[ "${DDP_TRACE}" = "1" ]]; then
    echo "ddp_trace=true"
  fi
  if [[ -n "${N_GPUS}" ]]; then
    echo "n_gpus_per_job=${N_GPUS}"
  fi
  echo "gpus=${GPUS}, devices=$(format_cuda_devices "${GPUS}"), batch_size=${BATCH}"
  echo "=========================="
fi

if [[ -n "${TRAIN_CONFIG}" && "${TRAIN_CONFIG}" != "train" ]]; then
  append_extra_override "--config-name ${TRAIN_CONFIG}"
fi

if [[ "${CONFIRM}" = "1" ]]; then
  read -r -p "Proceed with execution? [y/N]: " confirm_reply
  if [[ ! "${confirm_reply}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

LOG_DIR_ABS="$(abs_path "${LOG_DIR}")"
LOG_PATTERN="${LOG_NAME_FORMAT}"
TAIL_CMD="tail -n 200 -f \"${LOG_DIR_ABS}\"/*.log"
LOG_FILES=()
LOG_EXPS=()
if [[ "${#EXPS[@]}" -gt 0 ]]; then
  LOG_EXPS=( "${EXPS[@]}" )
else
  if [[ -f "${PY_RUNNER}" ]]; then
    mapfile -t LOG_EXPS < <(python "${PY_RUNNER}" --list_default_exps 2>/dev/null)
  fi
fi
if [[ "${#LOG_EXPS[@]}" -gt 0 ]]; then
  idx=0
  for exp_spec in "${LOG_EXPS[@]}"; do
    if exp_to_plm "${exp_spec}" "${DEFAULT_PROBLEM}" exp_problem exp_model exp_loss; then
      LOG_FILES+=( "${LOG_DIR_ABS}/$(format_log_name "${idx}" "${exp_problem}" "${exp_model}" "${exp_loss}")" )
    fi
    idx=$((idx+1))
  done
fi

echo "===== GPU POOL LAUNCH ====="
echo "LOG_DIR: ${LOG_DIR_ABS}"
if [[ "${#LOG_FILES[@]}" -gt 0 ]]; then
  echo "Logs:    ${LOG_FILES[*]}"
else
  echo "Logs:    ${LOG_PATTERN}"
fi
echo "Watch:   ${TAIL_CMD}"
echo "==========================="

if [[ "${DRY_RUN}" = "1" ]]; then
  if [[ "${#EXPS[@]}" -eq 0 ]]; then
    echo "Selected problems/models/losses: default runner behavior (no --exp provided)"
  else
    if [[ "${#SELECTED_PROBLEMS[@]}" -eq 0 ]]; then
      summarize_exps
    fi
    if [[ "${#SELECTED_PROBLEMS[@]}" -gt 0 ]]; then
      echo "Selected problems: ${SELECTED_PROBLEMS[*]}"
    else
      echo "Selected problems: (default)"
    fi
    if [[ "${#SELECTED_MODELS[@]}" -gt 0 ]]; then
      echo "Selected models: ${SELECTED_MODELS[*]}"
    fi
    if [[ "${#SELECTED_LOSSES[@]}" -gt 0 ]]; then
      echo "Selected losses: ${SELECTED_LOSSES[*]}"
    fi
    echo "Expanded experiments:"
    for e in "${EXPS[@]}"; do
      echo "  ${e}"
    done
  fi

  DRY_CMD=( python "${PY_RUNNER}"
    --gpus "${GPUS}"
    --n_nodes "${N_NODES}"
    --n_scenarios "${N_SCENARIOS}"
    --n_instances "${N_INSTANCES}"
    --batch "${BATCH}"
    --train_config "${TRAIN_CONFIG}"
    --log_dir "${LOG_DIR}"
    --dist_name "${DIST_NAME}"
    --log_name_format "${LOG_NAME_FORMAT}"
    --run_id "${RUN_ID}"
  )
  if [[ -n "${N_GPUS}" ]]; then
    DRY_CMD+=( --n_gpus "${N_GPUS}" )
  fi
  if [[ -n "${N_CUSTOMERS}" ]]; then
    DRY_CMD+=( --n_customers "${N_CUSTOMERS}" )
  fi
  if [[ -n "${EXTRA}" ]]; then
    DRY_CMD+=( --extra "${EXTRA}" )
  fi
  if [[ "${#EXPS[@]}" -gt 0 ]]; then
    for e in "${EXPS[@]}"; do
      DRY_CMD+=( --exp "${e}" )
    done
  fi
  echo "Dry-run command:"
  if [[ -n "${GPU_AUTO_SELECTED:-}" ]]; then
    echo "Detected busy GPUs: ${DETECTED_BUSY:-none} (processes=${DETECTED_BUSY_NAMES:-none})"
    echo "Selected free GPUs: ${GPUS}"
  fi
  echo "Resolved GPUs: ${GPUS} ($(format_cuda_devices "${GPUS}"))"
  printf '  %q' "${DRY_CMD[@]}"
  echo
  exit 0
fi

CMD=( python "${PY_RUNNER}"
  --gpus "${GPUS}"
  --n_nodes "${N_NODES}"
  --n_scenarios "${N_SCENARIOS}"
  --n_instances "${N_INSTANCES}"
  --batch "${BATCH}"
  --train_config "${TRAIN_CONFIG}"
  --log_dir "${LOG_DIR}"
  --dist_name "${DIST_NAME}"
  --log_name_format "${LOG_NAME_FORMAT}"
  --run_id "${RUN_ID}"
)
if [[ -n "${N_GPUS}" ]]; then
  CMD+=( --n_gpus "${N_GPUS}" )
fi
if [[ -n "${N_CUSTOMERS}" ]]; then
  CMD+=( --n_customers "${N_CUSTOMERS}" )
fi

if [[ -n "${EXTRA}" ]]; then
  CMD+=( --extra "${EXTRA}" )
fi

# Forward any explicitly provided experiments (repeatable --exp)
if [[ "${#EXPS[@]}" -gt 0 ]]; then
  for e in "${EXPS[@]}"; do
    CMD+=( --exp "${e}" )
  done
fi

echo "Running GPU-pool scheduler:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
