#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/script_lib.sh"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROBLEM_DIR="${ROOT_DIR}/configs/problem"
MODEL_DIR="${ROOT_DIR}/configs/model"
LOSS_DIR="${ROOT_DIR}/configs/losses"


export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Alt eval data (when USE_ALT_DATA=1):
#   Set via env vars: EVAL_N_NODES, EVAL_N_SCENARIOS, EVAL_N_INSTANCES,
#   optionally EVAL_P_EDGE, EVAL_SCENARIO_NAME, EVAL_DATA_SEED.
# Eval randomness:
#   Optionally set EVAL_SEED to control eval RNG while keeping data seed unchanged.

INTERACTIVE="${INTERACTIVE:-${INT:-0}}"
CONFIRM="${CONFIRM:-0}"
VERBOSE="${VERBOSE:-0}"
DATA_NORMALIZE="${DATA_NORMALIZE:-}"
DATA_NORMALIZE_MODE="${DATA_NORMALIZE_MODE:-}"
VERIFY_FULL_OBJ_MAX_INSTANCES="${VERIFY_FULL_OBJ_MAX_INSTANCES:-0}"
SEQUENTIAL="${SEQUENTIAL:-${SEQ:-0}}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/04_05_eval_ml_2ro_e2e.sh [PROBLEM] [TRAIN_N_NODES] [TRAIN_N_SCENARIOS] [TRAIN_N_INSTANCES] [MODEL] [LOSS]
                                   [TRAIN_CONFIG_NAME] [DEVICE] [BATCH_SIZE] [THREADS] [MILP_TIME_LIMIT] [DIST_NAME]

Defaults:
  TRAIN_CONFIG_NAME=train_scale_norm

Examples:
  ./scripts/04_05_eval_ml_2ro_e2e.sh
  ./scripts/04_05_eval_ml_2ro_e2e.sh selection 50 50 500 one_shot_crossattn_edge_dim1_small score_weighted_loss_flat train_scale_norm cuda 25 40 900 uniform
  SEQ=1 INT=1 ./scripts/04_05_eval_ml_2ro_e2e.sh
  SEQ=1 ./scripts/04_05_eval_ml_2ro_e2e.sh cflp_ebp 30 50 2500 config_early_fusion_mlp score_weighted_loss_flat train_scale_norm cuda:1 16 40 300 uniform
  DATA_NORMALIZE_MODE=scale ./scripts/04_05_eval_ml_2ro_e2e.sh vertex_cover 20 50 2500 one_shot_crossattn_edge_dim1_small score_weighted_loss_flat train_scale_norm cuda 32 40 900 uniform
  EVAL_SEED=1 DRY_RUN=1 K_LIST=0.1 ./scripts/04_05_eval_ml_2ro_e2e.sh vertex_cover 20 50 2500 one_shot_crossattn_edge_dim1_small score_weighted_loss_flat train_scale_norm cpu 32 40 900 uniform

Eval seed env vars:
  EVAL_SEED       Optional eval RNG seed; adds "_e<seed>" in eval.name (e.g. ML_s42_e1_...)
  EVAL_DATA_SEED  Optional eval data seed override (used only when USE_ALT_DATA=1)
  DRY_RUN         1 to print commands without executing python
  EXTRA_HYDRA     Optional extra Hydra overrides, appended verbatim (space split),
                  for problem params not encoded in the instance filename
                  (e.g. EXTRA_HYDRA="problem.instance.service_radius=0.215")
EOF
}

die() { echo "Error: $*" >&2; exit 2; }

# Problems whose shape is (n_facilities, n_customers) instead of n_nodes.
_is_cflp_family() {
  case "${1}" in
    cflp|cflp_ebp|cflp_ebp_mm|cflp_ebp_normal) return 0 ;;
    *) return 1 ;;
  esac
}

PROBLEM="${1:-vertex_cover}"

# --- TRAIN (checkpoint identity) ---
TRAIN_N_NODES="${2:-8}"
TRAIN_N_CUSTOMERS="${TRAIN_N_CUSTOMERS:-${TRAIN_N_NODES}}"
TRAIN_N_SCENARIOS="${3:-10}"
TRAIN_N_INSTANCES="${4:-500}"

MODEL="${5:-one_shot_crossattn_edge_dim1_small}"
LOSS="${6:-score_weighted_loss_flat}"
TRAIN_CONFIG_NAME="${7:-train_scale_norm}"

DEVICE="${8:-cuda}"
BATCH_SIZE="${9:-32}"
THREADS="${10:-40}"
MILP_TIME_LIMIT="${11:-900}"
DIST_NAME="${12:-uniform}" # this can control loading the label prise filename

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# --- EVAL (dataset identity; defaults to TRAIN unless overridden) ---
USE_ALT_DATA="${USE_ALT_DATA:-0}"             # 1 => use eval.data_override.*
OUTPUT_DIR_BY_EVAL="${OUTPUT_DIR_BY_EVAL:-0}" # 1 => write eval outputs under eval-data scale path
EVAL_SPLIT_OVERRIDE="${EVAL_SPLIT_OVERRIDE:-}" # optional explicit split: train|val|test|all|val_test
EVAL_SEED="${EVAL_SEED:-}"                    # optional eval RNG seed (does not change instance seed)

if [[ "${INTERACTIVE}" = "1" ]]; then
  echo "=== Interactive ML→2RO setup ==="
  PROBLEM="$(prompt_with_default "PROBLEM" "${PROBLEM}")"
  TRAIN_N_NODES="$(prompt_with_default "TRAIN_N_NODES" "${TRAIN_N_NODES}")"
  if _is_cflp_family "${PROBLEM}"; then
    TRAIN_N_CUSTOMERS="$(prompt_with_default "TRAIN_N_CUSTOMERS" "${TRAIN_N_CUSTOMERS}")"
  fi
  TRAIN_N_SCENARIOS="$(prompt_with_default "TRAIN_N_SCENARIOS" "${TRAIN_N_SCENARIOS}")"
  TRAIN_N_INSTANCES="$(prompt_with_default "TRAIN_N_INSTANCES" "${TRAIN_N_INSTANCES}")"
  if [[ "${SEQUENTIAL}" != "1" ]]; then
    MODEL="$(prompt_with_default "MODEL" "${MODEL}")"
    LOSS="$(prompt_with_default "LOSS" "${LOSS}")"
  fi
  TRAIN_CONFIG_NAME="$(prompt_with_default "TRAIN_CONFIG_NAME" "${TRAIN_CONFIG_NAME}")"
  BATCH_SIZE="$(prompt_with_default "BATCH_SIZE" "${BATCH_SIZE}")"
  THREADS="$(prompt_with_default "THREADS" "${THREADS}")"
  MILP_TIME_LIMIT="$(prompt_with_default "MILP_TIME_LIMIT" "${MILP_TIME_LIMIT}")"
  DIST_NAME="$(prompt_with_default "DIST_NAME" "${DIST_NAME}")"
  EVAL_SEED="$(prompt_with_default "EVAL_SEED (optional eval RNG seed)" "${EVAL_SEED}")"
  USE_ALT_DATA="$(prompt_with_default "USE_ALT_DATA (0/1)" "${USE_ALT_DATA}")"
  VERIFY_FULL_OBJ_MAX_INSTANCES="$(prompt_with_default "VERIFY_FULL_OBJ_MAX_INSTANCES (0=off)" "${VERIFY_FULL_OBJ_MAX_INSTANCES}")"
fi

if [[ -z "${DATA_NORMALIZE}" && "${TRAIN_CONFIG_NAME}" == "train_normalize" ]]; then
  DATA_NORMALIZE="1"
fi
if [[ -z "${DATA_NORMALIZE}" && "${TRAIN_CONFIG_NAME}" == "train_scale_norm" ]]; then
  DATA_NORMALIZE="1"
fi
if [[ -z "${DATA_NORMALIZE_MODE}" && "${TRAIN_CONFIG_NAME}" == "train_scale_norm" ]]; then
  DATA_NORMALIZE_MODE="scale"
fi

if [[ "${DEVICE}" =~ ^cuda($|:|[[:space:]]) ]]; then
  if [[ "$(type -t detect_free_gpus)" == "function" ]]; then
    FREE_GPUS="$(detect_free_gpus)"
    FIRST_GPU="${FREE_GPUS%%,*}"
    if [[ -n "${FIRST_GPU}" ]]; then
      DEVICE="cuda:${FIRST_GPU}"
    fi
  fi
fi

EVAL_N_NODES="${EVAL_N_NODES:-${TRAIN_N_NODES}}"
EVAL_N_CUSTOMERS_SET="${EVAL_N_CUSTOMERS+x}"
EVAL_N_CUSTOMERS="${EVAL_N_CUSTOMERS:-${TRAIN_N_CUSTOMERS}}"
EVAL_N_SCENARIOS="${EVAL_N_SCENARIOS:-${TRAIN_N_SCENARIOS}}"
EVAL_N_INSTANCES="${EVAL_N_INSTANCES:-${TRAIN_N_INSTANCES}}"
EVAL_P_EDGE="${EVAL_P_EDGE:-}"                # optional
EVAL_SCENARIO_NAME="${EVAL_SCENARIO_NAME:-}"  # optional
EVAL_DATA_SEED="${EVAL_DATA_SEED:-}"          # optional alt-data seed override

if [[ "${INTERACTIVE}" = "1" && "${USE_ALT_DATA}" = "1" ]]; then
  echo "--- Alt eval data override (USE_ALT_DATA=1) ---"
  EVAL_N_NODES="$(prompt_with_default "EVAL_N_NODES" "${EVAL_N_NODES}")"
  if _is_cflp_family "${PROBLEM}"; then
    if [[ -z "${EVAL_N_CUSTOMERS_SET}" ]]; then
      EVAL_N_CUSTOMERS="${EVAL_N_NODES}"
    fi
    EVAL_N_CUSTOMERS="$(prompt_with_default "EVAL_N_CUSTOMERS" "${EVAL_N_CUSTOMERS}")"
  fi
  EVAL_N_SCENARIOS="$(prompt_with_default "EVAL_N_SCENARIOS" "${EVAL_N_SCENARIOS}")"
  EVAL_N_INSTANCES="$(prompt_with_default "EVAL_N_INSTANCES" "${EVAL_N_INSTANCES}")"
  EVAL_P_EDGE="$(prompt_with_default "EVAL_P_EDGE (optional)" "${EVAL_P_EDGE}")"
  EVAL_SCENARIO_NAME="$(prompt_with_default "EVAL_SCENARIO_NAME (optional)" "${EVAL_SCENARIO_NAME}")"
  EVAL_DATA_SEED="$(prompt_with_default "EVAL_DATA_SEED (optional)" "${EVAL_DATA_SEED}")"
fi

if [[ "${USE_ALT_DATA}" = "1" ]] && _is_cflp_family "${PROBLEM}"; then
  if [[ -z "${EVAL_N_CUSTOMERS_SET}" ]]; then
    EVAL_N_CUSTOMERS="${EVAL_N_NODES}"
  fi
fi


EVAL_SPLIT="test"
HEURISTICS="ML"

if [[ -n "${EVAL_SPLIT_OVERRIDE}" ]]; then
  case "${EVAL_SPLIT_OVERRIDE}" in
    train|val|test|all|val_test)
      EVAL_SPLIT="${EVAL_SPLIT_OVERRIDE}"
      ;;
    *)
      die "Invalid EVAL_SPLIT_OVERRIDE='${EVAL_SPLIT_OVERRIDE}' (expected train|val|test|all|val_test)"
      ;;
  esac
else
  if [[ "${DIST_NAME}" == *bigmilp* || "${EVAL_LABEL_OUTPUT:-}" == "bigmilp" ]]; then
    EVAL_SPLIT="all"
  fi
fi

if [[ "${CONFIRM}" = "1" ]]; then
  read -r -p "Proceed with execution? [y/N]: " confirm_reply
  if [[ ! "${confirm_reply}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

EXPERIMENTS=()
if [[ "${SEQUENTIAL}" == "1" ]]; then
  selection="${SELECTION:-}"
  if [[ -z "${selection}" && "${INTERACTIVE}" != "1" && "${#}" -ge 6 ]]; then
    EXPERIMENTS+=( "${PROBLEM}|${MODEL}|${LOSS}" )
  else
    mapfile -t PROBLEM_LIST < <(list_yaml_basenames "${PROBLEM_DIR}")
    mapfile -t MODEL_LIST < <(list_yaml_basenames "${MODEL_DIR}")
    mapfile -t LOSS_LIST < <(list_yaml_basenames "${LOSS_DIR}")

    if [[ "${INTERACTIVE}" = "1" ]]; then
      print_indexed_list "Models" "${MODEL_LIST[@]}"
      print_indexed_list "Losses" "${LOSS_LIST[@]}"
      echo "Enter selections as: model_sel,loss_sel"
      echo "Examples: 1,2 | 0,3 | 2,1-3 | all,2"
      echo "Use 0 or all for ALL in a category."
      read -r -p "Selection: " selection
    fi

    IFS=',' read -r sel_model sel_loss <<< "${selection}"
    if [[ -z "${sel_model:-}" || -z "${sel_loss:-}" ]]; then
      die "expected format '<model_sel>,<loss_sel>'."
    fi

    parse_selection "${sel_model}" "${#MODEL_LIST[@]}" MODEL_IDX
    parse_selection "${sel_loss}" "${#LOSS_LIST[@]}" LOSS_IDX

    SELECTED_MODELS=()
    SELECTED_LOSSES=()
    for i in "${MODEL_IDX[@]}"; do
      SELECTED_MODELS+=( "${MODEL_LIST[$((i-1))]}" )
    done
    for i in "${LOSS_IDX[@]}"; do
      SELECTED_LOSSES+=( "${LOSS_LIST[$((i-1))]}" )
    done

    SELECTED_PROBLEMS=( "${PROBLEM}" )
    expand_exps_from_lists SELECTED_PROBLEMS SELECTED_MODELS SELECTED_LOSSES
  fi
else
  EXPERIMENTS+=( "${PROBLEM}|${MODEL}|${LOSS}" )
fi

run_one() {
  local problem="${1:?problem required}"
  local model="${2:?model required}"
  local loss="${3:?loss required}"
  local ckpt_path=""
  local name=""
  local output_dir=""
  local raw_list=""
  local k_list=""
  local eval_seed_suffix=""
  local data_seed_for_file=""

  eval_seed_suffix="$(get_eval_seed_suffix "${EVAL_SEED}")"
  if [[ -n "${EVAL_DATA_SEED}" ]]; then
    data_seed_for_file="${EVAL_DATA_SEED}"
  else
    data_seed_for_file="42"
  fi

  if _is_cflp_family "${problem}"; then
    ckpt_path="model/${problem}/s42_i${TRAIN_N_INSTANCES}_m${TRAIN_N_NODES}_n${TRAIN_N_CUSTOMERS}_s${TRAIN_N_SCENARIOS}__c-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_best.pt"
  else
    ckpt_path="model/${problem}/s42_i${TRAIN_N_INSTANCES}_n${TRAIN_N_NODES}_s${TRAIN_N_SCENARIOS}__c-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_best.pt"
  fi

  if [[ "${USE_ALT_DATA}" = "1" ]]; then
    EVAL_N_SCENARIOS="${EVAL_N_SCENARIOS:-${TRAIN_N_SCENARIOS}}"
    EVAL_N_INSTANCES="${EVAL_N_INSTANCES:-${TRAIN_N_INSTANCES}}"
    if _is_cflp_family "${problem}"; then
      name="${HEURISTICS}_s42${eval_seed_suffix}_tr-i${TRAIN_N_INSTANCES}_m${TRAIN_N_NODES}_n${TRAIN_N_CUSTOMERS}_s${TRAIN_N_SCENARIOS}_ev-i${EVAL_N_INSTANCES}_m${EVAL_N_NODES}_n${EVAL_N_CUSTOMERS}_s${EVAL_N_SCENARIOS}_t-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_${EVAL_SPLIT}"
    else
      name="${HEURISTICS}_s42${eval_seed_suffix}_tr-i${TRAIN_N_INSTANCES}_n${TRAIN_N_NODES}_s${TRAIN_N_SCENARIOS}_ev-i${EVAL_N_INSTANCES}_n${EVAL_N_NODES}_s${EVAL_N_SCENARIOS}_t-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_${EVAL_SPLIT}"
    fi
  else
    if _is_cflp_family "${problem}"; then
      name="${HEURISTICS}_s42${eval_seed_suffix}_i${TRAIN_N_INSTANCES}_m${TRAIN_N_NODES}_n${TRAIN_N_CUSTOMERS}_s${TRAIN_N_SCENARIOS}_t-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_${EVAL_SPLIT}"
    else
      name="${HEURISTICS}_s42${eval_seed_suffix}_i${TRAIN_N_INSTANCES}_n${TRAIN_N_NODES}_s${TRAIN_N_SCENARIOS}_t-${TRAIN_CONFIG_NAME}_m-${model}_l-${loss}_${EVAL_SPLIT}"
    fi
  fi

  if [[ "${OUTPUT_DIR_BY_EVAL}" == "1" && "${USE_ALT_DATA}" == "1" ]]; then
    if _is_cflp_family "${problem}"; then
      output_dir="eval/${problem}/s42_i${EVAL_N_INSTANCES}_m${EVAL_N_NODES}_n${EVAL_N_CUSTOMERS}_s${EVAL_N_SCENARIOS}"
    else
      output_dir="eval/${problem}/s42_i${EVAL_N_INSTANCES}_n${EVAL_N_NODES}_s${EVAL_N_SCENARIOS}"
    fi
  else
    if _is_cflp_family "${problem}"; then
      output_dir="eval/${problem}/s42_i${TRAIN_N_INSTANCES}_m${TRAIN_N_NODES}_n${TRAIN_N_CUSTOMERS}_s${TRAIN_N_SCENARIOS}"
    else
      output_dir="eval/${problem}/s42_i${TRAIN_N_INSTANCES}_n${TRAIN_N_NODES}_s${TRAIN_N_SCENARIOS}"
    fi
  fi
  raw_list="$(get_k_list_raw "${EVAL_N_SCENARIOS}")"
  k_list="${raw_list//,/ }"

  if [[ "${VERBOSE}" = "1" || "${INTERACTIVE}" = "1" ]]; then
    DATA_FILE="$(resolve_data_file "${problem}" "${EVAL_N_INSTANCES}" "${EVAL_N_NODES}" "${EVAL_N_SCENARIOS}" "${DIST_NAME}" "${data_seed_for_file}" "${EVAL_P_EDGE}" "${EVAL_N_CUSTOMERS}")"
    echo "=== E2E ML→2RO CONFIG ==="
    echo "PROBLEM=${problem}"
    echo "CKPT (TRAIN): n_nodes=${TRAIN_N_NODES}, n_customers=${TRAIN_N_CUSTOMERS}, n_scenarios=${TRAIN_N_SCENARIOS}, instance_cnt=${TRAIN_N_INSTANCES}"
    echo "CKPT_PATH=${ckpt_path}"
    if [[ "${USE_ALT_DATA}" = "1" ]]; then
      echo "EVAL DATA OVERRIDE: ENABLED"
      echo "EVAL: n_nodes=${EVAL_N_NODES}, n_customers=${EVAL_N_CUSTOMERS}, n_scenarios=${EVAL_N_SCENARIOS}, instance_cnt=${EVAL_N_INSTANCES}, p_edge=${EVAL_P_EDGE:-null}, scenario_name=${EVAL_SCENARIO_NAME:-null}, data_seed=${data_seed_for_file}"
    else
      echo "EVAL DATA OVERRIDE: DISABLED (eval data == train spec via cfg defaults)"
      echo "EVAL: n_nodes=${TRAIN_N_NODES}, n_customers=${TRAIN_N_CUSTOMERS}, n_scenarios=${TRAIN_N_SCENARIOS}, instance_cnt=${TRAIN_N_INSTANCES}"
    fi
    echo "EVAL_SEED=${EVAL_SEED:-default_cfg_seed}"
    echo "MODEL=${model}, LOSS=${loss}, TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
    echo "DEVICE=${DEVICE}, BATCH_SIZE=${BATCH_SIZE}, THREADS=${THREADS}, SPLIT=${EVAL_SPLIT}"
    echo "DIST_NAME=${DIST_NAME}, USE_ALT_DATA=${USE_ALT_DATA}"
    echo "EVAL_SPLIT_OVERRIDE=${EVAL_SPLIT_OVERRIDE:-<unset>}"
    echo "OUTPUT_DIR_BY_EVAL=${OUTPUT_DIR_BY_EVAL}"
    echo "OUTPUT_DIR=${output_dir}"
    echo "K_LIST=${k_list}"
    echo "DATA_FILE=${DATA_FILE}"
    echo "NAME=${name}"
    echo "========================="
  fi

  for k in ${k_list}; do
    echo "=== E2E ML→2RO: ${problem}, k_ratio=${k} ==="

    if [[ "${USE_ALT_DATA}" = "1" ]]; then
      USE_ALT_DATA_FLAG=true
    else
      USE_ALT_DATA_FLAG=false
    fi

    CMD=(python scripts/04_05_eval_ml_2ro_e2e.py
      problem="${problem}"
      eval.output_dir="${output_dir}"
      model="${model}"
      eval.load_ckpt=true
      eval.ckpt_path="${ckpt_path}"
      eval.split="${EVAL_SPLIT}"
      eval.name="${name}"
      eval.device="${DEVICE}"
      data.batch_size="${BATCH_SIZE}"
      eval.k_ratio="${k}"
      solver.threads="${THREADS}"
      eval.use_alt_data="${USE_ALT_DATA_FLAG}"
    )
    if [[ -n "${EVAL_SEED}" ]]; then
      CMD+=( eval.seed="${EVAL_SEED}" )
    fi
    if [[ "${USE_ALT_DATA}" = "1" ]]; then
      CMD+=(
        data.instance_cnt="${EVAL_N_INSTANCES}"
        problem.instance.n_scenarios="${EVAL_N_SCENARIOS}"
      )
      if [[ -n "${EVAL_SCENARIO_NAME}" ]]; then
        CMD+=( problem.instance.scenario_name="${EVAL_SCENARIO_NAME}" )
      else
        CMD+=( problem.instance.scenario_name="${DIST_NAME}" )
      fi
      if _is_cflp_family "${problem}"; then
        CMD+=( problem.instance.n_facilities="${EVAL_N_NODES}" problem.instance.n_customers="${EVAL_N_CUSTOMERS}" )
      else
        CMD+=( problem.instance.n_nodes="${EVAL_N_NODES}" )
      fi
    else
      CMD+=(
        data.instance_cnt="${TRAIN_N_INSTANCES}"
        problem.instance.n_scenarios="${TRAIN_N_SCENARIOS}"
        problem.instance.scenario_name="${DIST_NAME}"
      )
      if _is_cflp_family "${problem}"; then
        CMD+=( problem.instance.n_facilities="${TRAIN_N_NODES}" problem.instance.n_customers="${TRAIN_N_CUSTOMERS}" )
      else
        CMD+=( problem.instance.n_nodes="${TRAIN_N_NODES}" )
      fi
    fi
    CMD+=(
      solver.time_limit="${MILP_TIME_LIMIT}"
      )
    if [[ "${VERIFY_FULL_OBJ_MAX_INSTANCES}" != "0" ]]; then
      CMD+=( +eval.verify_full_obj.max_instances="${VERIFY_FULL_OBJ_MAX_INSTANCES}" )
    fi
    if [[ "${DATA_NORMALIZE}" == "1" ]]; then
      CMD+=( data.normalize=true )
    elif [[ "${DATA_NORMALIZE}" == "0" ]]; then
      CMD+=( data.normalize=false )
    fi
    if [[ -n "${DATA_NORMALIZE_MODE}" ]]; then
      CMD+=( +data.normalize_mode="${DATA_NORMALIZE_MODE}" )
    fi
    if [[ -n "${EVAL_LABEL_OUTPUT:-}" ]]; then
      CMD+=( label.prise.output="${EVAL_LABEL_OUTPUT}" )
    fi
    if [[ "${USE_ALT_DATA}" = "1" ]]; then
      CMD+=(
        eval.data_override.instance_cnt="${EVAL_N_INSTANCES}"
        eval.data_override.n_scenarios="${EVAL_N_SCENARIOS}"
      )
      if _is_cflp_family "${problem}"; then
        CMD+=( eval.data_override.n_facilities="${EVAL_N_NODES}" eval.data_override.n_customers="${EVAL_N_CUSTOMERS}" )
      else
        CMD+=( eval.data_override.n_nodes="${EVAL_N_NODES}" )
      fi
      if [[ -n "${EVAL_P_EDGE}" ]]; then
        CMD+=( eval.data_override.p_edge="${EVAL_P_EDGE}" )
      fi
      if [[ -n "${EVAL_SCENARIO_NAME}" ]]; then
        CMD+=( eval.data_override.scenario_name="${EVAL_SCENARIO_NAME}" )
      fi
      if [[ -n "${EVAL_DATA_SEED}" ]]; then
        CMD+=( eval.data_override.seed="${EVAL_DATA_SEED}" )
      fi
    fi
    # Extra Hydra overrides for problem parameters that are not in the instance
    # filename. EXTRA_HYDRA is appended verbatim, space split.
    if [[ -n "${EXTRA_HYDRA:-}" ]]; then
      # shellcheck disable=SC2206
      CMD+=( ${EXTRA_HYDRA} )
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "DRY_RUN: ${CMD[*]}"
    else
      echo "Running: ${CMD[*]}"
      "${CMD[@]}"
    fi
  done
}

for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r exp_problem exp_model exp_loss <<< "${exp}"
  run_one "${exp_problem}" "${exp_model}" "${exp_loss}"
done
