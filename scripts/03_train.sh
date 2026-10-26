#!/bin/bash
set -e
# shellcheck source=/dev/null
source "$(dirname -- "${BASH_SOURCE[0]}")/script_lib.sh"

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#ulimit -n 65536
usage() {
  cat <<'EOF'
Usage:
  scripts/03_train.sh PROBLEM [args...]

Examples:
  scripts/03_train.sh vertex_cover 20 50 2500 model=one_shot_crossattn_edge_dim1_small losses=score_weighted_loss_flat cuda:0 32
  DDP=1 NPROC_PER_NODE=4 scripts/03_train.sh vertex_cover 20 50 2500 model=... losses=... cuda 32
  scripts/03_train.sh cflp 30 30 200 2500 model=... losses=... cuda:0 16
  scripts/03_train.sh cflp_ebp 30 30 200 2500 model=... losses=... cuda:0 16
  scripts/03_train.sh vertex_cover 20 50 2500 model=... losses=... cuda:0 32 uniform --config-name train_normalize

Positional args (vertex_cover/selection):
  1  PROBLEM
  2  N_NODES
  3  N_SCENARIOS
  4  N_INSTANCES
  5  MODEL_OVERRIDE (optional)
  6  LOSSES_OVERRIDE (optional)
  7  DEVICE
  8  BATCH_SIZE (global batch size; will be split across DDP ranks)
  9  DIST_NAME
  10+ EXTRA Hydra overrides

Positional args (cflp variants):
  1  PROBLEM
  2  N_FACILITIES
  3  N_CUSTOMERS
  4  N_SCENARIOS
  5  N_INSTANCES
  6  MODEL_OVERRIDE (optional)
  7  LOSSES_OVERRIDE (optional)
  8  DEVICE
  9  BATCH_SIZE (global batch size; will be split across DDP ranks)
  10 DIST_NAME
  11+ EXTRA Hydra overrides

DDP env vars:
  DDP=1              Enable torchrun
  NPROC_PER_NODE=4   Number of processes per node (defaults to GPU count)
  MASTER_PORT=29500  Torchrun master port
  DRY_RUN=1          Print the resolved command and exit
  TRAIN_CONFIG_NAME  Optional Hydra config name (default: train_scale_norm)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
# 1–3: keep existing defaults / calling style
PROBLEM="${1:-vertex_cover}"
if [[ "${PROBLEM}" == cflp* ]]; then
  N_FACILITIES="${2:-30}"
  N_CUSTOMERS="${3:-30}"
  N_SCENARIOS="${4:-10}"
  N_INSTANCES="${5:-100}"
else
  N_NODES="${2:-8}"
  N_SCENARIOS="${3:-10}"
  N_INSTANCES="${4:-100}"
fi

# 4–5: optional model & losses overrides (can be config group or @file:)
# e.g. "model=simple" or "model=@file:../configs/model/simple.yaml"
if [[ "${PROBLEM}" == cflp* ]]; then
  MODEL_OVERRIDE="${6:-}"
  LOSSES_OVERRIDE="${7:-}"
  DEVICE="${8:-cuda}"
  BATCH_SIZE="${9:-24}"
  DIST_NAME="${10:-uniform}" # this can control loading the label prise filename
  EXTRA_SHIFT=10
else
  MODEL_OVERRIDE="${5:-}"
  LOSSES_OVERRIDE="${6:-}"
  DEVICE="${7:-cuda}"
  BATCH_SIZE="${8:-32}"
  DIST_NAME="${9:-uniform}" # this can control loading the label prise filename
  EXTRA_SHIFT=9
fi

DDP="${DDP:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-train_scale_norm}"

# 6+: any extra Hydra overrides (seed, epochs, etc.)
if [ "$#" -gt "${EXTRA_SHIFT}" ]; then
  shift "${EXTRA_SHIFT}"
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

CONFIG_NAME_ARG="${TRAIN_CONFIG_NAME:-}"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  new_extra=()
  i=0
  while [[ "${i}" -lt "${#EXTRA_ARGS[@]}" ]]; do
    arg="${EXTRA_ARGS[$i]}"
    if [[ "${arg}" == "--config-name" || "${arg}" == "--config_name" ]]; then
      next_index=$((i+1))
      if [[ "${next_index}" -lt "${#EXTRA_ARGS[@]}" ]]; then
        CONFIG_NAME_ARG="${EXTRA_ARGS[$next_index]}"
        i=$((i+2))
        continue
      fi
    fi
    new_extra+=( "${arg}" )
    i=$((i+1))
  done
  EXTRA_ARGS=( "${new_extra[@]}" )
fi

if [[ "${PROBLEM}" == cflp* ]]; then
  echo "start to run train ${PROBLEM} ${N_FACILITIES} ${N_CUSTOMERS} ${N_SCENARIOS} ${N_INSTANCES} ${MODEL_OVERRIDE} batch_size ${BATCH_SIZE}..."
else
  echo "start to run train ${PROBLEM} ${N_NODES} ${N_SCENARIOS} ${N_INSTANCES} ${MODEL_OVERRIDE} batch_size ${BATCH_SIZE}..."
fi

echo "DDP=${DDP} NPROC_PER_NODE=${NPROC_PER_NODE:-auto} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

TRAIN_SCRIPT="scripts/03_train.py"
CMD=(
  "${TRAIN_SCRIPT}"
)
if [[ -n "${CONFIG_NAME_ARG}" ]]; then
  CMD+=( "--config-name" "${CONFIG_NAME_ARG}" )
fi
CMD+=(
  problem="${PROBLEM}"
  data.instance_cnt="${N_INSTANCES}"
  problem.instance.n_scenarios="${N_SCENARIOS}"
  problem.instance.scenario_name="${DIST_NAME}"
  train.device="${DEVICE}"
  train.model_output_dir="model/${PROBLEM}"
  data.batch_size="${BATCH_SIZE}"
)

if [[ "${PROBLEM}" == cflp* ]]; then
  CMD+=(
    problem.instance.n_facilities="${N_FACILITIES}"
    problem.instance.n_customers="${N_CUSTOMERS}"
  )
else
  CMD+=( problem.instance.n_nodes="${N_NODES}" )
fi

# Append model/loss overrides only if provided
if [[ -n "$MODEL_OVERRIDE" ]]; then
  CMD+=("$MODEL_OVERRIDE")
fi

if [[ -n "$LOSSES_OVERRIDE" ]]; then
  CMD+=("$LOSSES_OVERRIDE")
fi

# Append any additional Hydra overrides
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DDP}" == "1" ]]; then
  if [[ -z "${NPROC_PER_NODE}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a _cuda_list <<< "${CUDA_VISIBLE_DEVICES}"
      NPROC_PER_NODE="${#_cuda_list[@]}"
    else
      NPROC_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count())')"
    fi
  fi
  if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "Error: NPROC_PER_NODE must be >= 1 for DDP." >&2
    exit 1
  fi
  LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
  echo "DDP: ${LAUNCHER[*]} ${CMD[*]}"
  "${LAUNCHER[@]}" "${CMD[@]}"
else
  echo "Running: python ${CMD[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
  fi
  python "${CMD[@]}"
fi
