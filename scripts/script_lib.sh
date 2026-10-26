#!/usr/bin/env bash
set -e

# Defaults: shared data/ and checkpoints/ inputs, with model/result runs in runs/.
_NEURPRISE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEURPRISE_INPUT_ROOT="${NEURPRISE_INPUT_ROOT:-$_NEURPRISE_REPO}"
export NEURPRISE_RUN_ROOT="${NEURPRISE_RUN_ROOT:-$_NEURPRISE_REPO/runs}"

get_k_list_raw() {
  local n_scenarios="${1:?n_scenarios required}"

  local default_k_list="0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16"
  if [[ "${n_scenarios}" == "100" ]]; then
    default_k_list="0.01,0.02,0.04,0.06,0.08"
  elif [[ "${n_scenarios}" == "200" ]]; then
    default_k_list="0.005,0.01,0.02,0.03,0.04"
  fi

  printf '%s\n' "${K_LIST_CLI:-${K_LIST:-${default_k_list}}}"
}

get_k_list() {
  local raw
  raw="$(get_k_list_raw "$1")"
  printf '%s\n' "${raw//,/ }"
}

get_eval_seed_suffix() {
  local eval_seed="${1:-}"
  if [[ -z "${eval_seed}" ]]; then
    printf '%s' ""
    return 0
  fi
  printf '_e%s' "${eval_seed}"
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

detect_free_gpus() {
  local map_lines=""
  local compute_lines=""
  local busy_pairs=""
  local busy_pairs_raw=""
  local fallback_used="0"
  local -A uuid_to_index=()
  local -A busy_uuid=()
  local -A busy_idx=()
  local -A busy_name=()
  local -a all_indices=()
  local -a free_indices=()
  local uuid=""
  local idx=""

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found; cannot auto-detect GPUs." >&2
    return 1
  fi

  map_lines="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null)" || true
  if [[ -z "${map_lines}" ]]; then
    echo "Error: failed to query GPU list from nvidia-smi." >&2
    return 1
  fi

  while IFS=',' read -r idx uuid; do
    idx="$(echo "${idx}" | tr -d ' ')"
    uuid="$(echo "${uuid}" | tr -d ' ')"
    if [[ -n "${idx}" && -n "${uuid}" ]]; then
      uuid_to_index["${uuid}"]="${idx}"
      all_indices+=( "${idx}" )
    fi
  done <<< "${map_lines}"

  if compute_lines="$(nvidia-smi --query-compute-apps=gpu_uuid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
    fallback_used="0"
  else
    fallback_used="1"
    compute_lines=""
  fi

  if [[ "${fallback_used}" = "0" ]]; then
    busy_pairs_raw="$(echo "${compute_lines}" | awk -F',' '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1);
        gsub(/^[ \t]+|[ \t]+$/, "", $2);
        gsub(/^[ \t]+|[ \t]+$/, "", $3);
        mem = $3 + 0;
        if ($1 != "") {
          memsum[$1] += mem;
          if ($2 != "") {
            if (names[$1] == "") {
              names[$1] = $2;
            } else if (names[$1] !~ ("(^|,)" $2 "(,|$)")) {
              names[$1] = names[$1] "," $2;
            }
          }
        }
      }
      END {
        for (u in memsum) {
          if (memsum[u] >= 2048) {
            print u "|" names[u] "|" memsum[u];
          }
        }
      }')"
    while IFS='|' read -r uuid names memsum; do
      [[ -z "${uuid}" ]] && continue
      busy_uuid["${uuid}"]=1
      busy_name["${uuid}"]="${names}"
      if [[ -n "${names}" ]]; then
        IFS=',' read -r -a _name_list <<< "${names}"
        for name in "${_name_list[@]}"; do
          busy_pairs+="${uuid}|${name}"$'\n'
        done
      else
        busy_pairs+="${uuid}|"$'\n'
      fi
    done <<< "${busy_pairs_raw}"
  else
    busy_pairs_raw="$(nvidia-smi 2>/dev/null | awk '
      BEGIN{in=0}
      /^Processes/ {in=1; next}
      in && $1 ~ /^[0-9]+$/ {
        gpu=$1; type=$5; pname=$6;
        mem=0;
        for (i=NF; i>=1; i--) {
          if ($i ~ /MiB/) {
            gsub(/MiB/, "", $i);
            mem = $i + 0;
            break;
          }
        }
        if (type=="C") {
          memsum[gpu] += mem;
          if (pname != "") {
            if (names[gpu] == "") {
              names[gpu] = pname;
            } else if (names[gpu] !~ ("(^|,)" pname "(,|$)")) {
              names[gpu] = names[gpu] "," pname;
            }
          }
        }
      }
      END {
        for (g in memsum) {
          if (memsum[g] >= 2048) {
            print g "|" names[g] "|" memsum[g];
          }
        }
      }
    ')"
    while IFS='|' read -r idx names memsum; do
      [[ -z "${idx}" ]] && continue
      busy_idx["${idx}"]=1
      busy_name["${idx}"]="${names}"
      if [[ -n "${names}" ]]; then
        IFS=',' read -r -a _name_list <<< "${names}"
        for name in "${_name_list[@]}"; do
          busy_pairs+="${idx}|${name}"$'\n'
        done
      else
        busy_pairs+="${idx}|"$'\n'
      fi
    done <<< "${busy_pairs_raw}"
  fi

  if [[ "${fallback_used}" = "0" ]]; then
    for uuid in "${!busy_uuid[@]}"; do
      idx="${uuid_to_index[${uuid}]}"
      if [[ -n "${idx}" ]]; then
        busy_idx["${idx}"]=1
      fi
    done
  fi

  for idx in "${all_indices[@]}"; do
    if [[ -z "${busy_idx[${idx}]+x}" ]]; then
      free_indices+=( "${idx}" )
    fi
  done

  DETECTED_BUSY=""
  DETECTED_BUSY_NAMES=""
  if [[ "${#busy_idx[@]}" -gt 0 ]]; then
    DETECTED_BUSY="$(printf '%s\n' "${!busy_idx[@]}" | sort -n | paste -sd, -)"
    DETECTED_BUSY_NAMES="$(printf '%s\n' "${busy_pairs}" | awk -F'|' '{print $2}' | sort -u | paste -sd, -)"
  fi
  DETECTED_FREE="$(printf '%s\n' "${free_indices[@]}" | sort -n | paste -sd, -)"
  FALLBACK_USED="${fallback_used}"

  if [[ -z "${DETECTED_FREE}" ]]; then
    echo "Error: no free GPUs detected." >&2
    return 1
  fi
  printf '%s' "${DETECTED_FREE}"
}

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
        EXPERIMENTS+=( "${p}|${m}|${l}" )
      done
    done
  done
}

yaml_value() {
  local file="${1:?file required}"
  local key="${2:?key required}"
  sed -n "s/^  ${key}: //p" "${file}" | head -n 1 | sed 's/^"//; s/"$//'
}

resolve_data_file() {
  local problem="${1:?problem required}"
  local n_instances="${2:-}"
  local n_nodes="${3:-}"
  local n_scenarios="${4:-}"
  local dist_name="${5:-}"
  local eval_seed="${6:-}"
  local eval_p_edge="${7:-}"
  local n_customers="${8:-}"
  local problem_config="configs/problem/${problem}.yaml"
  local data_root=""
  local inst_spec_pattern=""
  local resolved_pattern=""
  local unresolved=0

  if [[ ! -f "${problem_config}" ]]; then
    printf '%s' "cfg-derived (no problem config)"
    return 0
  fi

  data_root="$(yaml_value "${problem_config}" "path")"
  inst_spec_pattern="$(yaml_value "${problem_config}" "inst_spec_pattern")"
  if [[ -z "${data_root}" || -z "${inst_spec_pattern}" ]]; then
    printf '%s' "cfg-derived (missing data.path or inst_spec_pattern)"
    return 0
  fi

  data_root="${data_root/\${oc.env:PROJECT_DIR,.}/${PROJECT_DIR:-.}}"
  resolved_pattern="${inst_spec_pattern}"

  if [[ "${resolved_pattern}" == *"{seed}"* ]]; then
    if [[ -n "${eval_seed}" ]]; then
      resolved_pattern="${resolved_pattern//\{seed\}/${eval_seed}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{instance_cnt}"* ]]; then
    if [[ -n "${n_instances}" ]]; then
      resolved_pattern="${resolved_pattern//\{instance_cnt\}/${n_instances}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{n_nodes}"* ]]; then
    if [[ -n "${n_nodes}" ]]; then
      resolved_pattern="${resolved_pattern//\{n_nodes\}/${n_nodes}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{n_facilities}"* ]]; then
    if [[ -n "${n_nodes}" ]]; then
      resolved_pattern="${resolved_pattern//\{n_facilities\}/${n_nodes}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{n_customers}"* ]]; then
    if [[ -n "${n_customers}" ]]; then
      resolved_pattern="${resolved_pattern//\{n_customers\}/${n_customers}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{m}"* ]]; then
    if [[ -n "${n_nodes}" ]]; then
      resolved_pattern="${resolved_pattern//\{m\}/${n_nodes}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{n}"* ]]; then
    if [[ -n "${n_customers}" ]]; then
      resolved_pattern="${resolved_pattern//\{n\}/${n_customers}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{n_scenarios}"* ]]; then
    if [[ -n "${n_scenarios}" ]]; then
      resolved_pattern="${resolved_pattern//\{n_scenarios\}/${n_scenarios}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{p_edge}"* ]]; then
    if [[ -n "${eval_p_edge}" ]]; then
      resolved_pattern="${resolved_pattern//\{p_edge\}/${eval_p_edge}}"
    else
      unresolved=1
    fi
  fi
  if [[ "${resolved_pattern}" == *"{p_select}"* ]]; then
    unresolved=1
  fi
  if [[ "${resolved_pattern}" == *"{scenario_name}"* ]]; then
    if [[ -n "${dist_name}" ]]; then
      resolved_pattern="${resolved_pattern//\{scenario_name\}/${dist_name}}"
    else
      unresolved=1
    fi
  fi

  if [[ "${data_root}" == *'${'* ]]; then
    unresolved=1
  fi

  if [[ "${unresolved}" = "0" ]]; then
    printf '%s' "${data_root%/}/${resolved_pattern}"
  else
    printf '%s' "cfg-derived (${data_root%/}/${inst_spec_pattern})"
  fi
}
