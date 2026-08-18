#!/usr/bin/env bash
# Steal remaining Phase-G evals onto free GPUs so GPU7's serial tail does not block.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/h20_clean_auto_0817}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
BASE_OSS="${BASE_OSS:-/data/ppnm/models/gpt-oss-20b}"
FULL_S42="${FULL_S42:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint}"
RAW_JSONL="${RAW_JSONL:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/data/hf_raw/sft_trajectories.jsonl}"
LOG_DIR="${OUT_ROOT}/logs"
PARENT_ADAPTER="${FULL_S42}"
FREE_MIB_MIN="${FREE_MIB_MIN:-50000}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/ppnm/.cache/huggingface}"

ids_dev="${OUT_ROOT}/real_eval/dev_ids.json"
ids_smoke="${OUT_ROOT}/real_eval/smoke_ids.json"
ids_test="${OUT_ROOT}/real_eval/test_ids.json"

job_running() {
  local dest="$1"
  pgrep -f "run_auto_clean_real_eval.py --out ${dest}" >/dev/null 2>&1
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | awk '{print $1}'
}

gpu_lock_busy() {
  local gpu="$1"
  local lf="${OUT_ROOT}/pids/gpu${gpu}_steal.lock"
  [[ -f "$lf" ]] || return 1
  local pid
  pid=$(awk -F= '/^pid=/{print $2}' "$lf" | head -1)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

our_gpu_busy() {
  local gpu="$1" pid
  if gpu_lock_busy "$gpu"; then
    echo busy
    return 0
  fi
  while read -r pid; do
    pid="${pid// /}"
    [[ -z "$pid" || "$pid" == "[N/A]" ]] && continue
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "run_auto_clean_real_eval.py"; then
      echo busy
      return 0
    fi
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null)
  return 1
}

launch_eval() {
  local gpu="$1" tag="$2" model="$3" dest="$4" idfile="$5" steps="${6:-6}" parent="${7:-}"
  if [[ -f "${dest}/DONE" ]]; then
    echo "[steal] skip DONE $tag"
    return 0
  fi
  if job_running "$dest"; then
    echo "[steal] skip running $tag"
    return 0
  fi
  mkdir -p "$dest"
  echo "stolen_by=steal_h20_phase_g_jobs.sh gpu=${gpu} ts=$(date -Iseconds)" > "${dest}/STOLEN"
  local parent_args=()
  if [[ -n "$parent" ]]; then
    parent_args+=(--parent-adapter "$parent")
  fi
  echo "[steal] launch gpu=${gpu} ${tag} -> ${dest}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME="${HF_HOME:-/data/ppnm/.cache/huggingface}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_auto_clean_real_eval.py" \
      --out "${dest}" --model-path "${model}" --base-model "${BASE_OSS}" \
      --raw-jsonl "${RAW_JSONL}" --query-ids-json "${idfile}" \
      --tag "${tag}" --max-steps "${steps}" --gpu 0 --max-new-tokens 384 \
      "${parent_args[@]}" \
      >"${LOG_DIR}/G_gpu${gpu}_${tag}.log" 2>&1 &
  local spid=$!
  echo "$spid" > "${OUT_ROOT}/pids/steal_${tag}.pid"
  printf 'pid=%s\ntag=%s\ndest=%s\n' "$spid" "$tag" "$dest" > "${OUT_ROOT}/pids/gpu${gpu}_steal.lock"
  echo "[steal] python pid=${spid} gpu=${gpu} ${tag}"
}

# Remaining jobs in preferred steal order (longest / GPU7 serial tail first).
# Fields: tag|model|dest|idfile|steps|parent(1/0)
UNSH42="${OUT_ROOT}/phase_E/unshuffled_s42/lora_checkpoint"
UNSH43="${OUT_ROOT}/phase_E/unshuffled_s43/lora_checkpoint"
UNSH44="${OUT_ROOT}/phase_E/unshuffled_s44/lora_checkpoint"
UNSH45="${OUT_ROOT}/phase_E/unshuffled_s45/lora_checkpoint"
SH42="${OUT_ROOT}/phase_E/shuffled_s42/lora_checkpoint"
SH43="${OUT_ROOT}/phase_E/shuffled_s43/lora_checkpoint"
SH44="${OUT_ROOT}/phase_E/shuffled_s44/lora_checkpoint"
SH45="${OUT_ROOT}/phase_E/shuffled_s45/lora_checkpoint"

pending_line() {
  local tag="$1" model="$2" dest="$3" idfile="$4" steps="$5" use_parent="$6"
  if [[ -f "${dest}/DONE" ]]; then
    return 1
  fi
  if job_running "$dest"; then
    return 1
  fi
  echo "${tag}|${model}|${dest}|${idfile}|${steps}|${use_parent}"
}

list_pending() {
  # Only GPU7 serial tail (bash is SIGSTOP'd). Do not steal jobs still owned by live GPU1-6 queues.
  pending_line AUTO_CLEAN_SHUFFLED_s44_TEST "$SH44" "${OUT_ROOT}/phase_G/sh_s44_test" "$ids_test" 6 1 || true
  pending_line AUTO_CLEAN_SHUFFLED_s45_TEST "$SH45" "${OUT_ROOT}/phase_G/sh_s45_test" "$ids_test" 6 1 || true
  pending_line CLEAN_FULL_HARNESS_TEST "$FULL_S42" "${OUT_ROOT}/phase_G/full_harness_test" "$ids_test" 6 0 || true
}

free_gpus() {
  local g
  for g in 0 1 2 3 4 5 6 7; do
    local free busy
    free=$(gpu_free_mib "$g")
    free=${free:-0}
    busy=$(our_gpu_busy "$g" || true)
    if [[ "$busy" == "busy" ]]; then
      continue
    fi
    if (( free >= FREE_MIB_MIN )); then
      echo "$g"
    fi
  done
}

mapfile -t PENDING < <(list_pending | sed '/^$/d')
mapfile -t FREEG < <(free_gpus)

echo "[steal] pending=${#PENDING[@]} free_gpus=${FREEG[*]:-none}"
if ((${#PENDING[@]} == 0)); then
  echo "[steal] nothing to steal"
  exit 0
fi
if ((${#FREEG[@]} == 0)); then
  echo "[steal] no free GPU"
  exit 0
fi

n=${#FREEG[@]}
for i in "${!FREEG[@]}"; do
  (( i < ${#PENDING[@]} )) || break
  gpu="${FREEG[$i]}"
  IFS='|' read -r tag model dest idfile steps use_parent <<<"${PENDING[$i]}"
  parent=""
  if [[ "$use_parent" == "1" ]]; then
    parent="$PARENT_ADAPTER"
  fi
  launch_eval "$gpu" "$tag" "$model" "$dest" "$idfile" "$steps" "$parent"
  echo "[steal] dispatched gpu=${gpu} ${tag}"
done
if [[ "${DISPATCH_ONCE:-0}" == "1" ]]; then
  echo "[steal] dispatch-once; not waiting"
  exit 0
fi
wait || true
echo "[steal] batch exit"
