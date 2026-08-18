#!/usr/bin/env bash
# H20 2026-08-17 clean-init AUTO OPD — 8×H20 phase queues.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/h20_clean_auto_0817}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
BASE_OSS="${BASE_OSS:-/data/ppnm/models/gpt-oss-20b}"
FULL_S42="${FULL_S42:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint}"
FULL_S43="${FULL_S43:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/sft/gpu1/full_s43_full/lora_checkpoint}"
TOOL_S42="${TOOL_S42:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/sft/gpu2/tool_s42_full/lora_checkpoint}"
TOOL_S43="${TOOL_S43:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/sft/gpu3/tool_s43_full/lora_checkpoint}"
RAW_JSONL="${RAW_JSONL:-${SCAPE_ROOT}/outputs/0814_clean_mechanism/data/hf_raw/sft_trajectories.jsonl}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
GPU_ONLY="${GPU_ONLY:-}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${PID_DIR}" \
  "${OUT_ROOT}/base_recovery" "${OUT_ROOT}/auto_data" "${OUT_ROOT}/value" \
  "${OUT_ROOT}/training" "${OUT_ROOT}/real_eval"

if [[ ! -f "${OUT_ROOT}/PHASE" ]]; then
  echo A > "${OUT_ROOT}/PHASE"
fi
PHASE="$(tr -d '[:space:]' < "${OUT_ROOT}/PHASE")"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/ppnm/.cache/huggingface}"

wait_file() {
  local f="$1" secs="${2:-86400}" min_bytes="${3:-1}"
  local i=0
  while true; do
    local sz=0
    if [[ -f "$f" ]]; then
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      if (( sz >= min_bytes )); then
        echo "[wait] ready $f (${sz} bytes)"
        return 0
      fi
    fi
    if (( i % 60 == 0 )); then
      echo "[wait] heartbeat ${i}s for $f (sz=${sz})"
    fi
    sleep 15
    i=$((i + 15))
    if (( i >= secs )); then
      echo "[wait] timeout $f"
      return 1
    fi
  done
}

skip_done() {
  local d="$1"
  if [[ -f "${d}/DONE" ]]; then
    echo "[skip] $d"
    return 0
  fi
  if [[ -f "${d}/STOLEN" ]]; then
    echo "[skip] stolen $d — waiting for DONE"
    wait_file "${d}/DONE" 86400
    return 0
  fi
  if pgrep -f "run_auto_clean_real_eval.py --out ${d}" >/dev/null 2>&1; then
    echo "[skip] already running $d — waiting for DONE"
    wait_file "${d}/DONE" 86400
    return 0
  fi
  return 1
}

skip_if_low_mem() {
  local gpu="$1" dest="$2"
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | awk '{print $1}')
  free=${free:-0}
  if (( free < 45000 )); then
    echo "[skip] gpu${gpu} free=${free}MiB < 45000 — OOM risk"
    mkdir -p "${dest}"
    echo "{\"skipped_oom\": true, \"free_mib\": ${free}}" > "${dest}/SKIPPED_OOM.json"
    echo "{\"skipped_oom\": true, \"gpu\": ${gpu}}" > "${dest}/summary.json"
    touch "${dest}/DONE"
    return 0
  fi
  return 1
}

eval128() {
  local gpu="$1" tag="$2" model="$3" dest="$4" limit="${5:-0}"
  skip_done "${dest}" && return 0
  mkdir -p "${dest}"
  local extra=()
  if (( limit > 0 )); then extra+=(--limit "${limit}"); fi
  echo "[launch] eval128 gpu=${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_base_eval128.py" \
      --out "${dest}" \
      --model-path "${model}" \
      --base-model "${BASE_OSS}" \
      --manifest "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" \
      --gpu 0 \
      --tag "${tag}" \
      "${extra[@]}" \
      >"${LOG_DIR}/${PHASE}_gpu${gpu}_${tag}.log" 2>&1
}

phase_A() {
  local gpu="$1"
  case "${gpu}" in
    6)
      skip_done "${OUT_ROOT}/base_recovery/data_build" && return 0
      mkdir -p "${OUT_ROOT}/base_recovery/data_build"
      echo "[launch] GPU6 query manifests + format-repair data"
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/build_h20_query_manifests.py" \
        --out "${OUT_ROOT}" \
        >"${LOG_DIR}/A_gpu6_data.log" 2>&1
      touch "${OUT_ROOT}/base_recovery/data_build/DONE"
      ;;
    7)
      skip_done "${OUT_ROOT}/base_recovery/audit" && return 0
      mkdir -p "${OUT_ROOT}/base_recovery/audit"
      echo "[launch] GPU7 Harmony contract tests"
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_harmony_runtime_audit.py" \
        --out "${OUT_ROOT}/base_recovery" \
        --model-path "${BASE_OSS}" \
        >"${LOG_DIR}/A_gpu7_audit.log" 2>&1
      touch "${OUT_ROOT}/base_recovery/audit/DONE"
      ;;
    0)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 0 CLEAN_FULL_S42 "${FULL_S42}" "${OUT_ROOT}/phase_A/gpu0/full_s42_eval128"
      ;;
    1)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 1 CLEAN_FULL_S43 "${FULL_S43}" "${OUT_ROOT}/phase_A/gpu1/full_s43_eval128"
      ;;
    2)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 2 CLEAN_TOOL_S42 "${TOOL_S42}" "${OUT_ROOT}/phase_A/gpu2/tool_s42_eval128"
      ;;
    3)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 3 CLEAN_TOOL_S43 "${TOOL_S43}" "${OUT_ROOT}/phase_A/gpu3/tool_s43_eval128"
      ;;
    4)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 4 RAW_GPT_OSS "${BASE_OSS}" "${OUT_ROOT}/phase_A/gpu4/raw_eval128"
      ;;
    5)
      wait_file "${OUT_ROOT}/base_recovery/BASE_QUERY_MANIFEST.json" 7200 200
      eval128 5 CLEAN_FULL_S42_REPLAY32 "${FULL_S42}" "${OUT_ROOT}/phase_A/gpu5/full_s42_replay32" 32
      ;;
  esac
}

phase_B() {
  local gpu="$1"
  local frdata="${OUT_ROOT}/base_recovery/format_repair_data"
  wait_file "${frdata}/FORMAT_REPAIR_TRAIN.jsonl" 7200 1000
  case "${gpu}" in
    0)
      skip_done "${OUT_ROOT}/phase_B/gpu0/FR_A" && true || {
        mkdir -p "${OUT_ROOT}/phase_B/gpu0/FR_A"
        CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_format_repair_cell.py" \
          --out "${OUT_ROOT}/phase_B/gpu0/FR_A" --adapter-path "${FULL_S42}" \
          --train-jsonl "${frdata}/FORMAT_REPAIR_TRAIN.jsonl" --mask-mode format_aware \
          --seed 42 --tag FR_A --gpu 0 >"${LOG_DIR}/B_gpu0_FR_A.log" 2>&1
      }
      wait_file "${OUT_ROOT}/phase_B/gpu0/FR_A/DONE" 86400
      eval128 0 FR_A_EVAL "${OUT_ROOT}/phase_B/gpu0/FR_A/lora_checkpoint" "${OUT_ROOT}/phase_B/eval_FR_A"
      ;;
    1)
      skip_done "${OUT_ROOT}/phase_B/gpu1/FR_B" && true || {
        mkdir -p "${OUT_ROOT}/phase_B/gpu1/FR_B"
        CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_format_repair_cell.py" \
          --out "${OUT_ROOT}/phase_B/gpu1/FR_B" --adapter-path "${FULL_S43}" \
          --train-jsonl "${frdata}/FORMAT_REPAIR_TRAIN.jsonl" --mask-mode format_aware \
          --seed 43 --tag FR_B --gpu 0 >"${LOG_DIR}/B_gpu1_FR_B.log" 2>&1
      }
      wait_file "${OUT_ROOT}/phase_B/gpu1/FR_B/DONE" 86400
      eval128 1 FR_B_EVAL "${OUT_ROOT}/phase_B/gpu1/FR_B/lora_checkpoint" "${OUT_ROOT}/phase_B/eval_FR_B"
      ;;
    2)
      skip_done "${OUT_ROOT}/phase_B/gpu2/FR_C" && true || {
        mkdir -p "${OUT_ROOT}/phase_B/gpu2/FR_C"
        CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_format_repair_cell.py" \
          --out "${OUT_ROOT}/phase_B/gpu2/FR_C" --adapter-path "${FULL_S42}" \
          --train-jsonl "${frdata}/FORMAT_REPAIR_TRAIN_ENDUP.jsonl" --mask-mode format_aware \
          --seed 42 --tag FR_C --gpu 0 >"${LOG_DIR}/B_gpu2_FR_C.log" 2>&1
      }
      wait_file "${OUT_ROOT}/phase_B/gpu2/FR_C/DONE" 86400
      eval128 2 FR_C_EVAL "${OUT_ROOT}/phase_B/gpu2/FR_C/lora_checkpoint" "${OUT_ROOT}/phase_B/eval_FR_C"
      ;;
    3)
      skip_done "${OUT_ROOT}/phase_B/gpu3/FR_D" && true || {
        mkdir -p "${OUT_ROOT}/phase_B/gpu3/FR_D"
        CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_format_repair_cell.py" \
          --out "${OUT_ROOT}/phase_B/gpu3/FR_D" --adapter-path "${FULL_S42}" \
          --train-jsonl "${frdata}/FORMAT_REPAIR_TRAIN.jsonl" --mask-mode full \
          --seed 42 --tag FR_D --gpu 0 >"${LOG_DIR}/B_gpu3_FR_D.log" 2>&1
      }
      wait_file "${OUT_ROOT}/phase_B/gpu3/FR_D/DONE" 86400
      eval128 3 FR_D_EVAL "${OUT_ROOT}/phase_B/gpu3/FR_D/lora_checkpoint" "${OUT_ROOT}/phase_B/eval_FR_D"
      ;;
    *)
      echo "[gpu${gpu}] phase B spare — sleep"
      sleep 60
      ;;
  esac
}

clean_base() {
  "${PYTHON_BIN}" -c "import json; print(json.load(open('${OUT_ROOT}/base_recovery/CLEAN_AUTO_BASE.json'))['model_path'])"
}

phase_C() {
  local gpu="$1"
  wait_file "${OUT_ROOT}/base_recovery/CLEAN_AUTO_BASE.json" 86400 50
  wait_file "${OUT_ROOT}/auto_data/AUTO_CLEAN_SPLIT_MANIFEST.json" 7200 50
  local dest="${OUT_ROOT}/phase_C/gpu${gpu}"
  skip_done "${dest}" && return 0
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | awk '{print $1}')
  free=${free:-0}
  if (( free < 45000 )); then
    echo "[skip] gpu${gpu} free=${free}MiB < 45000 — OOM risk, skip collect shard"
    mkdir -p "${dest}"
    echo "{\"skipped_oom\": true, \"free_mib\": ${free}}" > "${dest}/SKIPPED_OOM.json"
    echo "{\"shard_id\": ${gpu}, \"n_kept\": 0, \"skipped_oom\": true}" > "${dest}/summary.json"
    touch "${dest}/DONE"
    return 0
  fi
  mkdir -p "${dest}"
  local base
  base="$(clean_base)"
  echo "[launch] collect shard gpu=${gpu} base=${base} free=${free}MiB"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/collect_auto_clean_states.py" \
      --out "${dest}" \
      --model-path "${base}" \
      --base-model "${BASE_OSS}" \
      --split-manifest "${OUT_ROOT}/auto_data/AUTO_CLEAN_SPLIT_MANIFEST.json" \
      --raw-jsonl "${RAW_JSONL}" \
      --shard-id "${gpu}" \
      --n-shards 8 \
      --gpu 0 \
      >"${LOG_DIR}/C_gpu${gpu}_collect.log" 2>&1
}

phase_D() {
  local gpu="$1"
  wait_file "${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN.jsonl" 86400 100
  wait_file "${OUT_ROOT}/base_recovery/CLEAN_AUTO_BASE.json" 86400 50
  local base k dest
  base="$(clean_base)"
  if (( gpu < 4 )); then
    k=4
    dest="${OUT_ROOT}/phase_D/k4_gpu${gpu}"
  else
    k=8
    dest="${OUT_ROOT}/phase_D/k8_gpu${gpu}"
  fi
  skip_done "${dest}" && return 0
  skip_if_low_mem "${gpu}" "${dest}" && return 0
  mkdir -p "${dest}"
  local shard=$(( gpu % 4 ))
  echo "[launch] value K${k} gpu=${gpu} shard=${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_auto_clean_value.py" \
      --out "${dest}" \
      --model-path "${base}" \
      --states-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN.jsonl" \
      --k "${k}" --n-seeds 2 \
      --shard-id "${shard}" --n-shards 4 \
      --raw-jsonl "${RAW_JSONL}" \
      --gpu 0 \
      >"${LOG_DIR}/D_gpu${gpu}_K${k}.log" 2>&1
}

phase_E() {
  local gpu="$1"
  wait_file "${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN.jsonl" 86400 100
  wait_file "${OUT_ROOT}/auto_data/AUTO_CLEAN_VALID.jsonl" 86400 50
  wait_file "${OUT_ROOT}/base_recovery/CLEAN_AUTO_BASE.json" 86400 50
  local base
  local -a seeds=(42 43 44 45)
  base="$(clean_base)"
  if (( gpu < 4 )); then
    local seed="${seeds[$gpu]}"
    local dest="${OUT_ROOT}/phase_E/unshuffled_s${seed}"
    skip_done "${dest}" && return 0
    mkdir -p "${dest}"
    echo "[launch] unshuffled seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
        --out "${dest}" --model-path "${base}" \
        --train-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN.jsonl" \
        --valid-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_VALID.jsonl" \
        --test-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_TEST.jsonl" \
        --component-id auto_populate_first_search \
        --n-samples 4096 --seed "${seed}" --loss-path route_kl \
        --gpu 0 --epochs 1 --lr 1e-5 --lora-r 8 --anchor-weight 0.05 --lambda-args 0.0 \
        >"${LOG_DIR}/E_gpu${gpu}_unsh_s${seed}.log" 2>&1
  else
    local si=$((gpu - 4))
    local seed="${seeds[$si]}"
    local shuf="${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN_SHUFFLED.jsonl"
    if [[ "${gpu}" == "4" && ! -f "${shuf}" ]]; then
      mkdir -p "${OUT_ROOT}/training"
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/shuffle_auto_clean_targets.py" \
        --in-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_TRAIN.jsonl" \
        --out-jsonl "${shuf}" \
        --audit "${OUT_ROOT}/training/AUTO_CLEAN_SHUFFLE_AUDIT.json" \
        --seed 817
    fi
    wait_file "${shuf}" 7200 100
    local dest="${OUT_ROOT}/phase_E/shuffled_s${seed}"
    skip_done "${dest}" && return 0
    mkdir -p "${dest}"
    echo "[launch] shuffled seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
        --out "${dest}" --model-path "${base}" \
        --train-jsonl "${shuf}" \
        --valid-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_VALID.jsonl" \
        --test-jsonl "${OUT_ROOT}/auto_data/AUTO_CLEAN_TEST.jsonl" \
        --component-id auto_populate_first_search \
        --n-samples 4096 --seed "${seed}" --loss-path route_kl \
        --gpu 0 --epochs 1 --lr 1e-5 --lora-r 8 --anchor-weight 0.05 --lambda-args 0.0 \
        >"${LOG_DIR}/E_gpu${gpu}_sh_s${seed}.log" 2>&1
  fi
}

write_ids_json() {
  local key="$1" dest="$2" limit="${3:-0}"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
man=json.loads(Path("${OUT_ROOT}/auto_data/AUTO_CLEAN_SPLIT_MANIFEST.json").read_text())
ids=list(man.get("${key}") or [])
lim=int("${limit}")
if lim>0: ids=ids[:lim]
Path("${dest}").write_text(json.dumps({"query_ids": ids})+"\n")
PY
}

phase_G() {
  local gpu="$1"
  wait_file "${OUT_ROOT}/base_recovery/CLEAN_AUTO_BASE.json" 86400 50
  wait_file "${OUT_ROOT}/auto_data/AUTO_CLEAN_SPLIT_MANIFEST.json" 7200 50
  local base ids_dev ids_smoke ids_test
  base="$(clean_base)"
  PARENT_ADAPTER="${base}"
  ids_dev="${OUT_ROOT}/real_eval/dev_ids.json"
  ids_smoke="${OUT_ROOT}/real_eval/smoke_ids.json"
  ids_test="${OUT_ROOT}/real_eval/test_ids.json"
  if [[ ! -f "${ids_dev}" ]]; then
    write_ids_json real_dev_query_ids "${ids_dev}" 0
  fi
  if [[ ! -f "${ids_smoke}" ]]; then
    write_ids_json real_dev_query_ids "${ids_smoke}" 16
  fi
  if [[ ! -f "${ids_test}" ]]; then
    write_ids_json real_test_query_ids "${ids_test}" 0
  fi
  real_eval() {
    local g="$1" tag="$2" model="$3" dest="$4" idfile="$5" steps="${6:-6}"
    skip_done "${dest}" && return 0
    mkdir -p "${dest}"
    local parent_args=()
    if [[ -n "${PARENT_ADAPTER:-}" && "${model}" != "${base}" && "${model}" != "${FULL_S42}" && "${model}" != "${BASE_OSS}" ]]; then
      parent_args+=(--parent-adapter "${PARENT_ADAPTER}")
    fi
    CUDA_VISIBLE_DEVICES="${g}" \
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_auto_clean_real_eval.py" \
        --out "${dest}" --model-path "${model}" --base-model "${BASE_OSS}" \
        --raw-jsonl "${RAW_JSONL}" --query-ids-json "${idfile}" \
        --tag "${tag}" --max-steps "${steps}" --gpu 0 --max-new-tokens 384 \
        "${parent_args[@]}" \
        >"${LOG_DIR}/G_gpu${g}_${tag}.log" 2>&1
  }
  case "${gpu}" in
    0)
      real_eval 0 CLEAN_BASE "${base}" "${OUT_ROOT}/phase_G/CLEAN_BASE" "${ids_dev}" 6
      real_eval 0 CLEAN_BASE_TEST "${base}" "${OUT_ROOT}/phase_G/CLEAN_BASE_TEST" "${ids_test}" 6
      real_eval 0 CLEAN_BASE_S10 "${base}" "${OUT_ROOT}/phase_G/CLEAN_BASE_S10" "${ids_smoke}" 10
      ;;
    1)
      real_eval 1 SMOKE_UNSH "${OUT_ROOT}/phase_E/unshuffled_s42/lora_checkpoint" "${OUT_ROOT}/phase_G/smoke_unsh" "${ids_smoke}" 6
      real_eval 1 AUTO_CLEAN_UNSHUFFLED_s42 "${OUT_ROOT}/phase_E/unshuffled_s42/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s42" "${ids_dev}" 6
      real_eval 1 AUTO_CLEAN_UNSHUFFLED_s42_TEST "${OUT_ROOT}/phase_E/unshuffled_s42/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s42_test" "${ids_test}" 6
      ;;
    2)
      real_eval 2 AUTO_CLEAN_UNSHUFFLED_s43 "${OUT_ROOT}/phase_E/unshuffled_s43/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s43" "${ids_dev}" 6
      real_eval 2 AUTO_CLEAN_UNSHUFFLED_s43_TEST "${OUT_ROOT}/phase_E/unshuffled_s43/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s43_test" "${ids_test}" 6
      ;;
    3)
      real_eval 3 AUTO_CLEAN_UNSHUFFLED_s44 "${OUT_ROOT}/phase_E/unshuffled_s44/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s44" "${ids_dev}" 6
      real_eval 3 AUTO_CLEAN_UNSHUFFLED_s44_TEST "${OUT_ROOT}/phase_E/unshuffled_s44/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s44_test" "${ids_test}" 6
      ;;
    4)
      real_eval 4 AUTO_CLEAN_UNSHUFFLED_s45 "${OUT_ROOT}/phase_E/unshuffled_s45/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s45" "${ids_dev}" 6
      real_eval 4 AUTO_CLEAN_UNSHUFFLED_s45_TEST "${OUT_ROOT}/phase_E/unshuffled_s45/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s45_test" "${ids_test}" 6
      real_eval 4 UNSH_S43_S10 "${OUT_ROOT}/phase_E/unshuffled_s43/lora_checkpoint" "${OUT_ROOT}/phase_G/unsh_s43_s10" "${ids_smoke}" 10
      ;;
    5)
      real_eval 5 AUTO_CLEAN_SHUFFLED_s42 "${OUT_ROOT}/phase_E/shuffled_s42/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s42" "${ids_dev}" 6
      real_eval 5 AUTO_CLEAN_SHUFFLED_s42_TEST "${OUT_ROOT}/phase_E/shuffled_s42/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s42_test" "${ids_test}" 6
      ;;
    6)
      real_eval 6 AUTO_CLEAN_SHUFFLED_s43 "${OUT_ROOT}/phase_E/shuffled_s43/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s43" "${ids_dev}" 6
      real_eval 6 AUTO_CLEAN_SHUFFLED_s43_TEST "${OUT_ROOT}/phase_E/shuffled_s43/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s43_test" "${ids_test}" 6
      ;;
    7)
      real_eval 7 SMOKE_BASE "${base}" "${OUT_ROOT}/phase_G/smoke_base" "${ids_smoke}" 6
      real_eval 7 AUTO_CLEAN_SHUFFLED_s44 "${OUT_ROOT}/phase_E/shuffled_s44/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s44" "${ids_dev}" 6
      real_eval 7 AUTO_CLEAN_SHUFFLED_s45 "${OUT_ROOT}/phase_E/shuffled_s45/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s45" "${ids_dev}" 6
      real_eval 7 AUTO_CLEAN_SHUFFLED_s44_TEST "${OUT_ROOT}/phase_E/shuffled_s44/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s44_test" "${ids_test}" 6
      real_eval 7 AUTO_CLEAN_SHUFFLED_s45_TEST "${OUT_ROOT}/phase_E/shuffled_s45/lora_checkpoint" "${OUT_ROOT}/phase_G/sh_s45_test" "${ids_test}" 6
      real_eval 7 CLEAN_FULL_HARNESS "${FULL_S42}" "${OUT_ROOT}/phase_G/full_harness" "${ids_dev}" 6
      real_eval 7 CLEAN_FULL_HARNESS_TEST "${FULL_S42}" "${OUT_ROOT}/phase_G/full_harness_test" "${ids_test}" 6
      ;;
  esac
}

run_gpu() {
  local gpu="$1"
  local log="${LOG_DIR}/${PHASE}_gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} phase=${PHASE} start"
    case "${PHASE}" in
      A) phase_A "${gpu}" ;;
      B) phase_B "${gpu}" ;;
      C) phase_C "${gpu}" ;;
      D) phase_D "${gpu}" ;;
      E) phase_E "${gpu}" ;;
      G) phase_G "${gpu}" ;;
      STOP|DONE) echo "[gpu${gpu}] terminal phase ${PHASE}"; sleep 30 ;;
      *) echo "[gpu${gpu}] unknown phase ${PHASE}" ;;
    esac
    mkdir -p "${OUT_ROOT}/phase_${PHASE}/gpu${gpu}"
    touch "${OUT_ROOT}/phase_${PHASE}/gpu${gpu}/ALL_DONE"
    echo "[$(date -Iseconds)] gpu${gpu} phase=${PHASE} queue exit"
  } >>"${log}" 2>&1
}

write_contract() {
  cat > "${OUT_ROOT}/real_eval/AUTO_CLEAN_REAL_EVAL_CONTRACT.md" <<'EOF'
# AUTO_CLEAN_REAL_EVAL_CONTRACT

- query source: `pat-jj/harness-1-train-data` stage=sft unique queries (query-disjoint from BASE_EVAL_128)
- query manifest: `auto_data/AUTO_CLEAN_SPLIT_MANIFEST.json`
- retriever: LOCAL_COMPAT_ONLY in-process overlap ranker over per-query `doc_store` (not official Chroma)
- qrel/evidence gold: `ground_truth_ids` when present; metric = curated recall
- final-answer gold: **N/A** (not written as 0)
- reward: evidence/qrel recall; if missing → N/A
- tool cost: 1 per tool call (logged, not subtracted unless specified)
- max steps: 6 (sanity 10/12 optional)
- termination: `end_search` or max_steps
- tool parser: strict Harmony `to=functions.NAME` + JSON
- tool set: 8 canonical Harness tools
- student inference privilege: **false**
EOF
}

if [[ ! -f "${OUT_ROOT}/RUN_MANIFEST.json" ]]; then
  cat > "${OUT_ROOT}/RUN_MANIFEST.json" <<EOF
{
  "run_id": "h20_clean_auto_0817",
  "started": "$(date -Iseconds)",
  "machine": "8xH20",
  "base_model": "${BASE_OSS}",
  "LOCAL_COMPAT_ONLY": true,
  "legacy_scope_path_used": false
}
EOF
fi
write_contract

if [[ -n "${GPU_ONLY}" ]]; then
  echo "[main] foreground phase=${PHASE} gpu${GPU_ONLY}"
  echo $$ > "${PID_DIR}/${PHASE}_gpu${GPU_ONLY}.pid"
  run_gpu "${GPU_ONLY}"
else
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu "${g}" &
    echo $! > "${PID_DIR}/${PHASE}_gpu${g}.pid"
    echo "[main] launched phase=${PHASE} gpu${g} pid=$(cat "${PID_DIR}/${PHASE}_gpu${g}.pid")"
  done
  wait || true
fi
