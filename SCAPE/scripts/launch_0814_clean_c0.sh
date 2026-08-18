#!/usr/bin/env bash
# SCAPE-0814-H20 Phase C0 — Clean Mechanism 8×H20 queue.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0814_clean_mechanism}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/gpt-oss-20b}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
DATA_DIR="${OUT_ROOT}/data"
TRAIN_JSONL="${DATA_DIR}/CLEAN_SFT_TRAIN.jsonl"
SMOKE_N="${SMOKE_N:-32}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-3}"
LORA_R="${LORA_R:-32}"
LR="${LR:-5e-6}"
MAX_LEN="${MAX_LEN:-4096}"
SMOKE_MAX_LEN="${SMOKE_MAX_LEN:-2048}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${PID_DIR}" \
  "${OUT_ROOT}/sft" "${OUT_ROOT}/evals" "${OUT_ROOT}/prestage"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/data/ppnm/.cache/huggingface}"

wait_file() {
  local f="$1" secs="${2:-7200}" min_bytes="${3:-1}"
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
      echo "[wait] heartbeat ${i}s for $f (sz=${sz}, need>=${min_bytes})"
    fi
    sleep 15
    i=$((i + 15))
    if (( i >= secs )); then
      echo "[wait] timeout waiting for $f (>= ${min_bytes} bytes)"
      return 1
    fi
  done
}

run_sft() {
  local gpu="$1" tag="$2" mask="$3" seed="$4" n="$5" epochs="$6" maxlen="$7"
  local out="${OUT_ROOT}/sft/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} mask=${mask} seed=${seed} n=${n} ep=${epochs}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_sft_cell.py" \
      --out "${out}" \
      --model-path "${MODEL_PATH}" \
      --train-jsonl "${TRAIN_JSONL}" \
      --mask-mode "${mask}" \
      --seed "${seed}" \
      --n-samples "${n}" \
      --epochs "${epochs}" \
      --lora-r "${LORA_R}" \
      --lr "${LR}" \
      --max-length "${maxlen}" \
      --gpu 0 \
      >"${LOG_DIR}/c0_gpu${gpu}_${tag}.log" 2>&1
}

run_eval() {
  local gpu="$1" tag="$2" model="$3"
  local out="${OUT_ROOT}/evals/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] eval gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] eval gpu${gpu} ${tag} model=${model}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_harness_eval.py" \
      --out "${out}" \
      --model-path "${model}" \
      --base-model "${MODEL_PATH}" \
      --gpu 0 \
      --tag "${tag}" \
      >"${LOG_DIR}/c0_gpu${gpu}_${tag}.log" 2>&1
}

run_v2v3() {
  local gpu="$1" tag="$2" model="$3"
  local out="${OUT_ROOT}/evals/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] v2v3 gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] v2v3 gpu${gpu} ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_v2v3_scorer.py" \
      --out "${out}" \
      --model-path "${model}" \
      --base-model "${MODEL_PATH}" \
      --gpu 0 \
      --n 128 \
      --seed 8141 \
      --tag "${tag}" \
      >"${LOG_DIR}/c0_gpu${gpu}_${tag}.log" 2>&1
}

run_gpu_queue() {
  local gpu="$1"
  local log="${LOG_DIR}/c0_gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} C0 start"
    wait_file "${DATA_DIR}/CLEAN_SFT_CONVERT.json" 10800 10
    wait_file "${MODEL_PATH}/model-00000-of-00002.safetensors" 14400 4792272488
    wait_file "${MODEL_PATH}/model-00001-of-00002.safetensors" 14400 4798702184
    wait_file "${MODEL_PATH}/model-00002-of-00002.safetensors" 14400 4170342232
    case "${gpu}" in
      0)
        run_sft 0 full_s42_smoke full 42 "${SMOKE_N}" "${SMOKE_EPOCHS}" "${SMOKE_MAX_LEN}"
        run_sft 0 full_s42_full full 42 0 "${FULL_EPOCHS}" "${MAX_LEN}"
        ;;
      1)
        run_sft 1 full_s43_smoke full 43 "${SMOKE_N}" "${SMOKE_EPOCHS}" "${SMOKE_MAX_LEN}"
        run_sft 1 full_s43_full full 43 0 "${FULL_EPOCHS}" "${MAX_LEN}"
        ;;
      2)
        run_sft 2 tool_s42_smoke tool 42 "${SMOKE_N}" "${SMOKE_EPOCHS}" "${SMOKE_MAX_LEN}"
        run_sft 2 tool_s42_full tool 42 0 "${FULL_EPOCHS}" "${MAX_LEN}"
        ;;
      3)
        run_sft 3 tool_s43_smoke tool 43 "${SMOKE_N}" "${SMOKE_EPOCHS}" "${SMOKE_MAX_LEN}"
        run_sft 3 tool_s43_full tool 43 0 "${FULL_EPOCHS}" "${MAX_LEN}"
        ;;
      4)
        run_eval 4 raw_harness_eval "${MODEL_PATH}"
        # Full public SFT takes ~3 days locally; do not fall back to smoke eval.
        wait_file "${OUT_ROOT}/sft/gpu0/full_s42_full/DONE" 345600
        run_eval 4 full_s42_harness_eval "${OUT_ROOT}/sft/gpu0/full_s42_full/lora_checkpoint"
        ;;
      5)
        run_v2v3 5 raw_v2v3 "${MODEL_PATH}"
        wait_file "${OUT_ROOT}/sft/gpu0/full_s42_full/DONE" 345600
        run_v2v3 5 full_s42_v2v3 "${OUT_ROOT}/sft/gpu0/full_s42_full/lora_checkpoint"
        ;;
      6)
        wait_file "${OUT_ROOT}/sft/gpu2/tool_s42_full/DONE" 345600
        CKPT="${OUT_ROOT}/sft/gpu2/tool_s42_full/lora_checkpoint"
        run_eval 6 tool_s42_harness_eval "${CKPT}"
        run_v2v3 6 tool_s42_v2v3 "${CKPT}"
        ;;
      7)
        echo "[gpu7] public-SFT audit + tool-mask + metric-v2 smoke + provenance"
        "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/prepare_clean_sft_data.py" \
          --out-dir "${OUT_ROOT}" --skip-convert \
          >"${LOG_DIR}/c0_gpu7_audit_refresh.log" 2>&1 || true
        "${PYTHON_BIN}" - <<'PY'
import json, sys
from pathlib import Path
repo = Path("/data/ppnm/Capability_Evolution/SCAPE")
sys.path.insert(0, str(repo))
from scape.training.clean_sft import tool_char_mask, parse_tool_name, CANONICAL_TOOLS, load_jsonl
out = repo / "outputs/0814_clean_mechanism"
train = out / "data/CLEAN_SFT_TRAIN.jsonl"
report = {"n": 0, "parse": 0, "tool_mask_nonempty": 0, "legal": 0}
if train.is_file():
    rows = load_jsonl(train)
    report["n"] = len(rows)
    for r in rows[:512]:
        text = r.get("response_text") or ""
        name = r.get("tool_name") or parse_tool_name(text)
        if name:
            report["parse"] += 1
        if name in CANONICAL_TOOLS:
            report["legal"] += 1
        if any(tool_char_mask(text)):
            report["tool_mask_nonempty"] += 1
    n = max(1, min(512, report["n"]))
    report["parse_rate"] = report["parse"] / n
    report["legal_rate"] = report["legal"] / n
    report["tool_mask_rate"] = report["tool_mask_nonempty"] / n
(out / "data/TOOL_MASK_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
PY
        wait_file "${MODEL_PATH}/model-00000-of-00002.safetensors" 14400 4792272488
        wait_file "${MODEL_PATH}/model-00001-of-00002.safetensors" 14400 4798702184
        wait_file "${MODEL_PATH}/model-00002-of-00002.safetensors" 14400 4170342232
        CUDA_VISIBLE_DEVICES=7 \
          "${PYTHON_BIN}" -m pytest -q "${SCAPE_ROOT}/tests/test_learnability_metrics_v2.py" \
          >"${LOG_DIR}/c0_gpu7_metric_v2_pytest.log" 2>&1 || true
        CUDA_VISIBLE_DEVICES=7 \
          "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_v2v3_scorer.py" \
            --out "${OUT_ROOT}/evals/gpu7/metric_v2_smoke" \
            --model-path "${MODEL_PATH}" \
            --gpu 0 \
            --n 16 \
            --seed 8141 \
            --tag metric_v2_smoke \
            >"${LOG_DIR}/c0_gpu7_metric_v2_smoke.log" 2>&1 || true
        "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_0814_clean.py" \
          --out-dir "${OUT_ROOT}" --phase c0 \
          >"${LOG_DIR}/c0_gpu7_aggregate.log" 2>&1 || true
        ;;
    esac
    echo "[$(date -Iseconds)] gpu${gpu} C0 ALL_DONE"
    mkdir -p "${OUT_ROOT}/sft/gpu${gpu}"
    touch "${OUT_ROOT}/sft/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

if [[ -n "${GPU_ONLY:-}" ]]; then
  run_gpu_queue "${GPU_ONLY}"
else
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu_queue "${g}" &
    echo $! >"${PID_DIR}/c0_gpu${g}.pid"
  done
  echo "[launch] C0 8-GPU under ${OUT_ROOT}"
fi
