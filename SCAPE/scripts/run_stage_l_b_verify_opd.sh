#!/usr/bin/env bash
# Provisional Candidate-B verify OPD on GPU2-5 (SCOPE train_opd).
# train_opd has no --vllm-port/--tensor-parallel-size: we start vLLM then pass --vllm-url.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE_ROOT="${SCOPE_ROOT:-$(cd "${SCAPE_ROOT}/../SCOPE" && pwd)}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OPD_OUT="${OPD_OUT:-${SCAPE_ROOT}/outputs/stage_l/B_verify_opd_provisional}"
OUT_CELL="${OPD_OUT}/L64_seed42"
LOG="${LOG:-${SCAPE_ROOT}/outputs/stage_l/logs/B_verify_opd.log}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
cd "${SCOPE_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="${SCOPE_ROOT}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OPD_OUT}/smoke" "${OUT_CELL}" "$(dirname "$LOG")"

# Smoke first (limit 2); smoke_opd_vllm_hf manages vLLM lifecycle
python training/smoke_opd_vllm_hf.py \
  --model-path "${MODEL_PATH}" \
  --queries-json tests/fixtures/browsecomp_sample_queries.json \
  --limit 2 \
  --max-new-tokens 24 \
  --vllm-port 8765 \
  --tensor-parallel-size 4 \
  --output-dir "${OPD_OUT}/smoke"

VLLM_URL='http://127.0.0.1:8766/v1'
VLLM_LOG="${OUT_CELL}/vllm_server.log"
VLLM_PID_FILE="${OUT_CELL}/vllm_server.pid"

cleanup_vllm() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}")
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    rm -f "${VLLM_PID_FILE}"
  fi
}
trap cleanup_vllm EXIT

nohup vllm serve "${MODEL_PATH}" \
  --served-model-name qwen \
  --host 127.0.0.1 \
  --port 8766 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  >"${VLLM_LOG}" 2>&1 &
echo $! >"${VLLM_PID_FILE}"

for i in $(seq 1 180); do
  if curl -sf "${VLLM_URL}/models" >/dev/null; then
    echo "[stageL] vLLM ready at ${VLLM_URL}"
    break
  fi
  if ! kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
    echo "[stageL] vLLM died; see ${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl -sf "${VLLM_URL}/models" >/dev/null || { echo "[stageL] vLLM not ready" >&2; exit 1; }

OUT_CELL="${OUT_CELL}" python - <<'PY'
import json
import os
from pathlib import Path
from training.opd.rollout_worker import RolloutConfig, resolve_query_records
cfg = RolloutConfig(dataset="browsecompplus", split="train", limit=64, seed=42)
recs = resolve_query_records(cfg)
out = Path(os.environ["OUT_CELL"]) / "queries_l64.json"
out.write_text(
    json.dumps([{"query_id": r.query_id, "query": r.query} for r in recs], indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"[stageL] wrote {len(recs)} queries -> {out}")
PY

rm -f "${OUT_CELL}/rollout_manifest.json"
(
  while [[ ! -f "${OUT_CELL}/rollout_manifest.json" ]]; do sleep 2; done
  echo "[stageL] rollout_manifest seen; stopping vLLM before HF train"
  cleanup_vllm
) &
WATCH_PID=$!

python training/train_opd.py \
  --model-path "${MODEL_PATH}" \
  --vllm-url "${VLLM_URL}" \
  --vllm-model-name qwen \
  --student-config harness/configs/ablate_verification.yaml \
  --teacher-config harness/configs/modules_full.yaml \
  --target-module verification \
  --queries-json "${OUT_CELL}/queries_l64.json" \
  --limit 64 \
  --seed 42 \
  --epochs 1 \
  --offline-shadow \
  --train \
  --output-dir "${OUT_CELL}"

wait "${WATCH_PID}" 2>/dev/null || true
cleanup_vllm
trap - EXIT
touch "${OPD_OUT}/DONE"
echo "[stageL] B provisional OPD DONE"
