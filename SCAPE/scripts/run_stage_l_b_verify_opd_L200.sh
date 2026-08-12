#!/usr/bin/env bash
# B verify OPD L200 cell: external vLLM + train_opd --train
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE_ROOT="${SCOPE_ROOT:-$(cd "${SCAPE_ROOT}/../SCOPE" && pwd)}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OPD_OUT="${OPD_OUT:-${SCAPE_ROOT}/outputs/stage_l/B_verify_opd_provisional}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-200}"
PORT="${PORT:-8769}"
TP="${TP:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
OUT_CELL="${OUT_CELL:-${OPD_OUT}/L${LIMIT}_seed${SEED}}"
TAG="L${LIMIT}_seed${SEED}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
cd "${SCOPE_ROOT}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SCOPE_ROOT}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_CELL}"
VLLM_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${OUT_CELL}/vllm_server.log"
VLLM_PID_FILE="${OUT_CELL}/vllm_server.pid"

cleanup_vllm() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid=$(cat "${VLLM_PID_FILE}")
    kill "${pid}" 2>/dev/null || true
    # also kill children (EngineCore)
    pkill -P "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    rm -f "${VLLM_PID_FILE}"
  fi
}
trap cleanup_vllm EXIT

echo "[${TAG}] starting vLLM TP=${TP} on ${PORT} GPUs=${CUDA_VISIBLE_DEVICES}"
nohup vllm serve "${MODEL_PATH}" \
  --served-model-name qwen \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  >"${VLLM_LOG}" 2>&1 &
echo $! >"${VLLM_PID_FILE}"
echo "[${TAG}] vllm pid=$(cat "${VLLM_PID_FILE}")"

for i in $(seq 1 180); do
  if curl -sf "${VLLM_URL}/models" >/dev/null; then
    echo "[${TAG}] vLLM ready (attempt ${i})"
    break
  fi
  if ! kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
    echo "[${TAG}] vLLM died; see ${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl -sf "${VLLM_URL}/models" >/dev/null || { echo "[${TAG}] vLLM not ready" >&2; exit 1; }

OUT_CELL="${OUT_CELL}" LIMIT="${LIMIT}" SEED="${SEED}" python - <<'PY'
import json, os
from pathlib import Path
from training.opd.rollout_worker import RolloutConfig, resolve_query_records
limit = int(os.environ["LIMIT"]); seed = int(os.environ["SEED"])
cfg = RolloutConfig(dataset="browsecompplus", split="train", limit=limit, seed=seed)
recs = resolve_query_records(cfg)
out = Path(os.environ["OUT_CELL"]) / f"queries_l{limit}.json"
out.write_text(
    json.dumps([{"query_id": r.query_id, "query": r.query} for r in recs], indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"[L{limit}_seed{seed}] wrote {len(recs)} queries -> {out}")
PY

QUERIES_JSON="${OUT_CELL}/queries_l${LIMIT}.json"
rm -f "${OUT_CELL}/rollout_manifest.json"
(
  while [[ ! -f "${OUT_CELL}/rollout_manifest.json" ]]; do sleep 2; done
  echo "[${TAG}] rollout_manifest seen; stopping vLLM before HF train"
  cleanup_vllm
) &
WATCH_PID=$!

echo "[${TAG}] starting train_opd limit=${LIMIT} seed=${SEED} --train"
python training/train_opd.py \
  --model-path "${MODEL_PATH}" \
  --vllm-url "${VLLM_URL}" \
  --vllm-model-name qwen \
  --student-config harness/configs/ablate_verification.yaml \
  --teacher-config harness/configs/modules_full.yaml \
  --target-module verification \
  --queries-json "${QUERIES_JSON}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --epochs 1 \
  --offline-shadow \
  --train \
  --output-dir "${OUT_CELL}"

wait "${WATCH_PID}" 2>/dev/null || true
cleanup_vllm
trap - EXIT
echo "[${TAG}] DONE"
