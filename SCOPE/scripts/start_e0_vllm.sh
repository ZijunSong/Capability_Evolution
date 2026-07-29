#!/usr/bin/env bash
# Start vLLM for E0 distillability experiments (Qwen2.5-7B-Instruct)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8776}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-e0-harness-policy}"
LOG="${LOG:-$REPO_ROOT/outputs/scope_e0_distillability/vllm_server.log}"
PID_FILE="${PID_FILE:-$REPO_ROOT/outputs/scope_e0_distillability/vllm_server.pid}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TP_SIZE="${TP_SIZE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

mkdir -p "$(dirname "${LOG}")"

if [[ -f "${PID_FILE}" ]]; then
  vpid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${vpid}" ]] && kill -0 "${vpid}" 2>/dev/null; then
    if python - <<PY
import urllib.request, sys
try:
    with urllib.request.urlopen("http://127.0.0.1:${VLLM_PORT}/v1/models", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    then
      echo "vLLM already running pid=${vpid} port=${VLLM_PORT}"
      exit 0
    fi
    echo "Stale vLLM pid=${vpid}; restarting ..."
    kill "${vpid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

echo "Starting vLLM on port ${VLLM_PORT} ..."
nohup vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

python - <<PY
import time, urllib.request, sys
url = "http://127.0.0.1:${VLLM_PORT}/v1/models"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("vLLM ready:", url, flush=True)
                sys.exit(0)
    except Exception:
        time.sleep(3)
print("vLLM failed; see ${LOG}", flush=True)
sys.exit(1)
PY
