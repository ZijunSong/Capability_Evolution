#!/usr/bin/env bash
# Official Harness-1 vLLM server launcher for SCAPE (localhost only).
# User authorized `--trust-remote-code` for external model source `pat-jj/harness-1`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS="$ROOT/external/harness-1"
MODEL="${HARNESS1_HF_MODEL:-/mnt/songzijun/models/pat-jj_harness-1-full/harness-1}"
PORT="${HARNESS1_VLLM_PORT:-8000}"
TP="${HARNESS1_TP:-8}"
PYTHON_BIN="${SCAPE_PYTHON:-/root/miniforge3/envs/scape/bin/python}"
VLLM_CLI="${SCAPE_VLLM_CLI:-/root/miniforge3/envs/scape/bin/vllm}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ ! -x "$VLLM_CLI" ]]; then
  VLLM_CLI="$(command -v vllm || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "python binary not found; set SCAPE_PYTHON" >&2
  exit 1
fi
if [[ -z "$VLLM_CLI" || ! -x "$VLLM_CLI" ]]; then
  echo "vllm binary not found; set SCAPE_VLLM_CLI" >&2
  exit 1
fi
export PATH="/root/miniforge3/envs/scape/bin:${PATH:-}"
export LD_LIBRARY_PATH="/root/miniforge3/envs/scape/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/mnt/songzijun/hf_cache}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
cd "$HARNESS"
exec "$PYTHON_BIN" "$VLLM_CLI" serve "$MODEL" \
  --served-model-name harness-1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --moe-backend triton

