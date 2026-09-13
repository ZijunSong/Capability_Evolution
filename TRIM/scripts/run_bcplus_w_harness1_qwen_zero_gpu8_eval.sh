#!/usr/bin/env bash
# Qwen3-4B-Instruct × zero — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Default 8×GPU layout: single actor on GPU 7 (port 8040), eval workers --tp 64.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_qwen_zero_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, GPU, TP, ZERO_ACTOR_PORT, PY, VLLM, MODEL_PATH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_zero_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/Qwen3-4B-Instruct-2507}"
MODEL_SLUG="Qwen3-4B-Instruct-2507"
API_MODEL="${API_MODEL:-Qwen3-4B-Instruct-2507}"
COMPONENT="zero"

lib_bcplus_gpu8_run_zero
