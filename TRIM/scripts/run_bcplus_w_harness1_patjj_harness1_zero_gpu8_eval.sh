#!/usr/bin/env bash
# pat-jj/harness-1 (gpt-oss-20b arch, bf16 dequant) × zero — bcplus_full (830 queries),
# upstream_api + local_bm25.
#
# Clone of run_bcplus_w_harness1_gptoss_zero_gpu8_eval.sh with model identity changed to
# pat-jj/harness-1 release checkpoint (gpt-oss-20b MoE bf16 dequant, no quantization_config).
# Same lib code path as gpt-oss-20b (vllm_extra_for_model keys off MODEL_SLUG="harness-1"
# → --tool-call-parser openai --max-model-len 32768 --moe-backend triton), but correct slug/
# path/served-name for provenance. No LoRA, no mxfp4 Marlin PTX risk.
#
# Default 8×GPU layout: multi-8-actor (ZERO_ACTOR_GPUS=0..7, 8 actors × 4 workers = TP 32,
# see entry override; was single-actor GPU7 + TP64 which lost queries).
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_patjj_harness1_zero_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, GPU, TP, ZERO_ACTOR_PORT, PY, VLLM, MODEL_PATH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_patjj_harness1_zero_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/pat-jj_harness-1-full/harness-1}"
MODEL_SLUG="harness-1"
API_MODEL="${API_MODEL:-harness-1}"
COMPONENT="zero"

lib_bcplus_gpu8_run_zero
