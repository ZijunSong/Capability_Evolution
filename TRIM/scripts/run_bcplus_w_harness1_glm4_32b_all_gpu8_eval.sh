#!/usr/bin/env bash
# GLM-4-32B-0414 × all — bcplus_full (830 queries), upstream_api + local_bm25.
#
# This wrapper is a clone of run_bcplus_w_harness1_qwen_all_gpu8_eval.sh with the model
# identity changed to GLM-4-32B-0414 (THUDM GLM-4, 32B dense, bf16, no quantization).
# It sources lib_bcplus_w_harness1_gpu8_eval.sh and calls lib_bcplus_gpu8_run_all, which
# runs the GLM-4-32B-0414 component=all cell only (6 single-card actor vLLMs GPU 0-5 +
# 1 harness-1 verifier GPU 6).
#
# Model profile: MODEL_SLUG="GLM-4-32B-0414" is matched by trim.eval.model_profiles
# (_GLM_RE -> GLM45_PROFILE), so vllm_extra_for_model yields:
#   --enable-auto-tool-choice --tool-call-parser glm45 --max-model-len 32768 --trust-remote-code
# GLM-4-32B-0414 ships a plain conversational chat_template (no thinking branch), so
# thinking is NOT enabled — the model answers directly, satisfying the "no thinking" request.
# No mxfp4+LoRA Marlin PTX risk: this checkpoint is bf16 dense with no quantization_config
# and eval loads no LoRA adapter.
#
# Default 8xGPU layout (lib defaults, unchanged):
#   6 single-card actor vLLMs (GPU 0-5, ports 8042/8044/8046/8048/8052/8054)
#   harness-1 verifier vLLM (GPU 6, port 8050). GPU 7 unused for the all pass.
#   ALL_TP=6, ALL_EVAL_TP=64. The verifier (VERIFY_MODEL_PATH=../../models/harness-1) and the
#   GLM-4-32B-0414 actor are distinct vLLM processes on disjoint GPUs/ports with distinct
#   served-model-names (actor served="GLM-4-32B-0414", verifier served="harness-1-verifier").
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_glm4_32b_all_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, PY, VLLM, MODEL_PATH, GPU_UTIL, WORKER_STAGGER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_glm4_32b_all_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/GLM-4-32B-0414}"
MODEL_SLUG="GLM-4-32B-0414"
API_MODEL="${API_MODEL:-GLM-4-32B-0414}"
COMPONENT="all"

lib_bcplus_gpu8_run_all
