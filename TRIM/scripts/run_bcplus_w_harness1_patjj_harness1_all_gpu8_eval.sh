#!/usr/bin/env bash
# pat-jj/harness-1 (gpt-oss-20b arch, bf16 dequant) × all — bcplus_full (830 queries),
# upstream_api + local_bm25.
#
# This wrapper is a clone of run_bcplus_w_harness1_gptoss_all_gpu8_eval.sh with the model
# identity changed to the pat-jj/harness-1 release checkpoint. That checkpoint is a gpt-oss-20b
# MoE already dequantized to bf16 (config.json has NO quantization_config), so it hits the same
# lib_bcplus_w_harness1_gpu8_eval.sh code path as gpt-oss-20b (vllm_extra_for_model keys off
# MODEL_SLUG="harness-1" → --tool-call-parser openai --max-model-len 32768 --moe-backend triton),
# but does NOT touch the mxfp4+LoRA Marlin PTX failure mode (no quantization_config, no LoRA in
# eval). The gpt-oss-20b wrapper hardcodes MODEL_SLUG="gpt-oss-20b" (no env override), so we
# cannot reuse it for a DIFFERENT checkpoint without mislabeling the output dir / provenance;
# this wrapper sets the correct slug/path/served-name instead.
#
# Default 8×GPU layout (lib defaults, unchanged):
#   6 single-card actor vLLMs (GPU 0-5, ports 8042/8044/8046/8048/8052/8054)
#   harness-1 verifier vLLM (GPU 6, port 8050). GPU 7 unused for the all pass.
#   ALL_TP=6, ALL_EVAL_TP=64. NOTE: the verifier (VERIFY_MODEL_PATH=../../models/harness-1)
#   and the actor both point at the same pat-jj/harness-1 release checkpoint here
#   (models/harness-1 is a symlink to pat-jj_harness-1-full/harness-1). They are distinct vLLM
#   processes on disjoint GPUs/ports with distinct served-model-names
#   (actor served="harness-1", verifier served="harness-1-verifier"), so there is no aliasing.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_patjj_harness1_all_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, PY, VLLM, MODEL_PATH, GPU_UTIL, WORKER_STAGGER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_patjj_harness1_all_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/pat-jj_harness-1-full/harness-1}"
MODEL_SLUG="harness-1"
API_MODEL="${API_MODEL:-harness-1}"
COMPONENT="all"

lib_bcplus_gpu8_run_all
