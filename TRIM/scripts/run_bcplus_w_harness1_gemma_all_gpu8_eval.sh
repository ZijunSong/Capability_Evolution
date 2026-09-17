#!/usr/bin/env bash
# gemma-3-27b-it × all — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Required conda env: bishop  (NOT bishop-gemma3 / torch 2.10)
#   python        3.11.6
#   torch         2.11.0+cu130
#   vllm          0.25.1
#   transformers  5.14.1
#   pyserini      1.6.0
#   jinja2        3.1.6
#
# H20-3e + torch 2.10.0+cu128 SIGFPEs on bf16 F.linear even at Gemma hidden
# size (5376×5376). Gemma3 also rejects --dtype float16. Serve and eval must
# stay on bishop. Export LD_LIBRARY_PATH to conda/lib (libicu / CXXABI).
#
# vLLM extras (from trim.eval.model_profiles GEMMA_PROFILE):
#   --enable-auto-tool-choice --tool-call-parser pythonic
#   --language-model-only --generation-config vllm
#   --chat-template trim/eval/templates/tool_chat_template_gemma3_pythonic.jinja
#
# Eval-side fix (trim.upstream_harness1.api_adapter): Gemma often writes
# [curate(add_ids=[...]importance={...})] into content with missing commas.
# Recover that as a structured curate call; do not treat it as implicit_user_text.
# TEMPERATURE must be 0.0 — temp 1.0 was 100% first-turn prose / zero searches.
#
# Default 8×GPU layout (same shape as gpt-oss all):
#   6 gemma-3-27b-it actors on GPU 0–5 (ports 8042…8054)
#   harness-1 verifier on GPU 6 (port 8050)
#   eval workers --tp 64
#
# If GPU 1/2 are occupied (~24GB free), override:
#   ALL_ACTOR_GPUS=0,3,4,5,6 ALL_ACTOR_PORTS=8040,8042,8044,8046,8048 \
#   ALL_TP=5 ALL_VERIFY_GPU=7 ALL_VERIFY_PORT=8050 \
#   bash TRIM/scripts/run_bcplus_w_harness1_gemma_all_gpu8_eval.sh
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_gemma_all_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, GPU_UTIL, VLLM_MAX_NUM_SEQS, PY, VLLM,
#   MODEL_PATH, VLLM_EXTRA, TEMPERATURE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gemma_all_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/gemma-3-27b-it}"
MODEL_SLUG="gemma-3-27b-it"
API_MODEL="${API_MODEL:-gemma-3-27b-it}"
COMPONENT="all"

PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
VLLM="${VLLM:-/data/ppnm/miniconda3/envs/bishop/bin/vllm}"
CONDA_ENV="$(cd "$(dirname "${PY}")/.." && pwd)"
export PATH="$(dirname "${PY}"):${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8052,8054}"
ALL_TP="${ALL_TP:-6}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-6}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
GPU_UTIL="${GPU_UTIL:-0.65}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"

lib_bcplus_gpu8_run_all
