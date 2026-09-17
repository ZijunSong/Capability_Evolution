#!/usr/bin/env bash
# gemma-3-27b-it × zero — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Required conda env: bishop  (python 3.11.6, torch 2.11.0+cu130, vLLM 0.25.1,
# transformers 5.14.1, pyserini 1.6.0). Do not use bishop-gemma3 / torch 2.10
# (H20-3e bf16 SIGFPE). TEMPERATURE=0.0. See gemma_all_gpu8_eval.sh header.
#
# Default layout uses every GPU that can hold a 27B actor. On a machine where
# GPU 1/2 are occupied (~24GB free), that is 6 actors on GPU 0,3,4,5,6,7.
# component=zero does not start a harness-1 verifier.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_gemma_zero_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ZERO_ACTOR_GPUS, ZERO_ACTOR_PORTS, TP,
#   GPU_UTIL, VLLM_MAX_NUM_SEQS, PY, VLLM, MODEL_PATH, VLLM_EXTRA, TEMPERATURE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gemma_zero_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/gemma-3-27b-it}"
MODEL_SLUG="gemma-3-27b-it"
API_MODEL="${API_MODEL:-gemma-3-27b-it}"
COMPONENT="zero"

# Serve + eval on bishop (torch 2.11.0+cu130, vLLM 0.25.1). bishop-gemma3's
# torch 2.10.0+cu128 SIGFPEs on H20-3e bf16 F.linear even at 5376×5376 (Gemma
# hidden size); Gemma3 also rejects --dtype float16. bishop GEMM is stable.
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
VLLM="${VLLM:-/data/ppnm/miniconda3/envs/bishop/bin/vllm}"
CONDA_ENV="$(cd "$(dirname "${PY}")/.." && pwd)"
export PATH="$(dirname "${PY}"):${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Skip GPU 1/2 by default: they currently have ~24GB free (lmdeploy).
ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,3,4,5,6,7}"
ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8052}"
TP="${TP:-64}"
GPU_UTIL="${GPU_UTIL:-0.65}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
# Gemma-3 generation_config uses top_p/top_k; temp 1.0 produced 100% implicit_user_text
# (zero searches). Greedy decoding matches the tool-call smoke that actually works.
TEMPERATURE="${TEMPERATURE:-0.0}"

lib_bcplus_gpu8_run_zero
