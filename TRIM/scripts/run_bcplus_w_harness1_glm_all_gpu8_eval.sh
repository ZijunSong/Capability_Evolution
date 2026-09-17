#!/usr/bin/env bash
# GLM-4-32B-0414 × all — Harness-1 upstream_api + local_bm25.
#
# Required conda env: bishop  (NOT bishop-glm4 / torch 2.10)
#   python        3.11.6
#   torch         2.11.0+cu130
#   vllm          0.25.1
#   transformers  5.14.1
#   pyserini      1.6.0
#   jinja2        3.1.6
#
# H20-3e + torch 2.10.0+cu128 SIGFPEs on bf16 F.linear at Gemma hidden size
# (5376×5376); GLM-4-32B hidden is 6144. Serve and eval must stay on bishop.
# Export LD_LIBRARY_PATH to conda/lib (libicu / CXXABI). GLM-4 rejects fp16
# (vLLM glm4 numerical-instability guard); leave dtype at checkpoint bf16.
#
# vLLM extras (from trim.eval.model_profiles GLM0414_PROFILE):
#   --enable-auto-tool-choice --tool-call-parser glm4_0414
#   --tool-parser-plugin trim/eval/tool_parsers/glm4_0414_tool_parser.py
#   --max-model-len 32768 --trust-remote-code
# Native GLM-4-0414 tool format is ``name\\n{json}``; stock glm45 expects
# GLM-4.5/4.7 XML and drops those calls into content.
# Eval recovery also accepts buried ``name\\n{json}`` / pythonic calls inside
# planning prose, and retries later-turn natural language (search→curate).
# Temperature 0: GLM-4-0414 otherwise narrates plans instead of calling tools.
#
# Default 8×GPU layout when GPU 1/2 are occupied (~24GB free, cannot hold 32B):
#   5 GLM-4-32B-0414 actors on GPU 0,3,4,5,7 (ports 8040…8048)
#   harness-1 verifier on GPU 6 (port 8050)
#   eval workers --tp 50 for bcplus_test_50 (override ALL_EVAL_TP for full 830)
#
# Usage:
#   BENCHMARK=bcplus_test_50 bash TRIM/scripts/run_bcplus_w_harness1_glm_all_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, GPU_UTIL, VLLM_MAX_NUM_SEQS, PY, VLLM,
#   MODEL_PATH, VLLM_EXTRA, TEMPERATURE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_glm_all_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/GLM-4-32B-0414}"
MODEL_SLUG="GLM-4-32B-0414"
API_MODEL="${API_MODEL:-GLM-4-32B-0414}"
COMPONENT="all"
BENCHMARK="${BENCHMARK:-bcplus_full}"

# Serve + eval on bishop (torch 2.11.0+cu130, vLLM 0.25.1). bishop-glm4's
# torch 2.10.0+cu128 SIGFPEs on H20-3e bf16 F.linear.
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
VLLM="${VLLM:-/data/ppnm/miniconda3/envs/bishop/bin/vllm}"
CONDA_ENV="$(cd "$(dirname "${PY}")/.." && pwd)"
export PATH="$(dirname "${PY}"):${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Skip GPU 1/2 by default: they currently have ~24GB free (lmdeploy).
ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,3,4,5,7}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8040,8042,8044,8046,8048}"
ALL_TP="${ALL_TP:-5}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-6}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_EVAL_TP="${ALL_EVAL_TP:-50}"
GPU_UTIL="${GPU_UTIL:-0.65}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
export MAX_FORMAT_RETRIES="${MAX_FORMAT_RETRIES:-5}"

lib_bcplus_gpu8_run_all
