#!/usr/bin/env bash
# GLM-4-32B-0414 × zero — Harness-1 upstream_api + local_bm25.
#
# Required conda env: bishop  (NOT bishop-glm4 / torch 2.10)
#   python        3.11.6
#   torch         2.11.0+cu130
#   vllm          0.25.1
#   transformers  5.14.1
#   pyserini      1.6.0
#   jinja2        3.1.6
#
# H20-3e + torch 2.10.0+cu128 SIGFPEs on bf16 F.linear; GLM-4-32B hidden is
# 6144. Serve and eval must stay on bishop. GLM-4 rejects fp16; leave dtype
# at checkpoint bf16. Native tool format is ``name\\n{json}`` (glm4_0414
# plugin + buried-call recovery). Temperature 0. See glm_all_gpu8_eval.sh.
#
# component=zero does not start a harness-1 verifier. Default layout when
# GPU 1/2 are occupied (~24GB free, cannot hold 32B):
#   6 GLM-4-32B-0414 actors on GPU 0,3,4,5,6,7 (ports 8040…8052)
#   eval workers --tp 64
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_glm_zero_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ZERO_ACTOR_GPUS, ZERO_ACTOR_PORTS, TP,
#   GPU_UTIL, VLLM_MAX_NUM_SEQS, PY, VLLM, MODEL_PATH, VLLM_EXTRA, TEMPERATURE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_glm_zero_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/GLM-4-32B-0414}"
MODEL_SLUG="GLM-4-32B-0414"
API_MODEL="${API_MODEL:-GLM-4-32B-0414}"
COMPONENT="zero"
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
# zero has no verifier, so GPU 6 is an extra actor vs the all layout.
ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,3,4,5,6,7}"
ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8052}"
TP="${TP:-64}"
GPU_UTIL="${GPU_UTIL:-0.65}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
export MAX_FORMAT_RETRIES="${MAX_FORMAT_RETRIES:-5}"

lib_bcplus_gpu8_run_zero
