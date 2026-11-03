#!/usr/bin/env bash
# Qwen3.5-27B + LoRA adapter eval (zero component, 4-bench x 3-seed).
# ADAPTER = zero-rl 200-step (wmimn, r=8 alpha=16, q/k/v/o_proj),
#   OUT=qwen35-27b-rl-ddp-200step-zero-recall-32gpu_20261023_r6/checkpoints/rl/step_000200/adapter
#
# THIN entry: injects vLLM --enable-lora via VLLM_EXTRA, then execs the VERIFIED
# base 27B entry (entry_qwen35_27b_harness_zero_all_bcplus_transfer_eval_8gpu.sh,
# base 3-seed 24/24) unchanged. This preserves two hard-won properties for free:
#   1. no-think --reasoning-parser qwen3 strip (the base wrapper's
#      vllm_extra_for_model override; WITHOUT it the parser swallows tool-call
#      text -> tool_calls=[] -> silent 0% recall);
#   2. the eval-lock four-pack (tf4.57.6 etc). The adapter loads via vLLM LoRA
#      (no HF AutoModel), so the training-side tf5.17 bump does NOT apply here —
#      evals stay comparable with the locked base 3-seed numbers.
# LoRA naming: module name == served name == API_MODEL "Qwen3.5-27B", the same
#   pattern the GPT-OSS adapter eval wrapper proved (--served-model-name X
#   --lora-modules X=adapter) — every request hits the adapter.
set -euo pipefail

JFS_ROOT="${JFS_ROOT:-/mnt/songzijun}"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
: "${ADAPTER_PATH:=${JFS_ROOT}/outputs/qwen35-27b-rl-ddp-200step-zero-recall-32gpu_20261023_r6/checkpoints/rl/step_000200/adapter}"
: "${EVAL_MODE:=zero}"
export EVAL_MODE

test -f "${ADAPTER_PATH}/adapter_config.json"
test -f "${ADAPTER_PATH}/adapter_model.safetensors"
test -f "${JFS_ROOT}/models/Qwen3.5-27B/config.json"

LORA_RANK=$(python3 -c "import json;print(int(json.load(open('${ADAPTER_PATH}/adapter_config.json'))['r']))")
MAX_LORA_RANK=$(( LORA_RANK * 2 ))
LORA_NAME="Qwen3.5-27B"
export VLLM_EXTRA="--enable-lora --lora-modules ${LORA_NAME}=${ADAPTER_PATH} --max-lora-rank ${MAX_LORA_RANK}"
echo "[adapter-eval] ADAPTER_PATH=${ADAPTER_PATH} r=${LORA_RANK} max_lora_rank=${MAX_LORA_RANK}"
echo "[adapter-eval] VLLM_EXTRA=${VLLM_EXTRA}"

exec bash "${TRIM_ROOT}/scripts/entry_qwen35_27b_harness_zero_all_bcplus_transfer_eval_8gpu.sh"
