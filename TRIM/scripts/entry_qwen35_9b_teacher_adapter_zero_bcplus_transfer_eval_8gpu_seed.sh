#!/usr/bin/env bash
# Qwen3.5-9B + LoRA adapter eval (teacher-ablation 50-step, student=zero, 4-bench x 3-seed).
# ADAPTER = teacher-ablation 50step scape_seed/step_000050/adapter（esp/ccso/trieve/msc 四联盟之一，
#   由 submit --env ADAPTER_PATH 注入；student=zero → EVAL_MODE=zero）。
#
# THIN entry（fork entry_qwen35_27b_adapter_zr200_zero_bcplus_transfer_eval_8gpu_seed.sh）：
# 注入 vLLM --enable-lora 后原样 exec 已验证的 9B base entry
# （entry_qwen35_9b_harness_zero_bcplus_full_eval_8gpu_20261021.sh），保留：
#   1. no-think --reasoning-parser qwen3 strip（base wrapper 的 vllm_extra_for_model
#      override；不保留会吞 tool call → tool_calls=[] → 静默 0% recall）；
#   2. eval-lock 四件套（tf4.57.6 等）。adapter 走 vLLM LoRA（非 HF AutoModel），
#      训练侧 tf5.17 不影响此处，与 base 3-seed 口径可比。
# LoRA naming: module name == served name == API_MODEL "Qwen3.5-9B"
#   （同 GPT-OSS / 27B zr200 adapter eval 验证过的 --lora-modules 模式）。
set -euo pipefail

JFS_ROOT="${JFS_ROOT:-/mnt/songzijun}"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
: "${ADAPTER_PATH:?FATAL: ADAPTER_PATH is required (teacher-ablation step_000050/adapter)}"
: "${EVAL_MODE:=zero}"
export EVAL_MODE

test -f "${ADAPTER_PATH}/adapter_config.json"
test -f "${ADAPTER_PATH}/adapter_model.safetensors"
test -f "${JFS_ROOT}/models/Qwen3.5-9B/config.json"

LORA_RANK=$(python3 -c "import json;print(int(json.load(open('${ADAPTER_PATH}/adapter_config.json'))['r']))")
MAX_LORA_RANK=$(( LORA_RANK * 2 ))
LORA_NAME="Qwen3.5-9B"
export VLLM_EXTRA="--enable-lora --lora-modules ${LORA_NAME}=${ADAPTER_PATH} --max-lora-rank ${MAX_LORA_RANK}"
echo "[adapter-eval] ADAPTER_PATH=${ADAPTER_PATH} r=${LORA_RANK} max_lora_rank=${MAX_LORA_RANK}"
echo "[adapter-eval] VLLM_EXTRA=${VLLM_EXTRA}"

exec bash "${TRIM_ROOT}/scripts/entry_qwen35_9b_harness_zero_bcplus_full_eval_8gpu_20261021.sh"
