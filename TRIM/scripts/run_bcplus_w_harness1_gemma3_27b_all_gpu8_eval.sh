#!/usr/bin/env bash
# Gemma-3-27B-it × all — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Forked from run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh. The lib's
# all layout (Qwen actors on GPU 0–5 + harness-1 verifier on GPU 6, eval workers
# --tp 64) is kept but overridden via env to fully utilize ALL 8 cards — the lib
# default 6 actors + verify on GPU 6 leaves GPU 7 idle. The gpt-oss-20B all
# precedent (job t-20260911222501-dda5w, 830/830 ok) uses 7 actors + 1 verify =
# 8 cards. Match that here.
#
# Gemma-3-27B-it specifics (vs Qwen3.5-27B):
#   - No thinking mode. Gemma3 IT chat_template has NO enable_thinking guard and
#     emits no standalone reasoning block, so there is nothing to strip:
#     GEMMA_PROFILE (trim.eval.model_profiles) carries NO reasoning_parser, only
#     tool_call_parser="pythonic" + needs_tool_smoke_test=True. Contrast
#     Qwen3.5-27B which had to STRIP --reasoning-parser qwen3 in no-thinking mode.
#   - Dense 27B (config text_config: hidden 5376, 62 layers, bf16, ~12 shards
#     ~50 GiB) on gm402-80g needs GPU_UTIL=0.90 (~72 GiB usable) to hold weights +
#     KV per single-GPU actor; lib default 0.70 is too tight. MAX_NUM_SEQS=8
#     matches the verified Qwen3.5-27B all run.
#   - VLM (Gemma3ForConditionalGeneration, siglip vision encoder); BC+ runs the
#     pure-text path. vLLM 0.19.1 Gemma3 VLM support is the preflight risk probed
#     in the boot script (ModelRegistry.get_supported_archs()).
#   - tokenizer: SentencePiece (tokenizer.model + tokenizer.json), NOT harmony
#     o200k; the harness import chain still imports openai_harmony so the boot
#     script extracts the tiktoken bundle (o200k_base) for that.
#
# Default 8×GPU layout (overridden via env to fully utilize ALL 8 cards):
#   7 Gemma-3-27B-it actors on GPU 0–6 (ports 8042,8044,8046,8048,8050,8052,8054)
#   harness-1 verifier on GPU 7 (port 8056)
#   eval workers --tp 64
# All 8 GPUs carry a model (7× Gemma-3-27B-it @ ~67GiB + 1× harness-1 verify).
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_gemma3_27b_all_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, PY, VLLM, MODEL_PATH, GPU_UTIL,
#   VLLM_MAX_NUM_SEQS, GEMMA3_MAX_MODEL_LEN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# When this wrapper is copied to /tmp (to dodge JuiceFS client-cache lag) and
# exec'd from there, BASH_SOURCE[0] resolves to /tmp and the sibling lib script
# is not found. Fall back to the canonical JuiceFS scripts dir in that case.
if [[ ! -f "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh" ]]; then
  SCRIPT_DIR="/mnt/songzijun/Capability_Evolution/TRIM/scripts"
fi
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gemma3_27b_all_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/gemma-3-27b-it}"
MODEL_SLUG="gemma-3-27b-it"
API_MODEL="${API_MODEL:-gemma-3-27b-it}"
COMPONENT="all"

# 27B dense (~50 GiB bf16) on gm402-80g: override lib defaults (GPU_UTIL 0.70,
# VLLM_MAX_NUM_SEQS 128) to the verified Qwen3.5-27B all-run values. GPU_UTIL
# 0.90 → ~72GiB usable; weights ~50GiB + KV headroom ~20GiB. MAX_NUM_SEQS 8 →
# KV ~16GiB. These are read by lib_bcplus_w_harness1_gpu8_eval.sh.
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
# Leave max-model-len at the profile default 32768 (set by vllm_extra_for_model).
# An earlier 16384 cap on Qwen3.5-27B caused HTTP 400 on BC+ multi-turn queries.
# Set GEMMA3_MAX_MODEL_LEN only to override downward.
export GEMMA3_MAX_MODEL_LEN="${GEMMA3_MAX_MODEL_LEN:-}"

_override_vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local base extra
  base="$(PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" \
    -m trim.eval.model_profiles vllm-extra "${slug}")"
  extra="${base}"
  # Keep the profile default max-model-len (32768) unless explicitly overridden.
  if [[ -n "${GEMMA3_MAX_MODEL_LEN:-}" ]]; then
    extra="$(printf '%s\n' "${extra}" | sed -E "s#--max-model-len [0-9]+#--max-model-len ${GEMMA3_MAX_MODEL_LEN}#")"
  fi
  # NOTE: Gemma3 has no thinking mode — GEMMA_PROFILE has no reasoning_parser, so
  # there is nothing to strip (unlike Qwen3.5-27B no-thinking which stripped
  # --reasoning-parser qwen3). tool_call_parser stays at the profile "pythonic".
  echo "${extra}"
}

# Override AFTER sourcing lib so start_all_stack picks this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

lib_bcplus_gpu8_run_all
