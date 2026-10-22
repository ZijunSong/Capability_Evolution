#!/usr/bin/env bash
# Qwen3.5-27B × all (no-thinking) — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Forked from run_bcplus_w_harness1_qwen_all_gpu8_eval.sh with two changes:
#   1. MODEL_PATH -> /mnt/songzijun/models/Qwen3.5-27B (Qwen3_5ForConditionalGeneration).
#   2. Thinking disabled: each actor vLLM serve gains
#        --chat-template-kwargs {"enable_thinking":false}
#      so the Qwen3.5 chat_template takes its `enable_thinking is false` branch
#      (emits no <think> block). The verifier (harness-1) is unchanged.
#
# Default 8×GPU layout (inherited from lib):
#   6 Qwen3.5-27B actors on GPU 0–5 (ports 8042…8054)
#   harness-1 verifier on GPU 6 (port 8050)
#   eval workers --tp 64
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_qwen35_27b_all_gpu8_eval_nothinking.sh
#
# Optional env: RUN_ID, BENCHMARK, ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, PY, VLLM, MODEL_PATH, GPU_UTIL,
#   VLLM_MAX_NUM_SEQS, QWEN35_NO_THINKING
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen35_27b_all_nothink_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/Qwen3.5-27B}"
MODEL_SLUG="Qwen3.5-27B"
API_MODEL="${API_MODEL:-Qwen3.5-27B}"
COMPONENT="all"

# Disable Qwen3.5 thinking at the vLLM serve layer.
# lib's start_vllm_bg splices ${extra} UNQUOTED into the vllm serve command line,
# so the JSON value must be a single whitespace-free shell word. We therefore
# emit {"enable_thinking":false} (no inner spaces, no quoting) which word-splits
# into exactly two argv tokens: --chat-template-kwargs and {"enable_thinking":false}.
# QWEN35_NO_THINKING=0 restores the default (thinking-on) behavior.
QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"

# Qwen3.5-27B is dense 51.75GiB bf16. On gm402-80g with GPU_UTIL=0.90 the
# usable budget is ~72GiB, leaving only ~20GiB for KV. The profile default
# --max-model-len 32768 is kept (capping to 16k made BC+ multi-turn queries
# hit HTTP 400). 8 concurrent short queries use ~11% KV (observed); paged KV
# evicts long-context outliers instead of hard-failing. Override downward
# only by setting QWEN35_MAX_MODEL_LEN explicitly.
# Empty/unset → use the profile default (32768). Set to an int to override
# downward. NOTE: ${QWEN35_MAX_MODEL_LEN:-} treats an empty-string env as
# unset, so the launch script's `export QWEN35_MAX_MODEL_LEN=""` correctly
# falls through to the profile default here (do NOT use `:-16384` here, it
# would re-cap to 16384 and re-trigger HTTP 400 on long BC+ queries).
QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-}"

_override_vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local base extra
  base="$(PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" \
    -m trim.eval.model_profiles vllm-extra "${slug}")"
  # Keep the profile default max-model-len (32768). An earlier attempt capped it
  # to 16384 to protect KV headroom, but BC+ multi-turn retrieval queries can
  # accumulate >16k prompt tokens (prompt 14337 + max_new_tokens 2048 = 16385),
  # causing HTTP 400 "maximum context length is 16384". With GPU_UTIL=0.90 the
  # KV headroom (~20GiB) handles 8 concurrent short queries easily (observed
  # KV usage ~11% at 5 reqs); paged KV evicts long-context outliers instead of
  # hard-failing them. QWEN35_MAX_MODEL_LEN only overrides when explicitly set.
  if [[ -n "${QWEN35_MAX_MODEL_LEN:-}" ]]; then
    extra="$(printf '%s\n' "${base}" | sed -E "s#--max-model-len [0-9]+#--max-model-len ${QWEN35_MAX_MODEL_LEN}#")"
  else
    extra="${base}"
  fi
  if [[ "${QWEN35_NO_THINKING}" == "1" ]]; then
    # vLLM 0.19.1 serves thinking via --default-chat-template-kwargs (NOT
    # --chat-template-kwargs, which is unrecognized and errors out at serve
    # startup). This sets the engine-default chat_template_kwargs, merged with
    # request-level kwargs (request takes precedence). The upstream_api worker
    # omits chat_template_kwargs, so the serve default {"enable_thinking":false}
    # applies → Qwen3.5 chat_template takes its `enable_thinking is false` branch
    # (no  mind block). JSON is whitespace-free so ${extra} word-splits into two
    # argv tokens: --default-chat-template-kwargs and {"enable_thinking":false}.
    extra="${extra} --default-chat-template-kwargs {\"enable_thinking\":false}"
  fi
  echo "${extra}"
}

# Override AFTER sourcing lib so start_zero_stack / start_all_stack pick this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

lib_bcplus_gpu8_run_all
