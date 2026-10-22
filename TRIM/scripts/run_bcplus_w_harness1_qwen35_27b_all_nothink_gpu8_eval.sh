#!/usr/bin/env bash
# Qwen3.5-27B × all (no-thinking) — bcplus_full (830 queries), upstream_api + local_bm25.
#
# Forked from run_bcplus_w_harness1_qwen_all_gpu8_eval.sh (lib all layout) with the
# Qwen3.5-27B no-thinking fixes ported from the VERIFIED zero run
# run_bcplus_full_qwen35_27b_zero_gpu8_eval.sh (job t-20260914151008-fcanl, 830/830 ok,
# Recall 32.0%). The lib's all layout (6 Qwen3.5-27B actors on GPU 0–5 + harness-1
# verifier on GPU 6, eval workers --tp 64) is kept; only the model-family serve flags
# and thinking handling are corrected for Qwen3.5-27B + no-thinking.
#
# Critical fix vs the 2026-09-14 14:47 draft (run_bcplus_w_harness1_qwen35_27b_all_gpu8_eval_nothinking.sh):
#   that draft overrode vllm_extra_for_model to APPEND --default-chat-template-kwargs
#   {"enable_thinking":false} but did NOT strip --reasoning-parser qwen3. That is exactly
#   the recovery1 smoke-failure root cause: with qwen3_coder + reasoning_parser=qwen3 and
#   thinking off, the reasoning_parser swallows the tool-call text into the `reasoning`
#   field → tool_calls=[] content=null. This wrapper ports the zero run's fix instead:
#
#   1. Thinking is OFF via the model chat_template itself (already edited in place on
#      /mnt/songzijun/models/Qwen3.5-27B: chat_template.jinja + tokenizer_config.json
#      embedded field, both with the `enable_thinking is not true` guard). The default
#      (worker does not pass enable_thinking) takes the no-thinking branch. vLLM 0.19.1
#      serve CLI does NOT support --chat-template-kwargs (errors out), so we rely on the
#      template default, NOT on a serve flag. Backups: *.orig_thinking_on_20260914_141539.
#   2. --reasoning-parser qwen3 is STRIPPED (no-thinking has no standalone reasoning
#      block → reasoning_parser only mis-routes tool-call text). Only --tool-call-parser
#      qwen3_coder is kept.
#
# Dense 27B (51.75GiB bf16) on gm402-80g needs GPU_UTIL=0.90 (~72GiB usable) to hold
# weights + KV per single-GPU actor; lib default 0.70 is too tight. MAX_NUM_SEQS=8
# matches the verified zero run. Both are forced here.
#
# Default 8×GPU layout (overridden via env to fully utilize ALL 8 cards — the lib
# default 6 actors + verify on GPU 6 leaves GPU 7 idle. The gpt-oss-20B all precedent
# uses 7 actors + 1 verify = 8 cards. Match that here):
#   7 Qwen3.5-27B actors on GPU 0–6 (ports 8042,8044,8046,8048,8050,8052,8054)
#   harness-1 verifier on GPU 7 (port 8056)
#   eval workers --tp 64
# All 8 GPUs carry a model (7× Qwen3.5-27B @ ~67GiB + 1× harness-1 verify).
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh
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

# Thinking is disabled in the model chat_template itself (see header). QWEN35_NO_THINKING=1
# is a marker that gates the --reasoning-parser strip below (the actual switch lives in the
# template). QWEN35_NO_THINKING=0 restores thinking-on (requires restoring the template
# backup AND re-adding --reasoning-parser qwen3 — not the no-thinking path used here).
QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"

# 27B dense 51.75GiB on gm402-80g: override lib defaults (GPU_UTIL 0.70, VLLM_MAX_NUM_SEQS
# 128) to the verified zero-run values. GPU_UTIL 0.90 → ~72GiB usable; 51.1GiB weights +
# KV headroom ~20GiB. MAX_NUM_SEQS 8 → KV ~16GiB (only 16 full_attention layers carry
# O(ctx) KV; ~2GiB/seq @32k). These are read by lib_bcplus_w_harness1_gpu8_eval.sh.
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
# Leave max-model-len at the profile default 32768. An earlier 16384 cap caused HTTP 400 on
# BC+ multi-turn queries (prompt 14337 + max_new_tokens 2048 = 16385). KV headroom ~20GiB
# handles 8 concurrent short queries (observed ~11% at 5 reqs); long outliers evict via
# paged KV instead of hard-failing. Set QWEN35_MAX_MODEL_LEN only to override downward.
# max-model-len: raised to 65536 (was 32768) to eliminate HTTP 400 infra_errors. The
# 32768 run (recovery t-20260915110217-kqa4v) hit 34/830 queries where BC+ multi-turn
# agent loops accumulate prompt to exactly 30721 tokens; 30721 + max_new_tokens 2048
# = 32769 > 32768 by 1 token → ConfigError infra_error (reward=0). Actual prompts stay
# ~30k (not tens of k), so 65536 is far above the real ceiling and KV headroom is fine
# (vLLM allocates KV by actual tokens, not max_model_len; paged KV evicts outliers).
# Lower MAX_NUM_SEQS 8→6 to give 64k KV pool headroom (vLLM may reserve more per-seq).
# NOTE: this diverges max_model_len from fcanl zero (32768), but zero's 32768 was
# incidental (16-worker low concurrency never hit the 32k edge), not a deliberate
# setting — accepted as the all-vs-zero comparison caliber (user decision 2026-09-16).
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-65536}"

_override_vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local base extra
  base="$(PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" \
    -m trim.eval.model_profiles vllm-extra "${slug}")"
  extra="${base}"
  # Keep the profile default max-model-len (32768) unless explicitly overridden.
  if [[ -n "${QWEN35_MAX_MODEL_LEN:-}" ]]; then
    extra="$(printf '%s\n' "${extra}" | sed -E "s#--max-model-len [0-9]+#--max-model-len ${QWEN35_MAX_MODEL_LEN}#")"
  fi
  if [[ "${QWEN35_NO_THINKING}" == "1" ]]; then
    # CRITICAL (ported from the verified zero run): no-thinking mode STRIPS
    # --reasoning-parser qwen3. recovery1 smoke failed because reasoning_parser=qwen3
    # swallowed the tool-call text into the `reasoning` field → tool_calls=[] content=null.
    # no-thinking has no standalone reasoning block, so reasoning_parser is unneeded and
    # harmful. Only --tool-call-parser qwen3_coder is kept; tool-call text stays in content.
    # Thinking itself is off via the model chat_template default (NOT a serve flag — vLLM
    # 0.19.1 serve CLI does not support --chat-template-kwargs).
    extra="${extra// --reasoning-parser qwen3/}"
  fi
  echo "${extra}"
}

# Override AFTER sourcing lib so start_all_stack picks this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

lib_bcplus_gpu8_run_all
