#!/usr/bin/env bash
# Qwen3.5-9B × zero (no-thinking) — 4-benchmark (bcplus_full/longsealqa/frames/hotpotqa),
# upstream_api + local_bm25. Thin lib-sourcing wrapper.
#
# FORKED FROM run_bcplus_w_harness1_qwen35_27b_zero_nothink_gpu8_eval.sh — identical logic
# (upstream_api path + vllm_extra_for_model override that STRIPS --reasoning-parser qwen3
# while KEEPING --tool-call-parser qwen3_coder), only model path/slug + GPU params adapted
# for 9B 18GiB.
#
# WHY THIS EXISTS (the legacy_local H0 failure root cause): run_h0_zero_rollout.py uses the
# legacy_local eval path whose vllm_rollout_worker.py does NOT pass --tool-call-parser, so
# Qwen3.5 tool calls parse to 0 (n_parse_ok=0, tool_names=unknown, recall=0). The
# upstream_api path here passes --tool-call-parser qwen3_coder via the lib's
# vllm_extra_for_model → tool calls parse correctly → real recall. This is the verified
# path (27B base eval 24/24 completed with real numbers via this same wrapper).
#
# Thinking is OFF via the model chat_template itself (already edited on
# /mnt/songzijun/models/Qwen3.5-9B: chat_template.jinja + tokenizer_config.json, both with
# `enable_thinking is not true` guard; backups *.orig_thinking_on_20261021_112520).
# QWEN35_NO_THINKING=1 gates the reasoning-parser strip.
#
# Dense 9B (~18GiB bf16) on gm402-80g: GPU_UTIL 0.90 → ~72GiB usable (ample, 18G weights +
# ~50G KV); MAX_MODEL_LEN 65536 (same as 27B, eliminates BC+ long-trajectory HTTP 400
# infra_error); MAX_NUM_SEQS 32 (9B smaller, can afford more concurrent than 27B's 6).
#
# Default 8×GPU layout for zero: 8 Qwen3.5-9B actors on GPU 0–7 (ports
# 8040,8042,8044,8046,8048,8050,8052,8054), no verifier. eval workers --tp 64.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_qwen35_9b_zero_nothink_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, ZERO_ACTOR_GPUS, ZERO_ACTOR_PORTS, TP, PY, VLLM,
#   MODEL_PATH, GPU_UTIL, VLLM_MAX_NUM_SEQS, QWEN35_NO_THINKING, QWEN35_MAX_MODEL_LEN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# When this wrapper is copied to /tmp (to dodge JuiceFS client-cache lag) and exec'd
# from there, BASH_SOURCE[0] resolves to /tmp and the sibling lib is not found. Fall
# back to the canonical JuiceFS scripts dir in that case.
if [[ ! -f "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh" ]]; then
  SCRIPT_DIR="/mnt/songzijun/Capability_Evolution/TRIM/scripts"
fi
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_qwen35_9b_zero_nothink_gpu8}"
MODEL_PATH="${MODEL_PATH:-/mnt/songzijun/models/Qwen3.5-9B}"
MODEL_SLUG="Qwen3.5-9B"
API_MODEL="${API_MODEL:-Qwen3.5-9B}"
COMPONENT="zero"

# Thinking is disabled in the model chat_template itself. QWEN35_NO_THINKING=1 gates the
# --reasoning-parser strip below.
QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"

# 9B dense ~18GiB on gm402-80g: GPU_UTIL 0.90 → ~72GiB usable (ample, 18G weights + ~50G KV).
# MAX_MODEL_LEN 65536 (same as 27B, avoids BC+ long-trajectory HTTP 400 infra_error).
# MAX_NUM_SEQS 32 (9B smaller than 27B's 6, can afford more concurrent).
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-65536}"

_override_vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  local base extra
  base="$(PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" \
    -m trim.eval.model_profiles vllm-extra "${slug}")"
  extra="${base}"
  # Override max-model-len if explicitly set (profile default is 32768).
  if [[ -n "${QWEN35_MAX_MODEL_LEN:-}" ]]; then
    extra="$(printf '%s\n' "${extra}" | sed -E "s#--max-model-len [0-9]+#--max-model-len ${QWEN35_MAX_MODEL_LEN}#")"
  fi
  if [[ "${QWEN35_NO_THINKING}" == "1" ]]; then
    # CRITICAL (ported from the verified 27B zero run): no-thinking mode STRIPS
    # --reasoning-parser qwen3. reasoning_parser=qwen3 swallows tool-call text into the
    # `reasoning` field → tool_calls=[] content=null. no-thinking has no standalone
    # reasoning block, so reasoning_parser is unneeded and harmful. Only --tool-call-parser
    # qwen3_coder is kept (THIS is what makes tool calls parse correctly). Thinking itself
    # is off via the model chat_template default.
    extra="${extra// --reasoning-parser qwen3/}"
  fi
  echo "${extra}"
}

# Override AFTER sourcing lib so start_zero_stack picks this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

lib_bcplus_gpu8_run_zero
