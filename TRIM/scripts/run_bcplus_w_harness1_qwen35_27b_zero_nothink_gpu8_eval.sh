#!/usr/bin/env bash
# Qwen3.5-27B × zero (no-thinking) — 4-benchmark (bcplus_full/longsealqa/frames/hotpotqa),
# upstream_api + local_bm25. Thin lib-sourcing wrapper (mirrors the all wrapper).
#
# WHY THIS EXISTS: the existing 27B zero wrapper
# (run_bcplus_full_qwen35_27b_zero_gpu8_eval.sh) is a STANDALONE script that hardcodes
# --benchmark bcplus_full + BrowseComp-Plus paths and does NOT source the lib, so it
# cannot route longsealqa/frames/hotpotqa to transfer_local indexes. This thin wrapper
# sources lib_bcplus_w_harness1_gpu8_eval.sh (which has the case-BENCHMARK transfer
# routing) so the zero path gains 4-benchmark support for free, matching the all path
# and the Qwen3-4B pattern (run_bcplus_w_harness1_qwen_zero_gpu8_eval.sh).
#
# Forked from run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh:
#   - COMPONENT all -> zero
#   - lib_bcplus_gpu8_run_all -> lib_bcplus_gpu8_run_zero (lib:388, reads ZERO_ACTOR_GPUS /
#     ZERO_ACTOR_PORTS / TP)
#   - the exact same vllm_extra_for_model override (strip --reasoning-parser qwen3 when
#     QWEN35_NO_THINKING=1) is kept verbatim — that fix is required for no-think tool calls
#     regardless of component (reasoning_parser=qwen3 swallows tool-call text → tool_calls=[]
#     content=null; see all wrapper header for the recovery1 root-cause history).
#
# Thinking is OFF via the model chat_template itself (already edited on
# /mnt/songzijun/models/Qwen3.5-27B: chat_template.jinja + tokenizer_config.json embedded
# field, both with `enable_thinking is not true` guard; backups *.orig_thinking_on_*).
# vLLM 0.19.1 serve CLI does NOT support --chat-template-kwargs, so we rely on the template
# default, NOT a serve flag. QWEN35_NO_THINKING=1 only gates the reasoning-parser strip.
#
# Dense 27B (51.75GiB bf16) on gm402-80g needs GPU_UTIL=0.90 (~72GiB usable). MAX_MODEL_LEN
# 65536 (same as all) eliminates BC+ long-trajectory HTTP 400 infra_error. MAX_NUM_SEQS=6.
#
# Default 8×GPU layout for zero (overridden via env by the unified entry to multi-8-actor,
# matching the all 64-worker caliber so zero/all are directly comparable):
#   8 Qwen3.5-27B actors on GPU 0–7 (ports 8040,8042,8044,8046,8048,8050,8052,8054), no
#   verifier (zero has no verify component). eval workers --tp 64.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_w_harness1_qwen35_27b_zero_nothink_gpu8_eval.sh
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_qwen35_27b_zero_nothink_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/Qwen3.5-27B}"
MODEL_SLUG="Qwen3.5-27B"
API_MODEL="${API_MODEL:-Qwen3.5-27B}"
COMPONENT="zero"

# Thinking is disabled in the model chat_template itself (see header). QWEN35_NO_THINKING=1
# is a marker that gates the --reasoning-parser strip below.
QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"

# 27B dense 51.75GiB on gm402-80g: override lib defaults to the verified values (same as
# all wrapper). GPU_UTIL 0.90 → ~72GiB usable; MAX_NUM_SEQS 6 + MAX_MODEL_LEN 65536.
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
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
    # CRITICAL (ported from the verified zero run fcanl + all wrapper): no-thinking mode
    # STRIPS --reasoning-parser qwen3. reasoning_parser=qwen3 swallows tool-call text into
    # the `reasoning` field → tool_calls=[] content=null. no-thinking has no standalone
    # reasoning block, so reasoning_parser is unneeded and harmful. Only --tool-call-parser
    # qwen3_coder is kept. Thinking itself is off via the model chat_template default.
    extra="${extra// --reasoning-parser qwen3/}"
  fi
  echo "${extra}"
}

# Override AFTER sourcing lib so start_zero_stack picks this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

lib_bcplus_gpu8_run_zero
