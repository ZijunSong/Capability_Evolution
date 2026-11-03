#!/usr/bin/env bash
# Qwen3.5-9B × custom --component mask (no-thinking) — 4-benchmark
# (bcplus_full/longsealqa/frames/hotpotqa), upstream_api + local_bm25.
# Thin lib-sourcing wrapper for the 2026-10-24 component-ablation batch:
#   ESP   = subtractive_curation,importance_tagging,adaptive_rerank_instruction
#   CCSO  = sentence_compress,content_dedup,evidence_graph
#   RIEVE = auto_populate_first_search,chunk_neighbors
#   MSC   = verify_tool,token_budget_marker
#
# Forked from run_bcplus_w_harness1_qwen35_9b_zero_nothink_gpu8_eval.sh:
#   - COMPONENT hardcoded "zero" -> "${COMPONENT:-zero}" (entry --env COMPONENT=... wins;
#     the 2026-10-06 lesson: entries must never hardcode COMPONENT over submit env).
#   - layout branch: verify_tool_enabled (COMPONENT==all OR comma token == verify_tool)
#     -> 7 actor + harness-1 verifier stack via lib_bcplus_gpu8_run_all (MSC only);
#     otherwise the 8-actor no-verifier stack via lib_bcplus_gpu8_run_zero (esp/ccso/rieve).
#     The lib run_eval_cell verify gate was broadened the same way (MSC patch 2026-10-03
#     had only fixed the adapter wrapper; the shared lib gate was still `== "all"`).
#   - the exact same vllm_extra_for_model override (strip --reasoning-parser qwen3 when
#     QWEN35_NO_THINKING=1) is kept verbatim — required for no-think tool calls regardless
#     of component.
#
# Thinking is OFF via the model chat_template itself (already edited on
# /mnt/songzijun/models/Qwen3.5-9B). QWEN35_NO_THINKING=1 only gates the parser strip.
# Dense 9B (~18GiB bf16) on gm402-80g: GPU_UTIL=0.90, MAX_MODEL_LEN=65536,
# MAX_NUM_SEQS=6 — identical to the 9B formal zero entry caliber (64-worker) so
# component results are directly comparable with that baseline's 3-seed means.
#
# Usage (normally driven by entry_qwen35_9b_harness_component_ablation_eval_8gpu.sh):
#   COMPONENT=ccso BENCHMARK=frames EVAL_SEED=42 RUN_ID=... \
#     bash TRIM/scripts/run_bcplus_w_harness1_qwen35_9b_component_nothink_gpu8_eval.sh
#
# Optional env: RUN_ID, BENCHMARK, COMPONENT, EVAL_SEED, ZERO_ACTOR_GPUS, ZERO_ACTOR_PORTS,
#   ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU, ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP,
#   TP, PY, VLLM, MODEL_PATH, GPU_UTIL, VLLM_MAX_NUM_SEQS, QWEN35_NO_THINKING,
#   QWEN35_MAX_MODEL_LEN
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_qwen35_9b_component_nothink_gpu8}"
MODEL_PATH="${MODEL_PATH:-../../models/Qwen3.5-9B}"
MODEL_SLUG="Qwen3.5-9B"
API_MODEL="${API_MODEL:-Qwen3.5-9B}"
# Respects submit --env COMPONENT (canonical lowercase comma list). Never hardcode.
COMPONENT="${COMPONENT:-zero}"

# Thinking is disabled in the model chat_template itself (see header). QWEN35_NO_THINKING=1
# is a marker that gates the --reasoning-parser strip below.
QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"

# 9B dense ~18GiB on gm402-80g: override lib defaults to the verified values (same as
# the 9B zero entry). GPU_UTIL 0.90 → ~72GiB usable; MAX_NUM_SEQS 6 + MAX_MODEL_LEN
# 65536. 64 eval workers (TP / ALL_EVAL_TP = 64) keeps the baseline caliber.
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

# Override AFTER sourcing lib so start_zero_stack/start_all_stack pick this up.
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

if verify_tool_enabled; then
  # MSC (or all): 7 Qwen3.5-9B actors (GPU 0..6) + harness-1 verifier (GPU 7). The entry
  # exports ALL_* layout vars; defaults below mirror the 27B-batch all layout (7+1).
  export ALL_TP="${ALL_TP:-7}"
  export ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
  export ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5,6}"
  export ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8050,8052,8054}"
  export ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-7}"
  export ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8056}"
  echo "[wrapper] Qwen3.5-9B component=${COMPONENT} verify_tool=ON -> 7 actor + verifier stack"
  lib_bcplus_gpu8_run_all
else
  # esp/ccso/rieve/zero: 8 Qwen3.5-9B actors (GPU 0..7), no verifier.
  export ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
  export ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"
  export TP="${TP:-64}"
  echo "[wrapper] Qwen3.5-9B component=${COMPONENT} verify_tool=OFF -> 8-actor stack"
  lib_bcplus_gpu8_run_zero
fi
