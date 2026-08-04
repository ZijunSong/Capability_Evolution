#!/usr/bin/env bash
# Rerun shard1 25q (parity fix) on one GPU: live -> HF/vLLM replay -> compare -> gate
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

export SCOPE7_NO_RESUME=1
export PARALLEL="${PARALLEL:-64}"

GPU="${1:?usage: run_rerun_shard1_worker.sh <gpu> <base|seed42|seed43|seed44>}"
VARIANT="${2:?}"

LIVE_ROOT="${OUT}/contract_trace/live_rerun"
PORT=$((9200 + GPU))

case "${VARIANT}" in
  base)
    SEED=0
    MODEL="${BASE_MODEL}"
    LABEL="base"
    TAG="base_shard1_tau0_rerun"
    LIVE="${LIVE_ROOT}/base_shard1_tau0"
    ;;
  seed42)
    SEED=42
    MODEL="${R5}/merged/o7_r64_seed42"
    LABEL="o7_r64_seed42"
    TAG="o7_seed42_shard1_rerun"
    LIVE="${LIVE_ROOT}/o7_seed42_shard1_tau0"
    ;;
  seed43)
    SEED=43
    MODEL="${R5}/merged/o7_r64_seed43"
    LABEL="o7_r64_seed43"
    TAG="o7_seed43_shard1_rerun"
    LIVE="${LIVE_ROOT}/o7_seed43_shard1_tau0"
    ;;
  seed44)
    SEED=44
    MODEL="${R5}/merged/o7_r64_seed44"
    LABEL="o7_r64_seed44"
    TAG="o7_seed44_shard1_rerun"
    LIVE="${LIVE_ROOT}/o7_seed44_shard1_tau0"
    ;;
  *)
    echo "unknown variant ${VARIANT}" >&2
    exit 1
    ;;
esac

scope7_log "RERUN worker GPU${GPU} variant=${VARIANT}"
scope7_run_live "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" shard1 "${SEED}" "${LABEL}" "${TAG}"
scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "${TAG}"

if scope7_gate_passed "${LIVE}"; then
  scope7_log "Gate passed for ${VARIANT}; running holdout shard2"
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0_rerun/${VARIANT}_shard2" "${MODEL}" "${PORT}" shard2 "${SEED}" "${LABEL}" "${TAG}_shard2"
  if [[ "${VARIANT}" == "base" ]] || [[ "${VARIANT}" == "seed44" ]]; then
    scope7_run_live "${GPU}" "${OUT}/holdout_tau0_rerun/${VARIANT}_shard3" "${MODEL}" "${PORT}" shard3 "${SEED}" "${LABEL}" "${TAG}_shard3"
  fi
else
  scope7_write_marker "${TAG}_holdout_skipped" "failed" "${LIVE}" 0
fi

scope7_log "RERUN worker GPU${GPU} ${VARIANT} complete"
