#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

echo "=== Round14 status $(date -Is) ==="
for gpu in 0 1 2 3 4 5 6 7; do
  case "${gpu}" in
    0) root="${OUT}/gpu0_dup_anchor" ;;
    1) root="${OUT}/gpu1_stop" ;;
    2) root="${OUT}/gpu2_verify_routing" ;;
    3) root="${OUT}/gpu3_evidence_admission" ;;
    4) root="${OUT}/gpu4_context_budget" ;;
    5) root="${OUT}/gpu5_external_verify" ;;
    6) root="${OUT}/gpu6_rollback_lite" ;;
    7) root="${OUT}/gpu7_method_ablation" ;;
  esac
  hb="missing"
  [[ -f "${root}/HEARTBEAT" ]] && hb="$(cat "${root}/HEARTBEAT" | head -1)"
  done="no"
  [[ -f "${root}/DONE" ]] && done="yes"
  gate="—"
  [[ -f "${root}/DATASET_GATE.json" ]] && gate="$(python -c "import json;g=json.load(open('${root}/DATASET_GATE.json'));print(g.get('status','?'))")"
  echo "GPU${gpu}: done=${done} hb=${hb} gate=${gate}"
done

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
