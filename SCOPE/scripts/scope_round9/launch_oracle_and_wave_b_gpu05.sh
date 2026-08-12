#!/usr/bin/env bash
# Round 9: oracle factorization + Wave B on free GPUs 0–5.
# GPU 6/7 are reserved for concurrent Wave A float32 repair
# (correct_only / soft_replan); their Wave B variants are queued after repair.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

NOTE="${OUT}/diagnosis/DEFERRED_WAVE_A_FAIL_VARIANTS.md"
mkdir -p "${OUT}/diagnosis"

cat > "${NOTE}" <<'EOF'
# Deferred Wave A FAIL variants (parallel track)

While main-seed oracle + Wave B run on GPU 0–5, GPU 6/7 run float32 Barrier A repair for:

- `rollback_correct_only`
- `rollback_soft_replan_only`

These are **ablation / control** Wave A variants, not the three main hierarchical seeds.
Main Wave A seeds (`rollback_o7_seed{42,43,44}`) already have Barrier A DONE markers.

Wave B variants that need GPU 6/7 are queued by this launcher after float32 workers exit:

- GPU6 → `rollback_hier_no_candidate_summary_seed42`
- GPU7 → `rollback_hier_prompt_hint_seed42`
EOF

scope9_log "ORACLE: factorization on rollback_o7_seed42 base_live HF replay"
REPLAY="${OUT}/wave_a/rollback_o7_seed42/base_live/hf_replay.jsonl"
if [[ ! -f "${REPLAY}" ]]; then
  scope9_log "ERROR: missing ${REPLAY}"
  exit 1
fi
python training/scope_round9/aggregate_oracle_factorization.py \
  --replay "${REPLAY}" \
  --output "${OUT}/diagnosis/ROOT_CAUSE_DECISION.json" \
  2>&1 | tee "${LOG_DIR}/oracle_factorization.log"
cp -f "${OUT}/diagnosis/ROOT_CAUSE_DECISION.json" "${OUT}/ROOT_CAUSE_DECISION.json"

python - <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path("outputs/scope_round9/ROOT_CAUSE_DECISION.json").read_text())
d = r.get("diagnosis", {})
print(json.dumps(d, indent=2))
if "labels_candidates_or_aggregator" in d.get("primary_bottlenecks", []):
    print("ORACLE SANITY FAIL", file=sys.stderr)
    sys.exit(3)
print("ORACLE OK")
PY

scope9_log "WAVE B: ensure hier dataset gate"
python training/scope_round9/build_hier_sdi_dataset.py 2>&1 | tee "${LOG_DIR}/build_hier_sdi.log"
python - <<'PY'
import json, sys
from pathlib import Path
gate = json.loads(Path("artifacts/datasets/scope_round9/hier_sdi/DATASET_GATE.json").read_text())
print(json.dumps(gate, indent=2))
if not gate.get("gate_pass"):
    sys.exit(2)
if gate.get("candidate_coverage", 0) < 0.99:
    print("coverage < 0.99", file=sys.stderr)
    sys.exit(2)
PY

scope9_log "WAVE B: launching GPUs 0-5 (main seeds + available ablations)"
for gpu in 0 1 2 3 4 5; do
  nohup bash "$(dirname "$0")/run_wave_b_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/wave_b_${WAVE_B_VARIANTS[$gpu]}_worker.log" 2>&1 &
  echo $! > "${PID_DIR}/wave_b_gpu${gpu}.pid"
  scope9_log "Wave B started gpu=${gpu} variant=${WAVE_B_VARIANTS[$gpu]} pid=$(cat "${PID_DIR}/wave_b_gpu${gpu}.pid")"
  sleep 2
done

# Waiter: after float32 repair exits, launch Wave B on GPU 6/7.
nohup bash -c '
set -euo pipefail
source /data/ppnm/Capability_Evolution/SCOPE/scripts/scope_round9/_common.sh
scope9_setup
scope9_log "WAVE B waiter: waiting for float32 repair on GPU6/7 to finish"
# Exclude this waiter itself: its cmdline also contains the pattern string.
while pgrep -af "rerun_float32_failing_variants.py" | grep -v "wave_b_gpu67\|bash -c" >/dev/null; do
  sleep 120
done
scope9_log "WAVE B waiter: float32 repair finished; launching GPU6/7"
for gpu in 6 7; do
  nohup bash /data/ppnm/Capability_Evolution/SCOPE/scripts/scope_round9/run_wave_b_gpu.sh "${gpu}" \
    >> "${LOG_DIR}/wave_b_${WAVE_B_VARIANTS[$gpu]}_worker.log" 2>&1 &
  echo $! > "${PID_DIR}/wave_b_gpu${gpu}.pid"
  scope9_log "Wave B started gpu=${gpu} variant=${WAVE_B_VARIANTS[$gpu]} pid=$(cat "${PID_DIR}/wave_b_gpu${gpu}.pid")"
  sleep 2
done
# Wait all 8 DONE then offline gate
while true; do
  done_n=0
  for v in "${WAVE_B_VARIANTS[@]}"; do
    [[ -f "${OUT}/wave_b/${v}/DONE" ]] && done_n=$((done_n + 1))
  done
  scope9_log "WAVE B waiter: DONE ${done_n}/8"
  if [[ "${done_n}" -ge 8 ]]; then
    python training/scope_round9/check_offline_gate_round9.py \
      --output "${OUT}/OFFLINE_GATE_ROUND9.json" \
      2>&1 | tee -a "${LOG_DIR}/offline_gate_round9.log" || true
    scope9_log "WAVE B all complete; offline gate written"
    break
  fi
  # If GPU0-5 unfinished for a long time, keep waiting
  sleep 180
done
' > "${LOG_DIR}/wave_b_gpu67_waiter.log" 2>&1 &
echo $! > "${PID_DIR}/wave_b_gpu67_waiter.pid"
scope9_log "WAVE B GPU6/7 waiter pid=$(cat "${PID_DIR}/wave_b_gpu67_waiter.pid")"

# Monitor for GPU0-5
nohup bash -c '
ROOT=/data/ppnm/Capability_Evolution/SCOPE
LOG=$ROOT/outputs/scope_round9/logs/wave_b_gpu05_monitor.log
cd "$ROOT"
while true; do
  {
    echo "==== $(date -Is) ===="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    for v in rollback_hier_o7_seed42 rollback_hier_o7_seed43 rollback_hier_o7_seed44 rollback_flat_o7_seed42_repro rollback_operation_only_seed42 rollback_checkpoint_ranker_seed42; do
      s=pending; [[ -f outputs/scope_round9/wave_b/$v/DONE ]] && s=DONE
      echo "wave_b $v $s"
    done
    echo -n "float32_repair "; pgrep -f rerun_float32_failing_variants.py >/dev/null && echo running || echo idle
  } | tee -a "$LOG"
  n=0
  for v in rollback_hier_o7_seed42 rollback_hier_o7_seed43 rollback_hier_o7_seed44 rollback_flat_o7_seed42_repro rollback_operation_only_seed42 rollback_checkpoint_ranker_seed42; do
    [[ -f outputs/scope_round9/wave_b/$v/DONE ]] && n=$((n+1))
  done
  [[ "$n" -ge 6 ]] && echo "GPU0-5 Wave B all DONE" | tee -a "$LOG" && break
  sleep 180
done
' > "${LOG_DIR}/wave_b_gpu05_monitor_launcher.log" 2>&1 &
echo $! > "${PID_DIR}/wave_b_gpu05_monitor.pid"

scope9_log "Launched oracle + Wave B on GPU0-5; GPU6/7 queued after float32 repair"
scope9_log "Note: ${NOTE}"
