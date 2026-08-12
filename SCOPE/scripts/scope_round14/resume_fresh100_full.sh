#!/usr/bin/env bash
# Resume Dup wave0 to full 100q after shard fix; then re-aggregate.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

MANIFEST="${MANIFEST_DIR}/R14_FRESH100.json"
OUT_GPU="${OUT}/gpu0_dup_anchor"
PARALLEL="${R14_PARALLEL:-16}"

r14_log "resume_fresh100_full: clearing DONE markers"

for d in B_OFF B_ON T_OFF_seed42 T_OFF_seed43 T_OFF_seed44; do
  rm -f "${OUT_GPU}/${d}/DONE"
  # Force re-entry into rollout even if summary claims complete for 50q
  if [[ -f "${OUT_GPU}/${d}/summary.json" ]]; then
    python - <<PY
import json
from pathlib import Path
p = Path("${OUT_GPU}/${d}/summary.json")
s = json.loads(p.read_text())
if int(s.get("n_completed") or 0) < 100:
    p.rename(p.with_suffix(".json.bak50"))
    print("archived partial summary ${d}")
PY
  fi
done
rm -f "${OUT_GPU}/DONE" \
  "${OUT_GPU}/DUP_RETIREMENT_GATE.json" \
  "${OUT_GPU}/RETIREMENT_EVAL.json" \
  "${OUT_GPU}/RETIREMENT_EVAL.json" 2>/dev/null || true

# Also clear per-condition RETIREMENT_EVAL if present
rm -f "${OUT_GPU}"/*/RETIREMENT_EVAL.json

launch() {
  local gpu="$1" cond="$2" seed="$3"
  local out="${OUT_GPU}/${cond}"
  if [[ "${cond}" == "T_OFF" ]]; then
    out="${OUT_GPU}/T_OFF_seed${seed}"
  fi
  r14_log "resume full100 GPU${gpu} ${cond} seed${seed} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round14/run_module_retirement_eval.py \
    --capability duplicate_evidence \
    --manifest "${MANIFEST}" \
    --output-dir "${out}" \
    --gpu "${gpu}" \
    --seed "${seed}" \
    --conditions "${cond}" \
    --temperature 0.0 \
    --parallel "${PARALLEL}" \
    --flat-output \
    --resume \
    --run-closed-loop \
    >> "${LOG_DIR}/wave0_dup_calibrate.log" 2>&1 &
  echo $! > "${PID_DIR}/wave0_gpu${gpu}.pid"
  sleep 4
}

launch 0 B_OFF 42
launch 1 B_ON 42
launch 2 T_OFF 42
launch 3 T_OFF 43
launch 4 T_OFF 44

r14_touch_hb "${OUT_GPU}/HEARTBEAT"
r14_log "resume_fresh100_full: launched 5 jobs"

# Wait and aggregate
while true; do
  if r14_wave0_complete; then
    # Verify each has >=100 episodes
    ok=1
    for d in B_OFF B_ON T_OFF_seed42 T_OFF_seed43 T_OFF_seed44; do
      n=0
      ep="${OUT_GPU}/${d}/episodes.jsonl"
      [[ -f "${ep}" ]] && n=$(wc -l < "${ep}" | tr -d ' ')
      r14_log "check ${d} ep=${n}"
      if [[ "${n}" -lt 100 ]]; then ok=0; fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      break
    fi
    r14_log "DONE markers present but ep<100 — waiting/watchdog"
  fi
  r14_touch_hb "${OUT_GPU}/HEARTBEAT"
  sleep 120
done

python training/scope_round14/aggregate_dup_anchor.py --anchor-dir "${OUT_GPU}"
r14_log "resume_fresh100_full: aggregate complete"

# Launch parallel 830 confirm (8 shards x T_OFF only for representative seed42,
# plus B_OFF/B_ON on remaining capacity after T_OFF shards)
GATE="${OUT_GPU}/DUP_RETIREMENT_GATE.json"
if [[ -f "${GATE}" ]] && [[ "$(r14_gate_pass "${GATE}" gate_c_pass)" == "True" ]]; then
  r14_log "launching parallel 830 confirm"
  bash "$(dirname "$0")/launch_830_confirm_parallel.sh" || true
else
  r14_log "skip 830 — gate_c not pass"
fi
