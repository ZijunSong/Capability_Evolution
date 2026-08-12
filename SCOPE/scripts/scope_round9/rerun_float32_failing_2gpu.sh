#!/usr/bin/env bash
# Targeted float32 repair for Wave A Barrier A failures:
#   GPU6 -> rollback_correct_only
#   GPU7 -> rollback_soft_replan_only
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

scope9_log "FLOAT32 repair launch: correct_only@GPU6 soft_replan@GPU7"

# Clear stale DONE markers for the two failing variants only.
rm -f "${MARKER_DIR}/wave_a_rollback_correct_only.DONE"
rm -f "${MARKER_DIR}/wave_a_rollback_soft_replan_only.DONE"

nohup env CUDA_VISIBLE_DEVICES=6 PYTHONPATH="${REPO_ROOT}" \
  python training/scope_round9/rerun_float32_failing_variants.py \
  --variant rollback_correct_only --full-on-fail \
  > "${LOG_DIR}/float32_repair_rollback_correct_only.log" 2>&1 &
echo $! > "${PID_DIR}/float32_repair_correct_only.pid"
scope9_log "started correct_only pid=$(cat "${PID_DIR}/float32_repair_correct_only.pid")"

nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${REPO_ROOT}" \
  python training/scope_round9/rerun_float32_failing_variants.py \
  --variant rollback_soft_replan_only --full-on-fail \
  > "${LOG_DIR}/float32_repair_rollback_soft_replan_only.log" 2>&1 &
echo $! > "${PID_DIR}/float32_repair_soft_replan.pid"
scope9_log "started soft_replan pid=$(cat "${PID_DIR}/float32_repair_soft_replan.pid")"

cat > "${LOG_DIR}/float32_repair_monitor.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/ppnm/Capability_Evolution/SCOPE"
LOG="$ROOT/outputs/scope_round9/logs/float32_repair_monitor.log"
cd "$ROOT"
while true; do
  {
    echo "==== $(date -Is) ===="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 6,7 || true
    for v in rollback_correct_only rollback_soft_replan_only; do
      m=pending
      [[ -f "$ROOT/outputs/scope_round9/markers/wave_a_${v}.DONE" ]] && m=DONE
      echo "$v marker=$m"
      tail -3 "$ROOT/outputs/scope_round9/logs/float32_repair_${v}.log" 2>/dev/null || true
    done
  } | tee -a "$LOG"
  if ! pgrep -f "rerun_float32_failing_variants.py" >/dev/null; then
    echo "workers finished $(date -Is)" | tee -a "$LOG"
    for v in rollback_correct_only rollback_soft_replan_only; do
      python - <<PY | tee -a "$LOG"
import json
from pathlib import Path
p=Path("outputs/scope_round9/wave_a/$v/WAVE_A_REPORT.json")
d=json.loads(p.read_text()) if p.exists() else {}
print("$v barrier_a_pass=", d.get("barrier_a_pass"), "failures=", d.get("split_failures"))
diag=Path("outputs/scope_round9/wave_a/$v/FLOAT32_REPAIR_DIAG.json")
print(diag.read_text()[:3000] if diag.exists() else "no diag")
PY
    done
    break
  fi
  sleep 120
done
EOS
chmod +x "${LOG_DIR}/float32_repair_monitor.sh"
nohup bash "${LOG_DIR}/float32_repair_monitor.sh" \
  > "${LOG_DIR}/float32_repair_monitor_launcher.log" 2>&1 &
echo $! > "${PID_DIR}/float32_repair_monitor.pid"
scope9_log "monitor pid=$(cat "${PID_DIR}/float32_repair_monitor.pid")"
scope9_log "Logs: ${LOG_DIR}/float32_repair_*.log"
