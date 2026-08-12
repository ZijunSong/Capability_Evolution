#!/usr/bin/env bash
# Long-running H20 orchestrator: monitor LOO -> aggregate -> Stage L queues.
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_LOO="${SCAPE_ROOT}/outputs/local_cal64_loo"
OUT_L="${SCAPE_ROOT}/outputs/stage_l"
LOG="${SCAPE_ROOT}/outputs/ORCHESTRATOR.log"
mkdir -p "${SCAPE_ROOT}/outputs"

# shellcheck disable=SC1091
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop

echo "[$(date -Iseconds)] orchestrator start" | tee -a "$LOG"

# Ensure LOO queues running
if [[ ! -f "${OUT_LOO}/ALL_DONE" ]]; then
  bash "${SCAPE_ROOT}/scripts/launch_local_cal64_loo_8gpu.sh" >>"$LOG" 2>&1 || true
fi

while true; do
  bash "${SCAPE_ROOT}/scripts/monitor_scape_loo.sh" >>"$LOG" 2>&1 || true

  if [[ -f "${OUT_LOO}/ALL_DONE" ]]; then
    echo "[$(date -Iseconds)] LOO complete — aggregate + select" | tee -a "$LOG"
    cd "${SCAPE_ROOT}"
    python scripts/aggregate_loo_and_select.py >>"$LOG" 2>&1 || {
      echo "[$(date -Iseconds)] aggregate failed" | tee -a "$LOG"
      sleep 120
      continue
    }

    # Stage L dry scaffolding queues (real HF OPD wired when samples ready)
    OUT_ROOT="${OUT_L}" bash "${SCAPE_ROOT}/scripts/launch_stage_l_8gpu.sh" >>"$LOG" 2>&1 || true

    # Kick micro-distill dry-runs for A/B on free GPUs if candidates exist
    if [[ -f "${SCAPE_ROOT}/outputs/scape_prestage/CANDIDATE_SELECTION.json" ]]; then
      python - <<'PY' >>"$LOG" 2>&1
import json
from pathlib import Path
from scape.training.train_tool_opd import run_micro_distill
root = Path("/data/ppnm/Capability_Evolution/SCAPE")
sel = json.loads((root/"outputs/scape_prestage/CANDIDATE_SELECTION.json").read_text())
base = "/data/ppnm/models/Qwen2.5-7B-Instruct"
for label, meta in sel.items():
    cid = meta["component_id"]
    for seed in (42, 43):
        for n in (512, 2000, 8000):
            out = root/f"outputs/stage_l/{label}_L{n}_s{seed}"
            if (out/"summary.json").exists():
                continue
            print("distill", label, cid, n, seed, flush=True)
            run_micro_distill(
                output_dir=out,
                component_id=cid,
                n_samples=n,
                seed=seed,
                base_checkpoint=base,
                d_pre=1.0,
                dry_run=True,
            )
print("stage_l_dry_runs_done", flush=True)
PY
    fi

    echo "[$(date -Iseconds)] prestage+stageL scaffolding complete; sleeping (waiting H100 imports / real OPD data)" | tee -a "$LOG"
    # Keep monitoring for stuck leftovers then idle
    sleep 300
    continue
  fi

  # If queue runners died but work remains, relaunch
  dead=0
  for g in 0 1 2 3 4 5 6 7; do
    pf="${OUT_LOO}/pids/gpu${g}_queue.pid"
    if [[ ! -f "$pf" ]] || ! kill -0 "$(cat "$pf")" 2>/dev/null; then
      dead=1
    fi
  done
  if [[ "$dead" -eq 1 && ! -f "${OUT_LOO}/ALL_DONE" ]]; then
    echo "[$(date -Iseconds)] relaunching GPU queues" | tee -a "$LOG"
    bash "${SCAPE_ROOT}/scripts/launch_local_cal64_loo_8gpu.sh" >>"$LOG" 2>&1 || true
  fi

  sleep 120
done
