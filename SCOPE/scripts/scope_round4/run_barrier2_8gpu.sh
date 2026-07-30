#!/usr/bin/env bash
# Round 4 Barrier 2: 8-GPU scorer consistency replay — nohup safe
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
MERGED_ROOT="${REPO_ROOT}/outputs/scope_round3/merged"
OUT_ROOT="${REPO_ROOT}/outputs/scope_round4/scorer_audit"
LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# Clear report header
echo "# Scorer Consistency Report (Round 4 Barrier 2)" > "${OUT_ROOT}/SCORE_CONSISTENCY_REPORT.md"
echo "" >> "${OUT_ROOT}/SCORE_CONSISTENCY_REPORT.md"
echo "Started: $(date -Is)" >> "${OUT_ROOT}/SCORE_CONSISTENCY_REPORT.md"
echo "" >> "${OUT_ROOT}/SCORE_CONSISTENCY_REPORT.md"

declare -a JOBS=(
  "0:Base:${BASE_MODEL}"
  "1:round3_op_main_seed42:${MERGED_ROOT}/round3_op_main_seed42"
  "2:round3_op_main_seed43:${MERGED_ROOT}/round3_op_main_seed43"
  "3:round3_op_main_seed44:${MERGED_ROOT}/round3_op_main_seed44"
  "4:round3_compact_json_sample_norm:${MERGED_ROOT}/round3_compact_json_sample_norm"
  "5:round3_op_no_balance:${MERGED_ROOT}/round3_op_no_balance"
  "6:round3_correct_only_op:${MERGED_ROOT}/round3_correct_only_op"
  "7:round3_endorse_only_op:${MERGED_ROOT}/round3_endorse_only_op"
)

PIDS=()
for entry in "${JOBS[@]}"; do
  IFS=':' read -r gpu variant model_path <<< "${entry}"
  log="${LOG_DIR}/barrier2_${variant}.log"
  echo "[barrier2] GPU${gpu} ${variant} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round4/replay_scorer_consistency.py \
    --variant "${variant}" \
    --model-path "${model_path}" \
    --output "${OUT_ROOT}/${variant}.jsonl" \
    --report "${OUT_ROOT}/SCORE_CONSISTENCY_REPORT.md" \
    --gpu 0 \
    > "${log}" 2>&1 &
  PIDS+=($!)
  sleep 2
done

echo "[barrier2] launched ${#PIDS[@]} jobs: ${PIDS[*]}"
echo "${PIDS[@]}" > "${LOG_DIR}/barrier2_pids.txt"

# Wait and merge in foreground (also nohup-able if caller wraps whole script)
wait "${PIDS[@]}" || true
echo "[barrier2] all jobs finished $(date -Is)"

# Append merge summary
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round4/scorer_audit")
lines = ["", "## Merge summary", ""]
for p in sorted(root.glob("*.summary.json")):
    s = json.loads(p.read_text())
    lines.append(
        f"- **{s['variant']}**: train-off={s['train_vs_offline_prediction_mismatch_rate']:.4f}, "
        f"off-rt={s['offline_vs_runtime_prediction_mismatch_rate']:.4f}, "
        f"prompt={s['prompt_mismatch_rate']:.4f}"
    )
report = root / "SCORE_CONSISTENCY_REPORT.md"
report.write_text(report.read_text() + "\n".join(lines) + "\n")
print("Merged summary into", report)
PY
