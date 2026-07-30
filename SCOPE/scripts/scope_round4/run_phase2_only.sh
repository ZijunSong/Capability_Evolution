#!/usr/bin/env bash
# Phase 2 only: B4 overfit128 (GPU0) + closed-loop retry (GPU1-4, max 4 vLLM)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
MERGED_ROOT="${REPO_ROOT}/outputs/scope_round3/merged"
OUT_ROOT="${REPO_ROOT}/outputs/scope_round4/postfix_replay"
LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
mkdir -p "${LOG_DIR}"

echo "[phase2] $(date -Is) start"

# B4 on GPU0 (background)
BARRIER4_GPU=0 nohup bash scripts/scope_round4/run_barrier4_overfit128.sh \
  > "${LOG_DIR}/barrier4_nohup.log" 2>&1 &
B4_PID=$!
echo "[phase2] B4 overfit128 PID=${B4_PID} on GPU0"

run_one_shard() {
  local gpu=$1 variant=$2 model_path=$3 port=$4 shard=$5
  local out="${OUT_ROOT}/closed_loop/${variant}/${shard}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "[skip] ${variant}/${shard}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[cl] GPU${gpu} ${variant}/${shard} port${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${model_path}" \
    --vllm-port "${port}" \
    --dup-operation \
    --parallel 1
}

launch_wave() {
  local -a jobs=("$@")
  local pids=() delay=0
  for job in "${jobs[@]}"; do
    IFS=':' read -r gpu variant model_path port shard <<< "${job}"
    (
      sleep "${delay}"
      run_one_shard "${gpu}" "${variant}" "${model_path}" "${port}" "${shard}" \
        > "${LOG_DIR}/phase2_cl_${variant}_${shard}.log" 2>&1
    ) &
    pids+=($!)
    delay=$((delay + 75))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

echo "[phase2] closed-loop wave 1 (GPU1-4, 75s stagger)"
launch_wave \
  "1:base:${BASE_MODEL}:9100:shard1" \
  "2:compact_json:${MERGED_ROOT}/round3_compact_json_sample_norm:9110:shard0" \
  "3:op_seed42:${MERGED_ROOT}/round3_op_main_seed42:9120:shard0" \
  "4:op_seed43:${MERGED_ROOT}/round3_op_main_seed43:9130:shard1"

echo "[phase2] closed-loop wave 2 (GPU1-2)"
launch_wave \
  "1:op_seed42:${MERGED_ROOT}/round3_op_main_seed42:9140:shard1" \
  "2:op_seed44:${MERGED_ROOT}/round3_op_main_seed44:9150:shard0"

wait "${B4_PID}" || true

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round4/postfix_replay")
lines = ["# Barrier 3 Retry Summary", "", f"Updated: {__import__('datetime').datetime.now().isoformat()}", ""]
lines.append("## Offline")
for p in sorted((root / "offline").glob("*.json")):
    d = json.loads(p.read_text())
    lines.append(f"- **{p.stem}**: macro_f1={d.get('macro_f1',0):.3f} acc={d.get('operation_accuracy',0):.3f}")
lines.append("", "## Closed-loop")
for vdir in sorted((root / "closed_loop").iterdir()):
    if not vdir.is_dir():
        continue
    for sdir in sorted(vdir.iterdir()):
        ok = (sdir / "summary.json").exists()
        lines.append(f"- {vdir.name}/{sdir.name}: {'OK' if ok else 'MISSING'}")
(root / "BARRIER3_RETRY_REPORT.md").write_text("\n".join(lines) + "\n")
b4 = Path("outputs/scope_round4/overfit128/overfit128_report.json")
if b4.exists():
    r = json.loads(b4.read_text())
    passed = r.get("pass_criteria", {}).get("B4_PASS", False)
    Path("outputs/scope_round4/B4_PASS").write_text(str(passed) + "\n")
    print(f"B4_PASS={passed}")
print("Wrote", root / "BARRIER3_RETRY_REPORT.md")
PY

echo "[phase2] complete $(date -Is)"
