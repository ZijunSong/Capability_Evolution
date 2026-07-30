#!/usr/bin/env bash
# Retry Barrier 3: fixed offline eval + staggered closed-loop (max 4 concurrent vLLM)
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
mkdir -p "${OUT_ROOT}/offline" "${OUT_ROOT}/closed_loop" "${LOG_DIR}"

# --- Offline eval (8 GPU, fixed script) ---
echo "[b3-retry] $(date -Is) offline eval 8-way"
declare -a OFFLINE=(
  "0:Base"
  "1:round3_compact_json"
  "2:round3_op_seed42"
  "3:round3_op_seed43"
  "4:round3_op_seed44"
  "5:round3_op_no_balance"
  "6:round3_correct_only"
  "7:round3_endorse_only"
)
OFFLINE_PIDS=()
for entry in "${OFFLINE[@]}"; do
  IFS=':' read -r gpu variant <<< "${entry}"
  log="${LOG_DIR}/b3retry_offline_${variant}.log"
  echo "[b3-retry-offline] GPU${gpu} ${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round4/run_postfix_offline_eval.py \
    --variant "${variant}" \
    > "${log}" 2>&1 &
  OFFLINE_PIDS+=($!)
  sleep 3
done
wait "${OFFLINE_PIDS[@]}" || true
echo "[b3-retry] offline eval done"

# --- Closed-loop: one shard per GPU, max 4 concurrent ---
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
  local pids=()
  local delay=0
  for job in "${jobs[@]}"; do
    IFS=':' read -r gpu variant model_path port shard <<< "${job}"
    (
      sleep "${delay}"
      run_one_shard "${gpu}" "${variant}" "${model_path}" "${port}" "${shard}" \
        > "${LOG_DIR}/b3retry_cl_${variant}_${shard}.log" 2>&1
    ) &
    pids+=($!)
    delay=$((delay + 75))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

echo "[b3-retry] Wave 1 closed-loop (4 shards, 75s stagger)"
launch_wave \
  "0:base:${BASE_MODEL}:9100:shard1" \
  "1:compact_json:${MERGED_ROOT}/round3_compact_json_sample_norm:9110:shard0" \
  "2:op_seed42:${MERGED_ROOT}/round3_op_main_seed42:9120:shard0" \
  "3:op_seed43:${MERGED_ROOT}/round3_op_main_seed43:9130:shard1"

echo "[b3-retry] Wave 2 closed-loop (2 shards)"
launch_wave \
  "0:op_seed42:${MERGED_ROOT}/round3_op_main_seed42:9140:shard1" \
  "1:op_seed44:${MERGED_ROOT}/round3_op_main_seed44:9150:shard0"

echo "[b3-retry] closed-loop complete $(date -Is)"

# Summary
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round4/postfix_replay")
lines = ["# Barrier 3 Retry Summary", ""]
for p in sorted((root / "offline").glob("*.json")):
    d = json.loads(p.read_text())
    lines.append(f"- **{p.stem}**: macro_f1={d.get('macro_f1',0):.3f} acc={d.get('operation_accuracy',0):.3f}")
lines.append("")
lines.append("## Closed-loop shards")
for vdir in sorted((root / "closed_loop").iterdir()):
    if not vdir.is_dir():
        continue
    for sdir in sorted(vdir.iterdir()):
        ok = (sdir / "summary.json").exists()
        lines.append(f"- {vdir.name}/{sdir.name}: {'OK' if ok else 'MISSING'}")
out = root / "BARRIER3_RETRY_REPORT.md"
out.write_text("\n".join(lines) + "\n")
print("Wrote", out)
PY

echo "[b3-retry] all done $(date -Is)"
