#!/usr/bin/env bash
# Dup-SDI Minimal Runtime smoke: base vs trained on same 20 queries.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

BASE_MODEL="${BASE_MODEL:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-$REPO_ROOT/outputs/dup_sdi_round1}"
MERGED_DIR="${MERGED_DIR:-$REPO_ROOT/outputs/dup_sdi_round1/merged_hf}"
SMOKE_ROOT="${SMOKE_ROOT:-$REPO_ROOT/outputs/dup_sdi_round1/minimal_runtime_smoke20}"
LIMIT="${LIMIT:-20}"
SMOKE_GPUS="${SMOKE_GPUS:-4,5,6,7}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}"
export CUDA_VISIBLE_DEVICES
VLLM_PORT_BASE="${VLLM_PORT_BASE:-8776}"
VLLM_PORT_TRAINED="${VLLM_PORT_TRAINED:-8777}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dup-sdi-minimal-smoke20}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_minimal.yaml}"
SCOPE_CONFIG="${SCOPE_CONFIG:-$REPO_ROOT/configs/scope/minimal_runtime.yaml}"

export PYTHONPATH="${REPO_ROOT}"
export LIMIT RESUME=0 PARALLEL=2 SPLIT=all
export MAX_TURNS=35 MAX_TOKENS=2048 TEMPERATURE=1.0 MAX_MODEL_LEN=32768
export USE_LEGACY_API_AGENT=0 RETRIEVAL=bm25 RERANKER=none

mkdir -p "${SMOKE_ROOT}"

_run_one() {
  local tag="$1"
  local model_path="$2"
  local port="$3"
  local out="${SMOKE_ROOT}/${tag}"
  rm -rf "${out}"
  mkdir -p "${out}"
  echo "[smoke20] === ${tag} model=${model_path} port=${port} gpus=${CUDA_VISIBLE_DEVICES} ==="
  export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}"
  export MODEL_PATH="${model_path}"
  export OUTPUT_DIR="${out}"
  export VLLM_PORT="${port}"
  export SERVED_MODEL_NAME
  export model_name="${SERVED_MODEL_NAME}"
  export MODEL_NAME="${SERVED_MODEL_NAME}"
  bash "${REPO_ROOT}/scripts/rollout_minimal_runtime_browsecomp.sh"
  python "${REPO_ROOT}/scripts/finalize_minimal_runtime_artifacts.py" \
    --output-dir "${out}" \
    --harness-config "${HARNESS_CONFIG}" \
    --scope-config "${SCOPE_CONFIG}"
}

if [[ ! -d "${MERGED_DIR}" || "${FORCE_MERGE:-0}" == "1" ]]; then
  echo "[smoke20] Merging LoRA -> ${MERGED_DIR}"
  env CUDA_VISIBLE_DEVICES="${SMOKE_GPUS%%,*}" python "${REPO_ROOT}/training/merge_lora_hf.py" \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --output "${MERGED_DIR}"
fi

export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}"

_run_one "base" "${BASE_MODEL}" "${VLLM_PORT_BASE}"
_run_one "trained" "${MERGED_DIR}" "${VLLM_PORT_TRAINED}"

python - <<PY
import json
from pathlib import Path
root = Path("${SMOKE_ROOT}")
rows = []
for tag in ("base", "trained"):
    s = json.loads((root / tag / "summary.json").read_text())
    rows.append({"tag": tag, **s})
(root / "compare_smoke20.json").write_text(
    json.dumps({"n": ${LIMIT}, "runs": rows}, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps({"ok": True, "compare": str(root / "compare_smoke20.json")}, indent=2))
PY

echo "[smoke20] Done. See ${SMOKE_ROOT}/compare_smoke20.json"
