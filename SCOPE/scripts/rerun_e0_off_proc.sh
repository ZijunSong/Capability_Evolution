#!/usr/bin/env bash
# Re-run only OFF/PROC modes (FULL stays reused from Phase-0 rollout).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
ROOT_OUT="${ROOT_OUT:-$REPO_ROOT/outputs/scope_e0_distillability}"
VLLM_PORT="${VLLM_PORT:-8776}"
PARALLEL="${PARALLEL:-2}"
CAPABILITIES="${CAPABILITIES:-duplicate_evidence,stop_decision,evidence_curation,verification_decision,external_verification,deterministic_truncation}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"
cd "${REPO_ROOT}"

bash "${REPO_ROOT}/scripts/start_e0_vllm.sh"
export base_url="http://127.0.0.1:${VLLM_PORT}/v1"
export api_key="EMPTY"
export model_name="${SERVED_MODEL_NAME:-e0-harness-policy}"

wait_vllm() {
  python - <<PY
import time, urllib.request, sys
url = "http://127.0.0.1:${VLLM_PORT}/v1/models"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(2)
sys.exit(1)
PY
}

IFS=',' read -ra CAPS <<< "${CAPABILITIES}"
for CAP in "${CAPS[@]}"; do
  CAP="$(echo "${CAP}" | xargs)"
  for MODE in off proc; do
    if [[ "${CAP}" == "deterministic_truncation" && "${MODE}" == "proc" ]]; then
      continue
    fi
    echo "=== Rerun ${CAP}/${MODE} ==="
    bash "${REPO_ROOT}/scripts/start_e0_vllm.sh"
    wait_vllm || { echo "vLLM not ready; aborting"; exit 1; }
    python training/scope/distillability/runner.py \
      --capability "${CAP}" \
      --mode "${MODE}" \
      --output-dir "${ROOT_OUT}" \
      --queries-json artifacts/datasets/e0_audit_100q/query_ids.json \
      --seed 42 \
      --model-path /data/ppnm/models/Qwen2.5-7B-Instruct \
      --parallel "${PARALLEL}" \
      --vllm-port "${VLLM_PORT}" \
      --no-manage-vllm \
      --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
      --resume
  done
done

python training/scope/distillability/build_map.py \
  --root "${ROOT_OUT}" \
  --output-map "${REPO_ROOT}/artifacts/capability/distillability_map.json" \
  --output-report "${ROOT_OUT}/E0_REPORT.md"

echo "Rerun complete. See ${ROOT_OUT}/E0_REPORT.md"

if [[ "${E0_CLEANUP_VLLM:-1}" == "1" ]]; then
  echo "=== Stopping E0 vLLM ==="
  bash "${REPO_ROOT}/scripts/stop_e0_vllm.sh" || true
fi
