#!/usr/bin/env bash
# SCOPE E0 distillability — nohup-safe orchestrator with flock + resume.
# Usage:
#   nohup bash scripts/run_e0_distillability_nohup.sh >> outputs/scope_e0_distillability/nohup_master.log 2>&1 &
#   echo $! > outputs/scope_e0_distillability/nohup_master.pid
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
ROOT_OUT="${ROOT_OUT:-$REPO_ROOT/outputs/scope_e0_distillability}"
LOCK_FILE="${LOCK_FILE:-$ROOT_OUT/.e0_orchestrator.lock}"
VLLM_PORT="${VLLM_PORT:-8776}"
PARALLEL="${PARALLEL:-1}"
# Prefer free GPU; override with CUDA_VISIBLE_DEVICES if needed.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-e0-harness-policy}"
CAPABILITIES="${CAPABILITIES:-duplicate_evidence,stop_decision,evidence_curation,verification_decision,external_verification,deterministic_truncation}"

mkdir -p "${ROOT_OUT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[e0-orchestrator] Another instance is running (lock=${LOCK_FILE}). Exit."
  exit 0
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"
cd "${REPO_ROOT}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_vllm() {
  python - <<PY
import time, urllib.request, sys
url = "http://127.0.0.1:${VLLM_PORT}/v1/models"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(3)
sys.exit(1)
PY
}

needs_run() {
  local cap="$1" mode="$2" target="${3:-100}"
  python - <<PY
import json
from pathlib import Path
p = Path("${ROOT_OUT}") / "${cap}" / "${mode}" / "episodes.jsonl"
if not p.exists():
    print("yes"); raise SystemExit
ok = 0
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    d = json.loads(line)
    if not d.get("error"):
        ok += 1
print("yes" if ok < int("${target}") else "no")
PY
}

all_jobs_complete() {
  local cap mode
  IFS=',' read -ra _CAPS <<< "${CAPABILITIES}"
  for cap in "${_CAPS[@]}"; do
    cap="$(echo "${cap}" | xargs)"
    for mode in full off proc; do
      if [[ "${cap}" == "deterministic_truncation" && "${mode}" == "proc" ]]; then
        continue
      fi
      if [[ "$(needs_run "${cap}" "${mode}")" == "yes" ]]; then
        return 1
      fi
    done
  done
  return 0
}

maybe_stop_vllm() {
  if [[ "${E0_CLEANUP_VLLM:-1}" != "1" ]]; then
    log "E0_CLEANUP_VLLM=0 — keeping vLLM running"
    return 0
  fi
  if all_jobs_complete; then
    log "=== Stopping E0 vLLM (all jobs complete) ==="
    bash "${REPO_ROOT}/scripts/stop_e0_vllm.sh" || log "WARN stop_e0_vllm failed"
  else
    log "=== Keeping E0 vLLM (incomplete jobs remain for resume) ==="
  fi
}

run_job() {
  local cap="$1" mode="$2"
  if [[ "$(needs_run "${cap}" "${mode}")" != "yes" ]]; then
    log "SKIP ${cap}/${mode} (already complete)"
    return 0
  fi
  log "START ${cap}/${mode}"
  export base_url="http://127.0.0.1:${VLLM_PORT}/v1"
  export api_key="EMPTY"
  export model_name="${SERVED_MODEL_NAME}"
  if ! bash "${REPO_ROOT}/scripts/start_e0_vllm.sh"; then
    log "WARN vLLM start failed for ${cap}/${mode}; skipping (GPU may be unavailable)"
    return 1
  fi
  if ! wait_vllm; then
    log "WARN vLLM not ready on port ${VLLM_PORT}; skipping ${cap}/${mode}"
    return 1
  fi
  if ! python training/scope/distillability/runner.py \
    --capability "${cap}" \
    --mode "${mode}" \
    --output-dir "${ROOT_OUT}" \
    --queries-json artifacts/datasets/e0_audit_100q/query_ids.json \
    --seed 42 \
    --model-path /data/ppnm/models/Qwen2.5-7B-Instruct \
    --parallel "${PARALLEL}" \
    --vllm-port "${VLLM_PORT}" \
    --no-manage-vllm \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --resume; then
    log "FAILED ${cap}/${mode} (runner exit $?)"
    return 1
  fi
  if [[ "$(needs_run "${cap}" "${mode}")" == "yes" ]]; then
    log "INCOMPLETE ${cap}/${mode} (will retry on next nohup run)"
    return 1
  fi
  log "DONE ${cap}/${mode} (verified 100 ok)"
}

log "=== E0 orchestrator start (GPU=${CUDA_VISIBLE_DEVICES}, port=${VLLM_PORT}) ==="

# FULL: reuse Phase-0 rollout (no GPU rollout needed)
IFS=',' read -ra CAPS <<< "${CAPABILITIES}"
for CAP in "${CAPS[@]}"; do
  CAP="$(echo "${CAP}" | xargs)"
  if [[ "$(needs_run "${CAP}" full)" == "yes" ]]; then
    log "FULL ${CAP} (reuse)"
    python training/scope/distillability/runner.py \
      --capability "${CAP}" \
      --mode full \
      --output-dir "${ROOT_OUT}" \
      --queries-json artifacts/datasets/e0_audit_100q/query_ids.json \
      --seed 42 \
      --model-path /data/ppnm/models/Qwen2.5-7B-Instruct \
      --parallel 1 \
      --no-manage-vllm \
      --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
      --resume || true
  fi
done

# OFF / PROC rollout jobs (GPU required)
for CAP in "${CAPS[@]}"; do
  CAP="$(echo "${CAP}" | xargs)"
  run_job "${CAP}" off || log "FAILED ${CAP}/off (will retry on next run)"
  if [[ "${CAP}" != "deterministic_truncation" ]]; then
    run_job "${CAP}" proc || log "FAILED ${CAP}/proc (will retry on next run)"
  fi
done

log "=== Building distillability map ==="
python training/scope/distillability/build_map.py \
  --root "${ROOT_OUT}" \
  --output-map "${REPO_ROOT}/artifacts/capability/distillability_map.json" \
  --output-report "${ROOT_OUT}/E0_REPORT.md"

log "=== E0 orchestrator finished ==="
log "Report: ${ROOT_OUT}/E0_REPORT.md"
maybe_stop_vllm
