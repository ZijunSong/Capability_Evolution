#!/usr/bin/env bash
# Round 7 shared helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round7"
R5="${REPO_ROOT}/outputs/scope_round5"
R6="${REPO_ROOT}/outputs/scope_round6"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
VALID522="${REPO_ROOT}/artifacts/datasets/dup_sdi_round3/valid.jsonl"
PARALLEL="${PARALLEL:-64}"

scope7_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/contract_trace/live" \
    "${OUT}/contract_trace/replay_hf" \
    "${OUT}/contract_trace/replay_vllm" \
    "${OUT}/contract_trace/comparisons" \
    "${OUT}/holdout_tau0" "${OUT}/sentinel" "${OUT}/preflight"
}

scope7_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round7_supervisor.log"
}

scope7_wait_gpu_free() {
  local gpu="$1" max_wait="${2:-7200}"
  local elapsed=0
  while (( elapsed < max_wait )); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | tr -d ' ')
    if [[ "${used}" -lt 10000 ]]; then
      scope7_log "GPU${gpu} free (${used} MiB used)"
      return 0
    fi
    scope7_log "GPU${gpu} busy (${used} MiB), waiting 60s..."
    sleep 60
    elapsed=$((elapsed + 60))
  done
  scope7_log "WARN: GPU${gpu} still busy after ${max_wait}s"
  return 1
}

scope7_write_marker() {
  local name="$1" status="$2" out_dir="$3" expected="${4:-25}"
  python - "${name}" "${status}" "${out_dir}" "${expected}" <<'PY'
import json, sys
from pathlib import Path
name, status, out_dir, expected = sys.argv[1:5]
out = Path(out_dir)
expected = int(expected)
actual = 0
n_trace = 0
n_errors = 0
artifacts = {}
ep = out / "episodes.jsonl"
if ep.exists():
    actual = sum(1 for l in ep.open() if l.strip())
tp = out / "live_dup_decision_trace.jsonl"
if tp.exists():
    n_trace = sum(1 for l in tp.open() if l.strip())
sm = out / "summary.json"
if sm.exists():
    artifacts[str(sm)] = "present"
marker = {
    "status": status,
    "expected_episodes": expected,
    "actual_episodes": actual,
    "n_trace_events": n_trace,
    "n_errors": n_errors,
    "telemetry_complete": actual >= expected and n_trace > 0,
    "artifacts": artifacts,
}
root = Path(out_dir).parents[2] if "contract_trace" in out_dir else Path(out_dir).parents[1]
# fallback marker dir
md = root / "markers"
if not md.exists():
    md = Path(out_dir).parent.parent / "markers"
md.mkdir(parents=True, exist_ok=True)
(md / f"{name}.json").write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps(marker))
PY
}

scope7_run_live() {
  local gpu="$1" out="$2" model="$3" port="$4" shard="$5" seed="$6" label="$7"
  local marker_name="$8"
  if [[ -f "${out}/contract_gate.json" ]]; then
    local passed
    passed=$(python -c "import json; print(json.load(open('${out}/contract_gate.json')).get('contract_gate_pass', False))")
    if [[ "${passed}" == "True" ]] && [[ -f "${out}/episodes.jsonl" ]]; then
      scope7_log "Skip complete run ${out}"
      return 0
    fi
  fi
  mkdir -p "${out}"
  if [[ "${SCOPE7_NO_RESUME:-0}" == "1" ]]; then
    rm -f "${out}/episodes.jsonl" "${out}/decision_states.jsonl" \
      "${out}/dup_admission_events.jsonl" "${out}/live_dup_decision_trace.jsonl" \
      "${out}/contract_gate.json" "${out}/summary.json" "${out}/aggregated_metrics.json"
    rm -rf "${out}/prompt_sidecar"
  fi
  scope7_log "Live harness GPU${gpu} -> ${out} shard=${shard}"
  local resume_flag=(--resume)
  if [[ "${SCOPE7_NO_RESUME:-0}" == "1" ]]; then
    resume_flag=(--no-resume)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --dup-operation \
    --decision-threshold 0 \
    --dup-seed "${seed}" \
    --checkpoint-label "${label}" \
    --round7-trace \
    --parallel "${PARALLEL}" \
    "${resume_flag[@]}" \
    >> "${LOG_DIR}/${marker_name}.log" 2>&1
  scope7_write_marker "${marker_name}" "complete" "${out}" 25
}

scope7_run_hf_replay() {
  local gpu="$1" trace_dir="$2" model="$3" tag="$4"
  scope7_log "HF replay ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round7/replay_live_trace_hf.py \
    --trace-dir "${trace_dir}" \
    --model-path "${model}" \
    --gpu "cuda:0" \
    --output-dir "${OUT}/contract_trace/replay_hf/${tag}" \
    >> "${LOG_DIR}/hf_replay_${tag}.log" 2>&1
}

scope7_run_vllm_replay() {
  local gpu="$1" trace_dir="$2" model="$3" port="$4" tag="$5"
  scope7_log "vLLM replay ${tag}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round7/replay_live_trace_vllm.py \
    --trace-dir "${trace_dir}" \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --output-dir "${OUT}/contract_trace/replay_vllm/${tag}" \
    >> "${LOG_DIR}/vllm_replay_${tag}.log" 2>&1
}

scope7_run_compare() {
  local trace_dir="$1" tag="$2"
  local hf="${OUT}/contract_trace/replay_hf/${tag}/hf_replay.json"
  local vl="${OUT}/contract_trace/replay_vllm/${tag}/vllm_replay.json"
  scope7_log "Compare ${tag}"
  python training/scope_round7/compare_live_replay.py \
    --trace-dir "${trace_dir}" \
    --hf-replay "${hf}" \
    --vllm-replay "${vl}" \
    --output-dir "${OUT}/contract_trace/comparisons/${tag}" \
    >> "${LOG_DIR}/compare_${tag}.log" 2>&1 || true
  python training/scope_round7/contract_gate.py --run-dir "${trace_dir}" \
    >> "${LOG_DIR}/gate_${tag}.log" 2>&1 || true
}

scope7_gate_passed() {
  local out="$1"
  [[ -f "${out}/contract_gate.json" ]] && \
    python -c "import json,sys; sys.exit(0 if json.load(open('${out}/contract_gate.json')).get('contract_gate_pass') else 1)"
}

scope7_contract_pipeline() {
  local gpu="$1" trace_dir="$2" model="$3" vllm_port="$4" tag="$5"
  scope7_run_hf_replay "${gpu}" "${trace_dir}" "${model}" "${tag}"
  scope7_run_vllm_replay "${gpu}" "${trace_dir}" "${model}" "${vllm_port}" "${tag}"
  scope7_run_compare "${trace_dir}" "${tag}"
}
