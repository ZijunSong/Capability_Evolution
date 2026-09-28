#!/usr/bin/env bash
# Rerun the 3 gpt-oss-20b/all queries that died on Ellipsis JSON serialization,
# then merge them back into the original official eval directories.
#
#   longsealqa: longseal-004
#   hotpotqa:   5a75f0ea5542994ccc91866c,5ab90d2755429916710eb0f0
#
# Usage:
#   bash TRIM/scripts/rerun_gptoss_all_ellipsis_queries.sh
#
# Optional env:
#   DRY_RUN=1, SKIP_SERVE=1, RUN_ID, ACTOR_GPU, VERIFY_GPU, ACTOR_PORT,
#   VERIFY_PORT, GPU_UTIL, FORCE_UNSHARE_SHM=1, FAIL_FAST=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x /data/ppnm/miniconda3/envs/bishop/bin/python ]]; then
    PY=/data/ppnm/miniconda3/envs/bishop/bin/python
  else
    PY=python
  fi
fi
if [[ -z "${VLLM:-}" ]]; then
  if [[ -x /data/ppnm/miniconda3/envs/bishop/bin/vllm ]]; then
    VLLM=/data/ppnm/miniconda3/envs/bishop/bin/vllm
  else
    VLLM=vllm
  fi
fi

# shellcheck source=scripts/vllm_model_extra.sh
source "${SCRIPT_DIR}/vllm_model_extra.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_ellipsis_rerun}"
REL_MODELS="${REL_MODELS:-../../models}"
REL_OUT="${REL_OUT:-outputs}"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="${REL_SCAPE_EASYOPD:-../SCAPE-EasyOPD}"
ACTOR_GPU="${ACTOR_GPU:-3}"
VERIFY_GPU="${VERIFY_GPU:-6}"
ACTOR_PORT="${ACTOR_PORT:-8042}"
VERIFY_PORT="${VERIFY_PORT:-8050}"
GPU_UTIL="${GPU_UTIL:-0.55}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
FORCE_UNSHARE_SHM="${FORCE_UNSHARE_SHM:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"
FAIL_FAST="${FAIL_FAST:-1}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

LONGSEAL_ORIG="${LONGSEAL_ORIG:-${REL_OUT}/eval_h1_longsealqa_gpt-oss-20b_all_gpu03457v6_20260919_0748_transfer_scale6}"
HOTPOT_ORIG="${HOTPOT_ORIG:-${REL_OUT}/eval_h1_hotpotqa_gpt-oss-20b_all_gpu03457v6_20260919_0748_transfer_scale6}"
LONGSEAL_QIDS="${LONGSEAL_QIDS:-longseal-004}"
HOTPOT_QIDS="${HOTPOT_QIDS:-5a75f0ea5542994ccc91866c,5ab90d2755429916710eb0f0}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export HARNESS1_FORBID_CHROMA=1
export CURATE_NUDGE_POLICY="${CURATE_NUDGE_POLICY:-legacy}"

mkdir -p "${REL_LOGS}"
MASTER_LOG="${REL_LOGS}/master.log"
ACTOR_PID=""
VERIFY_PID=""

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}" >&2
}

normalize_pid() {
  local raw="${1:-}"
  raw="${raw//$'\r'/}"
  while [[ "${raw}" == *$'\n' ]]; do raw="${raw%$'\n'}"; done
  if [[ "${raw}" =~ ^([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}"; return 0; fi
  local last="${raw##*$'\n'}"
  [[ "${last}" =~ ^([0-9]+)$ ]] && { echo "${BASH_REMATCH[1]}"; return 0; }
  return 1
}

stop_pid() {
  local pid="${1:-}"
  pid="$(normalize_pid "${pid}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then lsof -ti ":${port}" 2>/dev/null || true; return 0; fi
  if command -v ss >/dev/null 2>&1; then
    ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' || true
    return 0
  fi
}

stop_port() {
  local port="$1" pid_file="${2:-}" pid pids
  if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
    pid="$(normalize_pid "$(cat "${pid_file}")" 2>/dev/null || true)"
    stop_pid "${pid}"
    rm -f "${pid_file}"
  fi
  pids="$(pids_on_port "${port}")"
  if [[ -n "${pids}" ]]; then
    log "stopping stale listener(s) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

cleanup() {
  if [[ "${SKIP_SERVE}" != "1" ]]; then
    stop_pid "${ACTOR_PID}"
    stop_pid "${VERIFY_PID}"
  fi
}
trap cleanup EXIT

wait_api() {
  local port="$1" timeout_s="${2:-900}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  log "timeout waiting for API on port ${port}"
  return 1
}

need_unshare_shm() {
  [[ "${FORCE_UNSHARE_SHM}" == "1" ]] && return 0
  [[ -w /dev/shm ]] && return 1
  return 0
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"
  stop_port "${port}" "${log_path}.pid"
  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  if need_unshare_shm; then
    log "wrapping vLLM with unshare tmpfs /dev/shm"
    # shellcheck disable=SC2086
    nohup unshare --user --map-root-user --mount bash -c "
      mount -t tmpfs -o size=16G tmpfs /dev/shm
      chmod 1777 /dev/shm
      export CUDA_VISIBLE_DEVICES='${gpu}'
      exec '${VLLM}' serve '${model_path}' \
        --host 127.0.0.1 --port '${port}' --served-model-name '${served_name}' \
        ${extra} --gpu-memory-utilization '${GPU_UTIL}' --max-num-seqs '${VLLM_MAX_NUM_SEQS}' --enforce-eager
    " > "${log_path}" 2>&1 &
  else
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
      --host 127.0.0.1 --port "${port}" --served-model-name "${served_name}" \
      ${extra} --gpu-memory-utilization "${GPU_UTIL}" --max-num-seqs "${VLLM_MAX_NUM_SEQS}" --enforce-eager \
      > "${log_path}" 2>&1 &
  fi
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  echo "${pid}"
}

run_one() {
  local bench="$1" qids="$2" orig="$3"
  local out="${REL_OUT}/eval_h1_${bench}_gpt-oss-20b_all_ellipsis_rerun_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${bench}.log"
  log "START ${bench} qids=${qids} out=${out}"
  echo "JOB start benchmark=${bench} query_ids=${qids} out=${out} original=${orig}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${PY} scripts/run_eval.py --harness Harness-1 --benchmark ${bench} --model_name ${REL_MODELS}/gpt-oss-20b --evaluation-path upstream_api --retrieval-backend local_bm25 --api-base-url http://127.0.0.1:${ACTOR_PORT}/v1 --api-model gpt-oss-20b --component all --query-ids ${qids} --verify-base-url http://127.0.0.1:${VERIFY_PORT}/v1 --verify-model harness-1-verifier --max-turns ${MAX_TURNS} --max-new-tokens ${MAX_NEW_TOKENS} --reasoning-effort ${REASONING_EFFORT} --tp 1 --out ${out}"
    echo "DRY_RUN ${PY} scripts/merge_upstream_api_rerun.py --original-out ${orig} --rerun-out ${out} --query-ids ${qids}"
    return 0
  fi
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark "${bench}" \
    --model_name "${REL_MODELS}/gpt-oss-20b" \
    --evaluation-path upstream_api \
    --retrieval-backend local_bm25 \
    --api-base-url "http://127.0.0.1:${ACTOR_PORT}/v1" \
    --api-model gpt-oss-20b \
    --reranker none \
    --offline \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --temperature "${TEMPERATURE}" \
    --tp 1 \
    --component all \
    --query-ids "${qids}" \
    --verify-base-url "http://127.0.0.1:${VERIFY_PORT}/v1" \
    --verify-model harness-1-verifier \
    --reasoning-effort "${REASONING_EFFORT}" \
    --out "${out}" \
    > "${log_path}" 2>&1 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL eval ${bench} rc=${rc} see ${log_path}"
    tail -n 40 "${log_path}" | tee -a "${MASTER_LOG}" >&2 || true
    [[ "${FAIL_FAST}" == "1" ]] && return "${rc}"
    return 0
  fi
  log "MERGE ${bench} into ${orig}"
  "${PY}" scripts/merge_upstream_api_rerun.py \
    --original-out "${orig}" \
    --rerun-out "${out}" \
    --query-ids "${qids}" | tee -a "${MASTER_LOG}"
}

main() {
  log "RUN_ID=${RUN_ID} actor=GPU${ACTOR_GPU}:${ACTOR_PORT} verify=GPU${VERIFY_GPU}:${VERIFY_PORT} dry_run=${DRY_RUN} skip_serve=${SKIP_SERVE}"
  if [[ "${DRY_RUN}" != "1" && "${SKIP_SERVE}" != "1" ]]; then
    local extra
    extra="$(vllm_extra_for_model gpt-oss-20b) ${VLLM_EXTRA:-}"
    ACTOR_PID="$(start_vllm_bg "${ACTOR_GPU}" "${ACTOR_PORT}" "${REL_MODELS}/gpt-oss-20b" "gpt-oss-20b" \
      "${extra}" "${REL_LOGS}/actor.log")"
    ACTOR_PID="$(normalize_pid "${ACTOR_PID}")"
    VERIFY_PID="$(start_vllm_bg "${VERIFY_GPU}" "${VERIFY_PORT}" "${REL_MODELS}/harness-1" "harness-1-verifier" \
      "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
      "${REL_LOGS}/verify.log")"
    VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
  fi
  run_one longsealqa "${LONGSEAL_QIDS}" "${LONGSEAL_ORIG}"
  run_one hotpotqa "${HOTPOT_QIDS}" "${HOTPOT_ORIG}"
  if [[ "${DRY_RUN}" != "1" && "${SKIP_SERVE}" != "1" ]]; then
    stop_pid "${ACTOR_PID}"
    stop_pid "${VERIFY_PID}"
    ACTOR_PID=""
    VERIFY_PID=""
    trap - EXIT
  fi
  log "finished logs=${REL_LOGS}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
