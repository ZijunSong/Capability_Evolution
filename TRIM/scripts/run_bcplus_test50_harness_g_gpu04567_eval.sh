#!/usr/bin/env bash
# Harness-G bcplus_test_50 — sequential four-run eval on GPU 0,4,5,6,7:
#   gpt-oss-20b              × {zero, all}
#   Qwen3-4B-Instruct-2507   × {zero, all}
#
# Unlike Harness-1 upstream_api, Harness-G stays on its graph runtime
# (legacy_local). --tp is data-parallel replica count: five independent
# vLLM workers, one GPU each, then merge. No separate verifier GPU.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_test50_harness_g_gpu04567_eval.sh
#
# Optional env:
#   RUN_ID, MODELS, COMPONENTS, SKIP_ZERO=1, TP, EVAL_GPUS,
#   GPU_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN, MAX_TURNS, TEMPERATURE,
#   EVAL_STAGGER, PY, BETWEEN_S
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_hg_test50_gpu04567}"
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"

REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"
PKG="/data/ppnm/trim_bcplus_test50_eval_harness_g"

EVAL_GPUS="${EVAL_GPUS:-0,4,5,6,7}"
TP="${TP:-5}"
GPU_UTIL="${GPU_UTIL:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEARCH_K="${SEARCH_K:-10}"
EVAL_STAGGER="${EVAL_STAGGER:-8}"
BETWEEN_S="${BETWEEN_S:-20}"

MODEL_PATHS=(
  "/data/ppnm/models/gpt-oss-20b"
  "/data/ppnm/models/Qwen3-4B-Instruct-2507"
)
MODEL_SLUGS=(
  "gpt-oss-20b"
  "Qwen3-4B-Instruct-2507"
)
COMPONENTS_DEFAULT=(zero all)

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}${PYTHONPATH:+:${PYTHONPATH}}"
export TRIM_GPU_KEEPALIVE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Parent must see all physical ids in --eval-gpus; children pin one GPU each.
unset CUDA_VISIBLE_DEVICES || true

mkdir -p "${REL_LOGS}" "${PKG}/results/${RUN_ID}" "${PKG}/logs/${RUN_ID}"
MASTER_LOG="${REL_LOGS}/master.log"

log() {
  echo "[$(date -Is)] $*" >> "${MASTER_LOG}"
  echo "[$(date -Is)] $*" >&2
}

csv_to_array() {
  local csv="$1"
  local -n _arr="$2"
  _arr=()
  local IFS=,
  read -ra _arr <<< "${csv}"
}

model_selected() {
  local slug="$1"
  [[ -z "${MODELS:-}" ]] && return 0
  local IFS=,
  local wanted
  for wanted in ${MODELS}; do
    [[ "${wanted}" == "${slug}" ]] && return 0
  done
  return 1
}

component_selected() {
  local c="$1"
  [[ -z "${COMPONENTS:-}" ]] && return 0
  local IFS=,
  local wanted
  for wanted in ${COMPONENTS}; do
    [[ "${wanted}" == "${c}" ]] && return 0
  done
  return 1
}

log_gpu() {
  local msg
  msg="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")"
  log "gpu_status ${msg//$'\n'/ | }"
}

wait_between_runs() {
  local s="${1:-${BETWEEN_S}}"
  log "cooling ${s}s before next eval"
  sleep "${s}"
  log_gpu
}

run_eval() {
  local model_path="$1" model_slug="$2" component="$3"
  local gpu_tag="gpu${EVAL_GPUS//,/}"
  local out="${REL_OUT}/eval_hg_bcplus_test50_${model_slug}_${component}_${gpu_tag}_${RUN_ID}"
  local pkg_out="${PKG}/results/${RUN_ID}/${model_slug}/${component}"
  local log_path="${REL_LOGS}/eval_${model_slug}_${component}.log"

  log "START harness=Harness-G model=${model_slug} component=${component} tp=${TP} gpus=${EVAL_GPUS} out=${out}"
  rm -rf "${out}" "${pkg_out}"
  mkdir -p "${pkg_out}"

  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-G \
    --benchmark bcplus_test_50 \
    --model_name "${model_path}" \
    --evaluation-path legacy_local \
    --eval-mode harness \
    --component "${component}" \
    --tp "${TP}" \
    --eval-gpus "${EVAL_GPUS}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --search-k "${SEARCH_K}" \
    --eval-stagger-s "${EVAL_STAGGER}" \
    --rollout-backend vllm \
    --out "${out}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?

  if [[ -d "${out}" ]]; then
    rsync -a "${out}/" "${pkg_out}/"
  fi
  cp -a "${log_path}" "${PKG}/logs/${RUN_ID}/eval_${model_slug}_${component}.log" 2>/dev/null || true

  if [[ "${rc}" -ne 0 ]]; then
    log "WARN eval exited ${rc} model=${model_slug} component=${component}"
    return "${rc}"
  fi
  log "DONE model=${model_slug} component=${component}"
}

main() {
  csv_to_array "${EVAL_GPUS}" _gpus
  ((${#_gpus[@]} == TP)) || {
    log "EVAL_GPUS count (${#_gpus[@]}) must equal TP (${TP})"
    exit 1
  }
  [[ -x "${PY}" ]] || { log "missing python ${PY}"; exit 1; }

  local comps=()
  if [[ -n "${COMPONENTS:-}" ]]; then
    csv_to_array "${COMPONENTS}" comps
  else
    comps=("${COMPONENTS_DEFAULT[@]}")
  fi

  log "RUN_ID=${RUN_ID} harness=Harness-G benchmark=bcplus_test_50 eval_gpus=${EVAL_GPUS} tp=${TP} gpu_util=${GPU_UTIL} max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN} max_turns=${MAX_TURNS} temperature=${TEMPERATURE} models=${MODELS:-all} components=${comps[*]}"
  log_gpu

  local failed=0
  local idx slug path comp first=1
  for idx in "${!MODEL_PATHS[@]}"; do
    path="${MODEL_PATHS[$idx]}"
    slug="${MODEL_SLUGS[$idx]}"
    model_selected "${slug}" || { log "skip model ${slug} (not in MODELS)"; continue; }
    [[ -d "${path}" ]] || { log "missing model dir ${path}"; exit 1; }

    for comp in "${comps[@]}"; do
      if [[ "${comp}" == "zero" && "${SKIP_ZERO:-0}" == "1" ]]; then
        log "SKIP_ZERO=1 — skipping ${slug} zero"
        continue
      fi
      component_selected "${comp}" || { log "skip component ${comp}"; continue; }
      if [[ "${first}" != "1" ]]; then
        wait_between_runs
      fi
      first=0
      run_eval "${path}" "${slug}" "${comp}" || failed=$((failed + 1))
    done
  done

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  GIT_DIRTY="$(git -C . status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  cat > "${PKG}/results/${RUN_ID}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "harness": "Harness-G",
  "benchmark": "bcplus_test_50",
  "evaluation_path": "legacy_local",
  "eval_profile": "harness_g_graph_runtime",
  "eval_gpus": "${EVAL_GPUS}",
  "eval_replicas": ${TP},
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": ${GPU_UTIL},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_turns": ${MAX_TURNS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "temperature": ${TEMPERATURE},
  "search_k": ${SEARCH_K},
  "eval_stagger_s": ${EVAL_STAGGER},
  "models_filter": "${MODELS:-}",
  "components": ["zero", "all"],
  "experiments": [
    {"model": "/data/ppnm/models/gpt-oss-20b", "components": ["zero", "all"]},
    {"model": "/data/ppnm/models/Qwen3-4B-Instruct-2507", "components": ["zero", "all"]}
  ],
  "git_commit": "${GIT_COMMIT}",
  "git_dirty_files": ${GIT_DIRTY:-0},
  "failed_runs": ${failed}
}
EOF
  cp "${PKG}/results/${RUN_ID}/MANIFEST.json" "${REL_LOGS}/MANIFEST.json"
  log "finished; failed=${failed}; results ${PKG}/results/${RUN_ID}/ manifest ${REL_LOGS}/MANIFEST.json"
  [[ "${failed}" -eq 0 ]]
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
