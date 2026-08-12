#!/usr/bin/env bash
# Queue LOCAL_CAL64 Full + LOO(-m) across 8 GPUs (TP=1 each).
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$SCAPE_ROOT/outputs/local_cal64_loo}"
LIMIT="${LIMIT:-64}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"

# Jobs: full + minus each default-on semantic/hybrid component
JOBS=(
  "full:"
  "minus_subtractive_curation:subtractive_curation"
  "minus_importance_tagging:importance_tagging"
  "minus_auto_populate_first_search:auto_populate_first_search"
  "minus_evidence_graph:evidence_graph"
  "minus_sentence_compress:sentence_compress"
  "minus_content_dedup:content_dedup"
  "minus_verify_tool:verify_tool"
  "minus_token_budget_marker:token_budget_marker"
)

QUEUE_FILE="${OUT_ROOT}/JOB_QUEUE.txt"
# If .keep_queue exists, do not rebuild/wipe the queue (prevents stampede while jobs run).
if [[ -f "${OUT_ROOT}/.keep_queue" ]]; then
  echo "[launch] .keep_queue present — leaving JOB_QUEUE untouched"
else
: > "${QUEUE_FILE}"
for item in "${JOBS[@]}"; do
  name="${item%%:*}"
  comp="${item#*:}"
  if [[ -f "${OUT_ROOT}/${name}/DONE" ]]; then
    echo "[launch] skip done ${name}"
    continue
  fi
  echo "${name}|${comp}" >> "${QUEUE_FILE}"
done
fi

echo "[launch] queue:"
cat "${QUEUE_FILE}" || true

# Per-GPU queue runners (optional subset via GPUS env, e.g. GPUS="0 1 2 3")
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
for gpu in ${GPUS}; do
  runner_log="${OUT_ROOT}/logs/gpu${gpu}_queue.log"
  pidfile="${OUT_ROOT}/pids/gpu${gpu}_queue.pid"
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "[launch] gpu${gpu} queue already running pid=$(cat "${pidfile}")"
    continue
  fi
  # stagger model loads to avoid simultaneous HF weight reads
  sleep $((gpu * 20))
  nohup bash -c "
    set -euo pipefail
    SCAPE_ROOT='${SCAPE_ROOT}'
    OUT_ROOT='${OUT_ROOT}'
    LIMIT='${LIMIT}'
    MODEL_PATH='${MODEL_PATH}'
    GPU='${gpu}'
    LOCK='${OUT_ROOT}/queue.lock'
    while true; do
      job=''
      (
        flock -x 200
        if [[ ! -s '${QUEUE_FILE}' ]]; then
          exit 3
        fi
        job=\$(head -n1 '${QUEUE_FILE}')
        tail -n +2 '${QUEUE_FILE}' > '${QUEUE_FILE}.tmp'
        mv '${QUEUE_FILE}.tmp' '${QUEUE_FILE}'
        echo \"\$job\"
      ) 200>'${OUT_ROOT}/queue.lock' > '${OUT_ROOT}/logs/gpu${gpu}_claim.txt' || {
        code=\$?
        if [[ \$code -eq 3 ]]; then
          echo \"[gpu${gpu}] queue empty; exit\"
          exit 0
        fi
        exit \$code
      }
      job=\$(cat '${OUT_ROOT}/logs/gpu${gpu}_claim.txt')
      [[ -z \"\$job\" ]] && continue
      name=\${job%%|*}
      comp=\${job#*|}
      echo \"[gpu${gpu}] claimed \$name component=\$comp\"
      GPU=\$GPU JOB_NAME=\$name COMPONENT=\$comp OUT_ROOT=\$OUT_ROOT LIMIT=\$LIMIT MODEL_PATH=\$MODEL_PATH \
        bash \"${SCAPE_ROOT}/scripts/run_loo_worker.sh\" || {
          echo \"[gpu${gpu}] FAILED \$name — will not requeue automatically\"
          echo \"\$name|\$comp\" >> \"${OUT_ROOT}/FAILED_JOBS.txt\"
        }
    done
  " >>"${runner_log}" 2>&1 &
  echo $! > "${pidfile}"
  echo "[launch] gpu${gpu} queue pid=$(cat "${pidfile}") log=${runner_log}"
done

# Write top-level STATUS
cat > "${OUT_ROOT}/STATUS_LIVE.md" <<EOF
# STATUS_LIVE — local_cal64_loo

- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')
- limit: ${LIMIT}
- model: ${MODEL_PATH}
- retrieval: bm25_provisional (NOT Chroma)
- queue_file: ${QUEUE_FILE}
- n_jobs_pending_at_launch: $(wc -l < "${QUEUE_FILE}" | tr -d ' ')

## GPU queue PIDs
$(for g in 0 1 2 3 4 5 6 7; do echo "- gpu${g}: $(cat "${OUT_ROOT}/pids/gpu${g}_queue.pid" 2>/dev/null || echo missing)"; done)
EOF

echo "[launch] all GPU queues started. Monitor: bash ${SCAPE_ROOT}/scripts/monitor_scape_loo.sh"
