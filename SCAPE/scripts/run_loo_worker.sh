#!/usr/bin/env bash
# One GPU worker: start TP=1 vLLM + CAL64 harness rollout under a component mask.
# Provisional LOCAL_CAL64 backend: SCOPE BM25 (NOT official Harness-1 Chroma).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE_ROOT="${SCOPE_ROOT:-$(cd "${SCAPE_ROOT}/../SCOPE" && pwd)}"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

GPU="${GPU:?GPU required}"
JOB_NAME="${JOB_NAME:?JOB_NAME required}"          # full | minus_<component_id>
COMPONENT="${COMPONENT:-}"                         # empty for full
OUT_ROOT="${OUT_ROOT:-$SCAPE_ROOT/outputs/local_cal64_loo}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-64}"
SPLIT="${SPLIT:-test}"
MAX_TURNS="${MAX_TURNS:-35}"
PARALLEL="${PARALLEL:-1}"
PORT=$((19500 + GPU))
MAX_ERROR_RATE="${MAX_ERROR_RATE:-0.15}"
SERVED="scape-cal64-gpu${GPU}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$SCOPE_ROOT/harness/configs/modules_full_v2.yaml}"
BM25="${BROWSECOMP_BM25_INDEX_PATH:-$SCOPE_ROOT/external/BrowseComp-Plus/indexes/bm25}"

JOB_DIR="${OUT_ROOT}/${JOB_NAME}"
mkdir -p "${JOB_DIR}/logs"
LOG="${JOB_DIR}/logs/worker.log"
PIDF="${JOB_DIR}/worker.pid"
VLLM_PIDF="${JOB_DIR}/vllm.pid"
STATUS="${JOB_DIR}/STATUS_LIVE.md"
MANIFEST="${JOB_DIR}/RUN_MANIFEST.json"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${SCOPE_ROOT}:${SCAPE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$SCOPE_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$SCOPE_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMPPLUS_QRELS_GOLD_PATH="${BROWSECOMPPLUS_QRELS_GOLD_PATH:-$SCOPE_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_golds.txt}"
export BROWSECOMPPLUS_QRELS_EVIDENCE_PATH="${BROWSECOMPPLUS_QRELS_EVIDENCE_PATH:-$SCOPE_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt}"
export BROWSECOMP_BM25_INDEX_PATH="${BM25}"
export SAVE_TRAJECTORIES=1
export SAVE_FULL_TRAJECTORIES=0
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"
export CHAT_MAX_RECENT_TURNS="${CHAT_MAX_RECENT_TURNS:-4}"

# Full operating point, then disable target component
export V8D_SUBTRACTIVE_CURATION=1
export V8D_IMPORTANCE_TAGGING=1
export V8D_AUTO_POPULATE_FIRST_SEARCH=1
export V8D_EVIDENCE_GRAPH=1
export V8D_SENTENCE_COMPRESS=1
export V8D_CHUNK_NEIGHBORS=0
export V8D_CONTENT_DEDUP=1
export V8D_VERIFY_TOOL=1
export V8D_TOKEN_BUDGET_MARKER=1
export V8D_ADAPTIVE_RERANK_INSTRUCTION=0

case "${COMPONENT}" in
  subtractive_curation) export V8D_SUBTRACTIVE_CURATION=0 ;;
  importance_tagging) export V8D_IMPORTANCE_TAGGING=0 ;;
  auto_populate_first_search) export V8D_AUTO_POPULATE_FIRST_SEARCH=0 ;;
  evidence_graph) export V8D_EVIDENCE_GRAPH=0 ;;
  sentence_compress) export V8D_SENTENCE_COMPRESS=0 ;;
  chunk_neighbors) export V8D_CHUNK_NEIGHBORS=0 ;;
  content_dedup) export V8D_CONTENT_DEDUP=0 ;;
  verify_tool) export V8D_VERIFY_TOOL=0 ;;
  token_budget_marker) export V8D_TOKEN_BUDGET_MARKER=0 ;;
  adaptive_rerank_instruction) export V8D_ADAPTIVE_RERANK_INSTRUCTION=0 ;;
  "") ;;
  *) echo "unknown COMPONENT=${COMPONENT}" >&2; exit 2 ;;
esac

echo $$ > "${PIDF}"
cat > "${MANIFEST}" <<EOF
{
  "schema_version": "scape_run_manifest_v1",
  "run_id": "${JOB_NAME}",
  "stage": "local_cal64_loo",
  "gpu": ${GPU},
  "port": ${PORT},
  "component": "${COMPONENT:-none}",
  "job_name": "${JOB_NAME}",
  "model_path": "${MODEL_PATH}",
  "retrieval": "bm25_provisional",
  "limit": ${LIMIT},
  "split": "${SPLIT}",
  "status": "running",
  "started_at": "$(date -Iseconds)",
  "provisional": true,
  "note": "LOCAL_CAL64 provisional LOO; BM25 backend, not official Harness-1 Chroma"
}
EOF

write_status() {
  local n_done="$1"
  cat > "${STATUS}" <<EOF
# STATUS_LIVE — ${JOB_NAME}

- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')
- gpu: ${GPU}
- port: ${PORT}
- component: ${COMPONENT:-full}
- n_expected: ${LIMIT}
- n_finished: ${n_done}
- log: ${LOG}
EOF
}
write_status 0

cleanup() {
  if [[ -f "${VLLM_PIDF}" ]]; then
    vpid="$(cat "${VLLM_PIDF}" || true)"
    if [[ -n "${vpid}" ]] && kill -0 "${vpid}" 2>/dev/null; then
      kill "${vpid}" 2>/dev/null || true
      sleep 2
      kill -9 "${vpid}" 2>/dev/null || true
    fi
  fi
  # orphan engine on this GPU/port
  pkill -f "vllm serve.*--port ${PORT}" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "[$(date -Iseconds)] start job=${JOB_NAME} gpu=${GPU} component=${COMPONENT:-full}"
  echo "[$(date -Iseconds)] model=${MODEL_PATH}"

  # Resume shortcut: already DONE
  if [[ -f "${JOB_DIR}/DONE" ]]; then
    echo "[$(date -Iseconds)] DONE marker present; skip"
    exit 0
  fi

  VLLM_URL="http://127.0.0.1:${PORT}/v1"
  nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${JOB_DIR}/logs/vllm.log" 2>&1 &
  echo $! > "${VLLM_PIDF}"

  python - <<PY
import json, time, urllib.request, sys
url = "${VLLM_URL}/models"
deadline = time.time() + 1200
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                break
    except Exception:
        time.sleep(3)
else:
    print("[loo] vLLM /models not ready; see ${JOB_DIR}/logs/vllm.log", flush=True)
    sys.exit(1)

# Smoke a real completion before rollout (catches half-ready servers)
smoke_url = "${VLLM_URL}/chat/completions"
payload = json.dumps({
    "model": "${SERVED}",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 8,
    "temperature": 0.0,
}).encode()
ok = False
for _ in range(60):
    try:
        req = urllib.request.Request(
            smoke_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                ok = True
                break
    except Exception as exc:
        print("[loo] smoke wait:", exc, flush=True)
        time.sleep(5)
if not ok:
    print("[loo] vLLM smoke completion failed", flush=True)
    sys.exit(1)
print("[loo] vLLM ready+smoke_ok", url, flush=True)
PY

  export base_url="${VLLM_URL}"
  export api_key="EMPTY"
  export model_name="${SERVED}"
  export MODEL_NAME="${SERVED}"

  # Drop previous failed partials so resume does not keep Connection errors
  if [[ -f "${JOB_DIR}/harness_rollouts.jsonl" ]]; then
    python - <<PY
import json
from pathlib import Path
p = Path("${JOB_DIR}/harness_rollouts.jsonl")
keep = []
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    err = row.get("error")
    if err in (True, "True", 1) or (isinstance(err, str) and err.strip()):
        continue
    msg = str(row.get("error_message") or row.get("exception") or "")
    if "Connection" in msg:
        continue
    keep.append(line)
p.write_text(("\n".join(keep) + ("\n" if keep else "")), encoding="utf-8")
print(f"[loo] kept {len(keep)} clean rows after purge", flush=True)
PY
  fi

  cd "${SCOPE_ROOT}"
  python training/rollout_harness_browsecomp.py \
    --model-path "${MODEL_PATH}" \
    --harness-config "${HARNESS_CONFIG}" \
    --split "${SPLIT}" \
    --limit "${LIMIT}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens 2048 \
    --temperature 0.0 \
    --max-model-len 32768 \
    --parallel "${PARALLEL}" \
    --reranker none \
    --retrieval bm25 \
    --bm25-index-path "${BM25}" \
    --output-dir "${JOB_DIR}" \
    --policy api \
    --no-manage-vllm \
    --vllm-url "${VLLM_URL}" \
    --vllm-model-name "${SERVED}" \
    --resume

  python - <<PY
import json
from pathlib import Path
job = Path("${JOB_DIR}")
path = job / "harness_rollouts.jsonl"
n = 0
n_err = 0
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        err = row.get("error")
        msg = str(row.get("error_message") or row.get("exception") or "")
        if err in (True, "True", 1) or (isinstance(err, str) and err.strip()) or "Connection" in msg:
            n_err += 1
rate = (n_err / n) if n else 1.0
print(f"[loo] n={n} n_err={n_err} error_rate={rate:.3f}", flush=True)
manifest = json.loads(Path("${MANIFEST}").read_text())
manifest["n_finished"] = n
manifest["n_error"] = n_err
manifest["error_rate"] = rate
manifest["ended_at"] = __import__("datetime").datetime.now().isoformat()
ok = n >= int("${LIMIT}") and rate <= float("${MAX_ERROR_RATE}")
manifest["status"] = "completed" if ok else "failed_quality"
Path("${MANIFEST}").write_text(json.dumps(manifest, indent=2) + "\n")
status = job / "STATUS_LIVE.md"
status.write_text(
    f"# STATUS_LIVE — ${JOB_NAME}\n\n"
    f"- n_finished: {n}\n- n_error: {n_err}\n- error_rate: {rate:.3f}\n"
    f"- quality_ok: {ok}\n",
    encoding="utf-8",
)
if ok:
    (job / "DONE").write_text("ok\n", encoding="utf-8")
    print("[loo] DONE quality_ok", flush=True)
else:
    (job / "DONE").unlink(missing_ok=True)
    raise SystemExit(f"quality gate failed n={n} err_rate={rate}")
PY
  echo "[$(date -Iseconds)] finished quality_ok"
} >>"${LOG}" 2>&1
