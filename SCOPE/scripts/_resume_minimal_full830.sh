#!/usr/bin/env bash
set -euo pipefail
REPO=/data/ppnm/SCOPE
OUT=$REPO/outputs/minimal_runtime_browsecomp_full830
mkdir -p "$OUT"

n=$(wc -l < "$OUT/harness_rollouts.jsonl" | tr -d ' ')
echo "[resume] existing episodes=$n in $OUT/harness_rollouts.jsonl"

export OUTPUT_DIR=$OUT
export LIMIT=0
export RESUME=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_PORT=8772
export PARALLEL=2
export SPLIT=all
export HARNESS_CONFIG=$REPO/harness/configs/modules_minimal.yaml
export USE_LEGACY_API_AGENT=0
export V8D_SUBTRACTIVE_CURATION=0
export V8D_IMPORTANCE_TAGGING=0
export V8D_AUTO_POPULATE_FIRST_SEARCH=0
export V8D_EVIDENCE_GRAPH=0
export V8D_SENTENCE_COMPRESS=0
export V8D_CONTENT_DEDUP=0
export V8D_VERIFY_TOOL=0
export V8D_TOKEN_BUDGET_MARKER=0
export V8D_CHUNK_NEIGHBORS=0
export ABLATE_VERIFY_UNAVAILABLE=1
export ABLATE_REVIEW_DOCS_UNAVAILABLE=1
export CHAT_MIN_TURNS_BEFORE_END=8
export CHAT_MIN_CURATED_BEFORE_END=1
export CHAT_MAX_WM_CHARS=18000
export CHAT_MAX_RECENT_TURNS=4
export MAX_TURNS=35
export MAX_TOKENS=2048
export TEMPERATURE=1.0
export MAX_MODEL_LEN=32768
export RERANKER=none
export RETRIEVAL=bm25

cd "$REPO"
# rotate old log but keep history
if [[ -f "$OUT/nohup_rollout.log" ]]; then
  mv "$OUT/nohup_rollout.log" "$OUT/nohup_rollout.resume_$(date +%Y%m%d_%H%M%S).log"
fi

nohup bash scripts/rollout_minimal_runtime_browsecomp.sh > "$OUT/nohup_rollout.log" 2>&1 &
echo $! > "$OUT/nohup_rollout.pid"
echo "started pid=$(cat "$OUT/nohup_rollout.pid") at $(date)"
sleep 8
head -35 "$OUT/nohup_rollout.log"
