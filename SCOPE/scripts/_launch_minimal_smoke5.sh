#!/usr/bin/env bash
set -euo pipefail
REPO=/data/ppnm/SCOPE
SMOKE_DIR=$REPO/outputs/minimal_runtime_browsecomp_full830/smoke5
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

export OUTPUT_DIR=$SMOKE_DIR
export LIMIT=5
export RESUME=0
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

cd "$REPO"
nohup bash scripts/rollout_minimal_runtime_browsecomp.sh > "$SMOKE_DIR/nohup_smoke.log" 2>&1 &
echo $! > "$SMOKE_DIR/nohup_smoke.pid"
echo "started pid=$(cat "$SMOKE_DIR/nohup_smoke.pid") at $(date)"
sleep 5
head -25 "$SMOKE_DIR/nohup_smoke.log"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
