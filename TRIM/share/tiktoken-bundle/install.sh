#!/usr/bin/env bash
# Install the offline Harmony / tiktoken vocab next to an approved runtime prefix.
set -euo pipefail
PREFIX="${1:-/opt/scape-easyopd-smoke7}"
HERE="$(cd "$(dirname "$0")" && pwd)"

TIK="$PREFIX/share/tiktoken"
RS="$PREFIX/share/tiktoken_rs_cache"
PY="$PREFIX/share/tiktoken_cache"
HF="$PREFIX/share/hf_tokenizer"

mkdir -p "$TIK" "$RS" "$PY" "$HF"
cp -a "$HERE/share/tiktoken/." "$TIK/"
cp -a "$HERE/tiktoken_rs_cache/." "$RS/"
cp -a "$HERE/tiktoken_cache/." "$PY/"
cp -a "$HERE/share/hf_tokenizer/." "$HF/"

# Default openai_harmony cache when TIKTOKEN_RS_CACHE_DIR is unset.
if mkdir -p /tmp/tiktoken-rs-cache 2>/dev/null; then
  cp -a "$HERE/tiktoken_rs_cache/." /tmp/tiktoken-rs-cache/
fi
if mkdir -p /tmp/data-gym-cache 2>/dev/null; then
  cp -a "$HERE/tiktoken_cache/." /tmp/data-gym-cache/
fi

cat <<ENV

Installed Harmony vocab under $PREFIX

Export these on EVERY openai_harmony / four-cell command:

  export TIKTOKEN_ENCODINGS_BASE=$TIK
  export TIKTOKEN_RS_CACHE_DIR=$RS
  export TIKTOKEN_CACHE_DIR=$PY

Verify:

  TIKTOKEN_ENCODINGS_BASE=$TIK \\
  TIKTOKEN_RS_CACHE_DIR=$RS \\
  TIKTOKEN_CACHE_DIR=$PY \\
    $PREFIX/bin/python $HERE/verify_offline.py

ENV
