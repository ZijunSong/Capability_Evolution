#!/usr/bin/env bash
# Wait for model + SFT conversion, then launch C0 8-GPU + monitor.
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SCAPE_ROOT}/outputs/0814_clean_mechanism"
MODEL=/data/ppnm/models/gpt-oss-20b
LOG="${OUT}/logs/orchestrate_c0.log"
mkdir -p "${OUT}/logs" "${OUT}/pids"

wait_ok() {
  local f="$1"
  local min_bytes="${2:-1}"
  while true; do
    if [[ -f "$f" ]]; then
      local sz
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      if (( sz >= min_bytes )); then
        echo "[$(date -Iseconds)] ready $f bytes=$sz"
        return 0
      fi
    fi
    echo "[$(date -Iseconds)] waiting $f (need >= ${min_bytes} bytes)"
    sleep 30
  done
}

{
  echo "[$(date -Iseconds)] orchestrate start"
  wait_ok "${OUT}/data/CLEAN_SFT_CONVERT.json" 10
  echo "[$(date -Iseconds)] convert ready"
  wait_ok "${MODEL}/model-00000-of-00002.safetensors" 4792272488
  wait_ok "${MODEL}/model-00001-of-00002.safetensors" 4798702184
  wait_ok "${MODEL}/model-00002-of-00002.safetensors" 4170342232
  echo "[$(date -Iseconds)] model shards ready — launching C0"
  OUT_ROOT="${OUT}" bash "${SCAPE_ROOT}/scripts/launch_0814_clean_c0.sh"
  echo "[$(date -Iseconds)] launching monitor"
  nohup bash "${SCAPE_ROOT}/scripts/monitor_0814_clean.sh" >>"${OUT}/logs/monitor.log" 2>&1 &
  echo $! >"${OUT}/pids/monitor.pid"
  echo "[$(date -Iseconds)] C0+monitor launched"
} >>"${LOG}" 2>&1
