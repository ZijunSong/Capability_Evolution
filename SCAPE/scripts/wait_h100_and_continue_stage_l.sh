#!/usr/bin/env bash
# Watch imports/h100_* for V2 JSONs; when complete, select candidates and launch Stage L (true SCAPE).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUT_L="${SCAPE_ROOT}/outputs/true_scape_stage_l"
LOG="${SCAPE_ROOT}/logs/true_scape_smoke/wait_h100_stage_l.log"
POLL_SEC="${POLL_SEC:-120}"

mkdir -p "${OUT_L}" "$(dirname "$LOG")" \
  "${SCAPE_ROOT}/imports/h100_1" \
  "${SCAPE_ROOT}/imports/h100_2" \
  "${SCAPE_ROOT}/imports/h100_3" \
  "${SCAPE_ROOT}/imports/h100_4"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

need=(
  "${SCAPE_ROOT}/imports/h100_1/CONTRIBUTION_CONFIRM.json"
  "${SCAPE_ROOT}/imports/h100_2/LOO_REPLICATION_V2.json"
  "${SCAPE_ROOT}/imports/h100_3/REAL_INFLUENCE_BY_COMPONENT.json"
  "${SCAPE_ROOT}/imports/h100_4/CANDIDATE_RECOMMENDATION_FOR_H20.json"
)

log "watcher start; waiting for H100 V2 imports"

while true; do
  missing=0
  for f in "${need[@]}"; do
    [[ -f "$f" ]] || missing=1
  done

  if [[ $missing -eq 1 ]]; then
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/select_candidates_v2.py" \
      --imports-root "${SCAPE_ROOT}/imports" \
      --out "${SCAPE_ROOT}/outputs/scape_prestage_v2" >/dev/null || true
    log "still waiting for H100 JSON imports"
    sleep "${POLL_SEC}"
    continue
  fi

  log "H100 imports present — selecting candidates"
  if ! "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/select_candidates_v2.py" \
      --imports-root "${SCAPE_ROOT}/imports" \
      --out "${SCAPE_ROOT}/outputs/scape_prestage_v2"; then
    log "selection failed / no passing candidates"
    sleep "${POLL_SEC}"
    continue
  fi

  A=$("${PYTHON_BIN}" -c "import json;print(json.load(open('${SCAPE_ROOT}/outputs/scape_prestage_v2/CANDIDATE_SELECTION_V2.json'))['Candidate_A'] or '')")
  B=$("${PYTHON_BIN}" -c "import json;print(json.load(open('${SCAPE_ROOT}/outputs/scape_prestage_v2/CANDIDATE_SELECTION_V2.json'))['Candidate_B'] or '')")
  log "Candidate A=${A} B=${B}"
  if [[ -z "$A" || -z "$B" ]]; then
    log "top2 incomplete; will retry"
    sleep "${POLL_SEC}"
    continue
  fi

  # Stage L: Group A=GPU0-3 Candidate A; Group B=GPU4-7 Candidate B
  # Sequence per group: 512 → 2k → 8k → heldout (seed42), then 2k/heldout seed43
  launch_cand() {
    local cand="$1" gpus="$2" tag="$3"
    screen -dmS "scape_L_${tag}" bash -c "
      source /data/ppnm/miniconda3/etc/profile.d/conda.sh
      conda activate bishop
      export PYTHONPATH='${SCAPE_ROOT}'
      export CUDA_VISIBLE_DEVICES='${gpus}'
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      cd '${SCAPE_ROOT}'
      for n in 512 2048 8192; do
        out='${OUT_L}/${tag}_n'\${n}'_seed42'
        mkdir -p \"\$out\"
        '${PYTHON_BIN}' -m scape.training.train_tool_opd \
          --component-id '${cand}' \
          --n-samples \$n \
          --seed 42 \
          --base-checkpoint '${MODEL_PATH}' \
          --out \"\$out\" \
          --no-dry-run \
          --epochs 1 \
          2>&1 | tee \"\$out/train.log\"
      done
      out='${OUT_L}/${tag}_heldout_seed42'
      mkdir -p \"\$out\"
      '${PYTHON_BIN}' -m scape.training.train_tool_opd \
        --component-id '${cand}' --n-samples 256 --seed 42 \
        --base-checkpoint '${MODEL_PATH}' --out \"\$out\" --no-dry-run --epochs 1 \
        2>&1 | tee \"\$out/train.log\"
      out='${OUT_L}/${tag}_n2048_seed43'
      mkdir -p \"\$out\"
      '${PYTHON_BIN}' -m scape.training.train_tool_opd \
        --component-id '${cand}' --n-samples 2048 --seed 43 \
        --base-checkpoint '${MODEL_PATH}' --out \"\$out\" --no-dry-run --epochs 1 \
        2>&1 | tee \"\$out/train.log\"
      touch '${OUT_L}/${tag}_ALL_DONE'
    "
  }

  launch_cand "$A" "0,1,2,3" "A"
  launch_cand "$B" "4,5,6,7" "B"
  log "launched Stage L screens scape_L_A / scape_L_B"
  touch "${OUT_L}/STAGE_L_LAUNCHED"
  exit 0
done
