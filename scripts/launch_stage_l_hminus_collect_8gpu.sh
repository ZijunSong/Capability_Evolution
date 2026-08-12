#!/usr/bin/env bash
# Stage L: H_-m student rollouts for Candidate A/B + provisional OPD for B=verify.
# H100 imports NOT required (LOCAL_CAL64 provisional path).
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE_ROOT="${SCOPE_ROOT:-$(cd "${SCAPE_ROOT}/../SCOPE" && pwd)}"
OUT_ROOT="${OUT_ROOT:-$SCAPE_ROOT/outputs/stage_l_hminus_data}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-512}"
SEL="${SCAPE_ROOT}/outputs/scape_prestage/CANDIDATE_SELECTION.json"

CAND_A=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["A"]["component_id"])' "$SEL")
CAND_B=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["B"]["component_id"])' "$SEL")
echo "[stageL] A=${CAND_A} B=${CAND_B} limit=${LIMIT}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids" \
  "${SCAPE_ROOT}/outputs/stage_l/logs"

launch_collect() {
  local gpu=$1 name=$2 comp=$3
  local job_dir="${OUT_ROOT}/${name}"
  mkdir -p "${job_dir}/trajectories"
  if [[ -f "${job_dir}/DONE" ]]; then
    echo "[stageL] skip done ${name}"
    return 0
  fi
  local log="${OUT_ROOT}/logs/${name}.log"
  local pf="${OUT_ROOT}/pids/${name}.pid"
  nohup env \
    GPU="$gpu" JOB_NAME="$name" COMPONENT="$comp" \
    OUT_ROOT="$OUT_ROOT" LIMIT="$LIMIT" MODEL_PATH="$MODEL_PATH" \
    PARALLEL=1 SAVE_FULL_TRAJECTORIES=1 SAVE_TRAJECTORIES=1 \
    TRAJECTORY_SAVE_PATH="${job_dir}/trajectories" \
    bash "${SCAPE_ROOT}/scripts/run_loo_worker.sh" \
    >"$log" 2>&1 &
  echo $! >"$pf"
  echo "[stageL] collect ${name} gpu=${gpu} pid=$(cat "$pf")"
}

# GPU0: A H_-m collect 512
# GPU1: B H_-m collect 512
launch_collect 0 "A_${CAND_A}" "$CAND_A"
sleep 40
launch_collect 1 "B_${CAND_B}" "$CAND_B"

# GPU2-5: provisional OPD smoke+train for Candidate B (verify_tool) via SCOPE stack
# (closest existing real trainer; labeled provisional_scape_stage_l_B)
# train_opd supports --vllm-url (not --vllm-port/--tensor-parallel-size); see run_stage_l_b_verify_opd.sh
OPD_OUT="${SCAPE_ROOT}/outputs/stage_l/B_verify_opd_provisional"
mkdir -p "${OPD_OUT}" "${SCAPE_ROOT}/outputs/stage_l/logs"
if [[ ! -f "${OPD_OUT}/DONE" ]]; then
  nohup env     SCOPE_ROOT="${SCOPE_ROOT}"     MODEL_PATH="${MODEL_PATH}"     OPD_OUT="${OPD_OUT}"     CUDA_VISIBLE_DEVICES=2,3,4,5     LOG="${SCAPE_ROOT}/outputs/stage_l/logs/B_verify_opd.log"     bash "${SCAPE_ROOT}/scripts/run_stage_l_b_verify_opd.sh"     >"${SCAPE_ROOT}/outputs/stage_l/logs/B_verify_opd.log" 2>&1 &
  echo $! >"${SCAPE_ROOT}/outputs/stage_l/pids_B_opd.pid"
  echo "[stageL] B provisional OPD on GPU2-5 pid=$(cat "${SCAPE_ROOT}/outputs/stage_l/pids_B_opd.pid")"
fi

# GPU6-7: reserved — will attach A OPD / same-state CE baseline after A trajectories exist
cat > "${SCAPE_ROOT}/outputs/stage_l/STATUS_LIVE.md" <<EOF
# STATUS_LIVE — Stage L

- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')
- H100_required: NO (provisional LOCAL_CAL64 path)
- Candidate A: ${CAND_A} — H_-m collect on GPU0 (limit ${LIMIT})
- Candidate B: ${CAND_B} — H_-m collect on GPU1 + provisional SCOPE-OPD on GPU2-5
- GPU6-7: reserved for A OPD / baselines
- model: ${MODEL_PATH}
- retrieval: bm25_provisional
EOF

cp "${SCAPE_ROOT}/outputs/stage_l/STATUS_LIVE.md" "${SCAPE_ROOT}/outputs/STATUS_LIVE.md"
echo "[stageL] launch complete"
