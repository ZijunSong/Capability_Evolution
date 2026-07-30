#!/usr/bin/env bash
# Round 5 status dashboard
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"

echo "=== Round 5 Status $(date -Is) ==="

if [[ -f "${SUPERVISOR_PID}" ]]; then
  spid="$(cat "${SUPERVISOR_PID}")"
  if kill -0 "${spid}" 2>/dev/null; then
    echo "supervisor: RUNNING pid=${spid}"
  else
    echo "supervisor: stopped (stale pid ${spid}) — run: bash scripts/scope_round5/run_nohup.sh"
  fi
else
  echo "supervisor: not started — run: bash scripts/scope_round5/run_nohup.sh"
fi

echo "pipeline_stage: $(scope5_get_stage)"

echo ""
for gate in B1_PASS B2_PASS B3_PASSED_OBJECTIVES B4_PASS B5_COMPLETE B6_COMPLETE ROUND5_COMPLETE; do
  f="${OUT}/${gate}"
  if [[ -f "${f}" ]]; then
    echo "${gate}: $(tr -d '\n' < "${f}")"
  else
    echo "${gate}: (pending)"
  fi
done

echo ""
echo "--- B4 train ($(scope5_b4_done_count)/6 DONE, $(scope5_b4_train_count) running) ---"
for tag in o7_r64_seed42 o7_r64_seed43 o7_r64_seed44 compact_json_seed42 compact_json_seed43 compact_json_seed44; do
  if [[ -f "${OUT}/b4_full/${tag}/DONE" ]]; then
    echo "${tag}: DONE"
  elif pgrep -f "run_b4_train.py --variant.*${tag%%_seed*}" >/dev/null 2>&1; then
    echo "${tag}: RUNNING"
  else
    echo "${tag}: pending"
  fi
done

echo ""
echo "--- B3 micro-overfit ---"
for obj in O7 O1 O0; do
  d="${OUT}/micro_overfit/${obj}"
  [[ -f "${d}/summary.json" ]] && python3 -c "import json; s=json.load(open('${d}/summary.json')); print('${obj}: all_pass='+str(s.get('all_pass')))" 2>/dev/null || true
done

echo ""
echo "tail: tail -f ${SUPERVISOR_LOG}"
echo "resume: bash scripts/scope_round5/run_nohup.sh"
