#!/usr/bin/env bash
# Submit the AppWorld bounded self-evolution retry.
# PRECONDITION (guide §4.5): H0 v3 re-measure job must have PASSED its gates.
# Check before running: runs/h0v3_appworld_9b_r1/verdict.json gates == ["PASS"]
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
# accept either the 8-GPU run dir (r1) or the 4-GPU run dir (r1g4)
VERDICT="$RSI_ROOT/runs/h0v3_appworld_9b_r1/verdict.json"
[ -f "$VERDICT" ] || VERDICT="$RSI_ROOT/runs/h0v3_appworld_9b_r1g4/verdict.json"
if [ -f "$VERDICT" ]; then
  python3 - "$VERDICT" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
assert v.get("gates") == ["PASS"], f"H0 v3 gates not PASS: {v.get('gates')}"
print("precondition OK: H0 v3 re-measure gates PASS | paired_partial_delta_pp:", v.get("paired_partial_delta_pp"))
PY
else
  echo "WARNING: $VERDICT not found — H0 v3 re-measure has not completed."
  echo "guide §4.5 requires the H0 re-measure BEFORE the retry. Aborting."
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_appworld_retry_v1_9B_8gpu_20261109_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-appworld-retry-v1-9b-8gpu-20261109-r1 \
  --description "AppWorld bounded self-evolution retry (guide 4.5): Qwen3.5-9B 8gpu, contract v3 H0 (verified interface). FIXED budget frozen pre-launch: 4 gens x 2 candidates, max 8 evals, 1 repair/candidate, patience 2 — no extensions. End states labeled honestly (BASELINE_RETAINED/NO_CONFIRMED_GAIN/PROPOSAL_STALLED or champion pending fresh confirm). Mechanism coverage: API-doc discovery, small-step execution, completion verification. OUT runs/appworld_retry_r1." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 5 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
