#!/usr/bin/env bash
# Submit the WebShop HarnessOPD PILOT (guide §9.1 E4).
# PRECONDITION: ALFWorld train-verify r6 must have completed G1-G5 —
# the pilot launches only after BOTH domains' gate chains are green.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
WS_VERDICT="$RSI_ROOT/runs/alfworld_trainverify_r1/train_verify_verdict.json"  # ALFWorld r6 verdict (green)
if [ -f "$WS_VERDICT" ]; then
  python3 - "$WS_VERDICT" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
assert v.get("G1_lora_changed") and v.get("G4_runtime_env_digests_nonempty") and v.get("G5_opd_consumed"), \
    f"WebShop train-verify gates not green: {v}"
print("precondition OK: WebShop train-verify green | opd_rows:", v.get("n_opd_rows_total"))
PY
else
  echo "WARNING: $WS_VERDICT not found — WebShop train-verify (dims6) still running."
  echo "guide §9.1: pilot launches after BOTH domains' gate chains complete. Aborting."
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_pilot_harness_opd_webshop_9b_32gpu_4node_20261115_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-pilot-harnessopd-webshop-9b-4n32g-20261115-r1 \
  --description "WebShop HarnessOPD PILOT (guide 9.1 E4): Qwen3.5-9B 4node32gpu, 50 steps, LoRA r32 cumulative. Student=H0 zero; Teacher=H0+confirmed bank bd397b10 (confirm reward +27.4pp / strict +12.5pp, probe admitted 34/34, readiness READY_FOR_PILOT). PR-05 hard gate ON. All campaign fixes in: per-row backward, tilelang, export payload. 32-rank data-parallel via torchrun." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
