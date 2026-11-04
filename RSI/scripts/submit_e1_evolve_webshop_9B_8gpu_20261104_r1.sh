#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_e1_evolve_webshop_9B_8gpu_20261104_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-e1-evolve-webshop-9b-8gpu-20261104-r8 \
  --description "E1 9B webshop evolution (guide §12): 8 gens x 2 candidates, frozen 12+12 official-gated manifest, H0-ReAct base (E0 G1-G4 passed). r8: +data symlink alongside search_engine (n2bxo r7: load_products opens <SP>/data/items_ins_v2+items_human_ins via BASE_DIR too -> bare FileNotFoundError persisted). All BASE_DIR-external paths now symlinked; verified locally. OUT runs/e1_webshop_9b_r9." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
