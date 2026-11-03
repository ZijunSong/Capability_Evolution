#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-9B，单节点 8xM402
# r2 = u7mzf(r1) 失败后重提：修 torch/driver 不兼容（dep preflight 钉 torch==2.10.0
#       + transformers==5.17.0 + CUDA 硬门），换全新输出前缀 _r2，禁复用脏目录。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-9b-8gpu-20261102-r2 \
  --description "RSI phase-1 self-evolution (Qwen3.5-9B) on complete AppWorld (single domain, full harness_select panel, no 6-task sampling): model as actor+self-proposer derives harness components from empty bank; 7xM402 panel workers + proposer GPU7. r2: pinned torch==2.10.0 (cu128, gm402 driver match) + transformers==5.17.0 + CUDA fail-fast after r1 failed on torch 2.15.0+cu132 vs driver 12080. Artifacts runs/evolve_appworld_qwen35_9b_p1_r2/evolve/." \
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
