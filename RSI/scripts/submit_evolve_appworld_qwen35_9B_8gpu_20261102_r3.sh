#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-9B，单节点 8xM402
# r3 = fcr8u(r2) 失败后重提。r2 已修好 torch/driver（实测 torch 2.10.0+cu128 cuda
#       True devices 8），死在 appworld 缺 urllib3（r1 里它被无钉 transformers 的
#       依赖树顺带带上，r2 钉死后漏了）。r3 显式补 appworld 传递依赖 + 导入自愈重试。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-9b-8gpu-20261102-r3 \
  --description "RSI phase-1 self-evolution (Qwen3.5-9B) on complete AppWorld: model as actor+self-proposer derives harness components from empty bank; 7xM402 panel workers + proposer GPU7. r3: torch==2.10.0+cu128 pin (r2 verified cuda True devices 8) + appworld transitive deps + import self-heal. Artifacts runs/evolve_appworld_qwen35_9b_p1_r3/evolve/." \
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
