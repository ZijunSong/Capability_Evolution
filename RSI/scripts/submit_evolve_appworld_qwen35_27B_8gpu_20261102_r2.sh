#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-27B，单节点 8xM402
# r2 = ntoow(r1) 用户确认后 kill 重提。r1 旧 preflight 装出 torch 2.15.0+cu132、
#      fla 内核缺失（导入名 flash_linear_attention 不存在）走慢速参考实现；
#      r2 套 9B r4 修复 preflight：torch2.10/tf5.17 钉死 + CUDA 硬门 +
#      fla==0.6.0/causal-conv1d(--no-build-isolation 重试) + appworld 自愈导入。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_27B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-27b-8gpu-20261102-r2 \
  --description "RSI phase-1 self-evolution (Qwen3.5-27B) on complete AppWorld (single domain, full harness_select panel): model as actor+self-proposer derives harness components from empty bank; 7xM402 panel workers + proposer GPU7. r2: r1 killed - old preflight pulled torch 2.15+cu132 and missed fla kernels (wrong import name); now 9B-r4 fixed preflight (torch2.10/tf5.17 pin + CUDA hard gate + fla/causal-conv1d + appworld self-heal imports). Artifacts runs/evolve_appworld_qwen35_27b_p1_r2/evolve/." \
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
