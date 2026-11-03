#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-9B，单节点 8xM402
# r4 = puiyr(r3) 用户确认后 kill 重提。r3 已跑通全链路并进入正式 evolve，但走的是
#      PyTorch 参考核（我重写 preflight 时误删了 flash-linear-attention/causal-conv1d
#      安装），日志持续刷 "falling back ... much slower"，吞吐损失数倍。
#      r4 补回内核依赖：fla==0.6.0 必装（镜像有纯 py wheel），causal-conv1d 尽力
#      （sdist-only，编译失败吃慢速回退），另补 openpyxl/swebench。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-9b-8gpu-20261102-r4 \
  --description "RSI phase-1 self-evolution (Qwen3.5-9B) on complete AppWorld: model as actor+self-proposer derives harness components from empty bank; 7xM402 panel workers + proposer GPU7. r4: r3 full pipeline OK but slow reference kernels (missing flash-linear-attention); restoring fla==0.6.0 + causal-conv1d + openpyxl/swebench. Artifacts runs/evolve_appworld_qwen35_9b_p1_r4/evolve/." \
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
