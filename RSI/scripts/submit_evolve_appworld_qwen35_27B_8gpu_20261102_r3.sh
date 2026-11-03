#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-27B，单节点 8xM402
# r3 = npqqi(r2) FAILED 后重提。r2 preflight/panel 全通（3 轮 ~130min），但 proposer
#      18/18 提议全废（9fhn3 同款：extract_json_object 与 parse_candidate_patch 不同源，
#      拒绝原因又被旧 search.py 吞掉）。共享代码 proposer/search 已修（16:37/15:39），
#      r3 entry 加解析同源硬门 + accelerate 补齐 low-mem 路径。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_27B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-27b-8gpu-20261102-r3 \
  --description "RSI phase-1 self-evolution (Qwen3.5-27B) on complete AppWorld (full harness_select panel): 7xM402 panel workers + proposer GPU7. r3: r2 failed CANDIDATES_REPEATEDLY_INVALID (18/18 proposals wasted, 9fhn3 parse mismatch, silent rejects); shared proposer/search fix landed + entry adds parse same-source hard gate + accelerate. Artifacts runs/evolve_appworld_qwen35_27b_p1_r3/evolve/." \
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
