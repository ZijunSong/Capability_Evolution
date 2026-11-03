#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-9B，单节点 8xM402
# r6 = 3b1bb(r5) 失败后重提。r5 的诊断留痕第一次给出确切拒绝原因：
#      `candidate patch malformed: KeyError('operation'); keys=['content','entrypoint',
#      'parent_bank_digest','path']` —— 模型吐的是**压平的文件对象**，不是 patch 封套
#      {operation, files[], manifest_patch}。机制：输出 schema 埋在 prompt 模板中段，
#      之后紧跟一大坨 EvolutionPacket JSON，模型照着"最近看到的形状"压平了。
#      r6 修复：①输出封套骨架重述到 prompt **最末**（近因效应）
#               ②窄口径 schema 规范化兜底（只抬 path/content 进封套，内容一字不造，
#                 无歧义才抬，留痕可审计）③拒绝原因继续实时落 stdout
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-9b-8gpu-20261102-r6 \
  --description "RSI phase-1 self-evolution (Qwen3.5-9B) on complete AppWorld, 7xM402 panel + proposer GPU7. r6: r5 diagnostics proved model emits flat file object {path,content,entrypoint} instead of patch envelope {operation,files,manifest_patch} (schema buried mid-prompt before a huge packet JSON). Fix: restate output envelope at prompt end + narrow auditable schema coercion. Artifacts runs/evolve_appworld_qwen35_9b_p1_r6/evolve/." \
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
