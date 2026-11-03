#!/usr/bin/env bash
# RSI 第一阶段自进化（完整 AppWorld 单域）— Qwen3.5-9B，单节点 8xM402
# r5 = 9fhn3(r4) 失败后重提。r4 基础设施全绿（torch/cuda、appworld、fla 融合核），
#      管线跑完 3 代但 evaluated_candidates=0 / CANDIDATES_REPEATEDLY_INVALID。
#      根因是解析器不对称：停机判据用宽松的 extract_json_object（容忍散文前缀/思考块/
#      围栏、按括号深度配平），proposer 却用严格版（非贪婪围栏正则 + 必须整体首尾 {}），
#      于是"生成端判定完整、解析端说无 JSON"，18 次提议全废。
#      r5 修复：①parse_candidate_patch 与停机判据同源 ②from_json 畸形收成 CandidateInvalid
#      ③proposer_max_new_tokens 4096→12288（整组件 JSON 转义极易截断）④拒绝原因实时落
#      stdout + history 落盘（此前只在内存，跑完查无可查）。
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_appworld_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-evolve-appworld-qwen35-9b-8gpu-20261102-r5 \
  --description "RSI phase-1 self-evolution (Qwen3.5-9B) on complete AppWorld, 7xM402 panel + proposer GPU7. r5: fix proposer parser asymmetry (stop-criterion lenient vs parser strict -> 18/18 proposals rejected, evaluated_candidates=0), share extract_json_object, bound from_json errors, proposer_max_new_tokens 4096->12288, persist rejection history. Artifacts runs/evolve_appworld_qwen35_9b_p1_r5/evolve/." \
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
