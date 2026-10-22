#!/usr/bin/env bash
# 提交脚本：TRIM Harness-1 BC+ full / Gemma-3-27B-it / ALL eval / 8gpu
# 日期 2026-09-15，标签 MemQA-General-V5，priority 8，queue 2334 (gm402-80g)。
#
# 用法：bash submit_cloudml_eval_bcplus_full_gemma3_27b_all_8gpu_20260915_memqa_v5.sh
#
# JuiceFS 通过直接 CLI 参数 --juicefs_mount_configs 显式 JSON 传入（不用 YAML --filename 方式）：
#   volume=songzijun, endpoint=bj9-cloudml, JuiceFSCluster=bj9-cloudml, mount=/mnt/songzijun, read-write。
# JSON 字符串与 2026-09-13/15 gpt-oss/Qwen3.5 all eval 已验证提交脚本完全一致
# （Go client keys off juiceFsCluster，endpoint 双写保险）。
#
# 本脚本不自动执行提交——需要人工 review 后再运行；当前仅为生成提交命令。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="${SCRIPT_DIR}/cloudml_custom_train_harness1_bcplus_full_gemma3_27b_all_8gpu_20260915_memqa_v5.yaml"

# 确认当前 context（只读，不修改）
cml config show

# JuiceFS 挂载 JSON（endpoint 必须为 bj9-cloudml）
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

echo "=================================================================="
echo "[submit] YAML=${YAML}"
echo "[submit] juicefs_mount_configs=${JUICEFS_JSON}"
echo "[submit] tag=MemQA-General-V5  priority=8  queue=2334 (gm402-80g)  resourceNumber=8"
echo "=================================================================="

cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag "MemQA-General-V5"
