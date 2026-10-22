#!/usr/bin/env bash
# Submit TRIM Harness-1 BC+ full Qwen3-4B "zero" eval (8x gm402-80g, queue 2334) to CloudML.
# 状态: NOT YET EXECUTED. 运行本脚本即消耗真实资源（8 张 gm402-80g）。
#
# 提交方式（按用户指定）：
#   - JuiceFS 通过直接 CLI 参数 --juicefs_mount_configs 显式传入 JSON，
#     不通过 YAML --filename 的 JuiceFS 字段提交；
#     endpoint / juiceFsCluster 均显式写为 "bj9-cloudml"。
#     （Go 后端实际消费 juiceFsCluster；endpoint 双写以明确意图，最稳。）
#   - 其余配置（image/command/env/resource/priority/tag）来自 --filename 指向的 YAML 审计底稿。
#
# 启动脚本 ImageCommand 调用：
#   /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_qwen3_4b_zero_eval_cloudml_8gpu.sh
#   该 wrapper 已补齐 tiktoken 三离线变量（TIKTOKEN_ENCODINGS_BASE /
#   TIKTOKEN_CACHE_DIR / TIKTOKEN_RS_CACHE_DIR），对齐已 DONE 的 gpt-oss zero eval。
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_trim_run_eval_harness1_bcplus_full_qwen3_4b_zero_8gpu_20260913.yaml"

# JuiceFS: volume=songzijun, cluster/endpoint=bj9-cloudml, mount at /mnt/songzijun, read-write.
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag "MemQA-General-V5"
