#!/usr/bin/env bash
# 提交 harness_g_corpus_graph 构建任务（1 node 8 gpu，~10-20min，跑完即释放）
# 输出 .pkl 到 JuiceFS，4 个训练 job 的 entry 自动发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_build_harness_g_corpus_graph_8gpu_20261008.sh"
test -f "${ENTRY}"

JOB_NAME="build-harness-g-corpus-graph-8gpu-20261008"

echo "[submit] build corpus graph job=${JOB_NAME} ENTRY=${ENTRY}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "Build harness_g_corpus_graph.pkl from browsecomp_plus_corpus_full.jsonl (100195 docs) on 1-node 8gpu (224GiB mem). Dev box OOMs at 24GB; pod has enough. Output to JuiceFS SCOPE/external/BrowseComp-Plus/indexes/. ~10-20min." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
