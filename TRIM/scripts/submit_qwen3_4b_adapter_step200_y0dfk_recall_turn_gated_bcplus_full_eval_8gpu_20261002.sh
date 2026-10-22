#!/usr/bin/env bash
# Submit TRIM Harness-1 bcplus_full eval: Qwen3-4B-Instruct-2507 + step200 LoRA (h3eb8 reward=final-answer-recall ONLY rl/zero 200step).
# component=zero, 8gpu, 830 queries. 8 单卡 vLLM actor (TP=1) + base+LoRA, local_bm25.
# h3eb8 adapter = step200 (step_000200/adapter, r=8 alpha=16 target q/k/v/o_proj).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_h3eb8_recall_only_transfer_eval_8gpu.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-qwen3-4b-adapter-step200-y0dfk-recall-turn-gated-eval-8gpu-20261002 \
  --description "TRIM bcplus_full eval: Qwen3-4B-Instruct-2507 step200 LoRA (y0dfk (vvptp) reward=recall+gated-turn-punish rl/zero 200step). component=zero 8gpu 830q. 8 single-GPU vLLM actors base+LoRA local_bm25." \
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
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline \
  --env CHROMA_DATABASE=local-offline \
  --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env ADAPTER_PATH=/mnt/songzijun/outputs/qwen3-4B-rl-recall-turn-punish-gated-32gpu_20260930/checkpoints/rl/step_000200/adapter \
  --env BENCHMARK=bcplus_full \
  --env RUN_ID=qwen3_4b_adapter_step200_y0dfk_recall_turn_gated_bcplus_full_8gpu_20261002_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
