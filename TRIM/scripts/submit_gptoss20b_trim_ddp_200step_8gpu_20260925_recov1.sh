#!/usr/bin/env bash
# Submit Harness-1 BC+ full trim GPT-OSS-20B (original mxfp4, non-dequant)
# via torch_ddp_lora, 8x gm402-80g, 200 steps. Recovery resubmit 2026-09-25.
#
# WHY RESUBMIT: the previous run t-20260924232047-jjlcb FAILED@step0 in the
# teacher-projection phase with:
#   ValueError: projected action curate references docs not visible after clipping:
#   ['0000950103-25-007493_26', '0001213900-25-049013_54']
# Root cause (confirmed via git diff + mtime): jjlcb ran the pre-patch code tree
# where build_tinker_opd_datums (tinker_opd_datum.py:332) and
# build_projected_seed_datums (:489) called assert_action_visible() (HARD RAISE).
# A teacher curate projected to docs dropped by context clipping tripped the
# raise. 0xzst (identical config) hit the same query/doc but didn't trip it
# (rollout nondeterminism). The tolerant-skip patch (replace assert_action_visible
# with action_is_visible -> skip step, not abort) is now in the working tree
# (action_encoding.py:178 + tinker_opd_datum.py:332/489, py_compile verified).
# This resubmit runs the patched tree -> the same curate will be skipped, not
# abort. Fresh full run (not resume; jjlcb had no checkpoint at step0).
#
# Pure CLI flags; JuiceFS via --juicefs_mount_configs (juiceFsCluster=bj9-cloudml,
# the cluster-identity field — there is no literal "endpoint" key). image_command
# read verbatim from the entry script via "$(cat ...)" to avoid quote bugs.
#
# world_size=8 REQUIRED. Provenance audit (CLAUDE.md): verify by the three-tuple
# (torchrun --nproc_per_node 8 banner; 8 optimizer.rank0000..0007.pt after step0
# backward; 8 vllm_sessions _r0.._r7).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gptoss20b_trim_ddp_200step_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gptoss20b-trim-ddp-200step-8gpu-20260925-recov1 \
  --description "Harness-1 BC+ trim GPT-OSS-20B (original mxfp4) torch_ddp_lora 8x gm402-80g 200steps. Recovery of failed jjlcb: tolerant-skip patch now in tree (action_is_visible replaces assert_action_visible at tinker_opd_datum.py:332/489) so projected curate referencing clipped docs skips step not abort. mb1 max-turns40 max-model-len8192 gpu-mem0.90. Four-pack torch2.10.0/vllm0.19.1/tf4.57.6/pyserini2.0.0. Patches startup_timeout3600/4tuple/eaddrinuse. Model cp /tmp. Fresh run. world_size=8." \
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
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
