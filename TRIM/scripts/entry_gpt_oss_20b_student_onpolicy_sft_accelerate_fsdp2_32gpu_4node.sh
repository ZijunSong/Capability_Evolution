#!/usr/bin/env bash
# === GPT-OSS-20B Student on-policy SFT — accelerate_fsdp2 phase_jobs (2026-10-22) ===
#
# Consumes ONE sealed round (--resume-round) and performs ONE CE update, then
# exits. This is not a 200-round loop: an outer controller re-submits the next
# round after the rollout side seals it. The phase decision file is per-round
# (phase_decision_<round>.json) and a COMMITTED-but-unconsumed round only gets
# its bookkeeping repaired — it never trains twice (11.6).
#
# Usage:
#   ROUND0=/mnt/songzijun/outputs/.../rounds/r000000 \
#   SFT_NUM_UPDATES=200 OUT=... bash entry_..._student_onpolicy_sft_accelerate_fsdp2_32gpu_4node.sh

set -euo pipefail

# NOTE: image_command is executed via `bash -c` (no BASH_SOURCE); use the
# absolute lib path (same convention as the harness eval entries).
LIB=/mnt/songzijun/Capability_Evolution/TRIM/scripts/lib_sft_fsdp2_common.sh
test -f "$LIB"
# shellcheck source=lib_sft_fsdp2_common.sh
source "$LIB"

: "${ROUND0:?set ROUND0 to a sealed student round directory (targets.json present)}"
test -f "$ROUND0/targets.json"
python3 - <<PY
import json
targets = json.load(open("$ROUND0/targets.json"))
rows = targets.get("rows") or []
assert rows, "sealed student round has no rows"
assert all(row.get("experiment") == "student_onpolicy_sft" for row in rows)
print("STUDENT_ROUND_OK", {"n_rows": len(rows), "policy": (targets.get("policy") or {}).get("digest", "")[:16]}, flush=True)
PY

INIT_ADAPTER=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current
test -f "$INIT_ADAPTER/adapter_config.json"
test -f "$INIT_ADAPTER/adapter_model.safetensors"
echo "INIT_ADAPTER_OK $INIT_ADAPTER"

SFT_CKPT_MODE="${SFT_CKPT_MODE:-off}"
SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-8192}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-32}"
SFT_QUERIES_PER_ROUND="${SFT_QUERIES_PER_ROUND:-64}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_student_onpolicy_sft_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/online_runtime.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/backend.py

cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_student_onpolicy_sft_multinode.py \
  --nnodes "${NNODES}" \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas 1 \
  --tensor-parallel-size 1 \
  --rdzv-id "student-sft-fsdp2-${SFT_RUN_TAG}" \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --init-adapter "$INIT_ADAPTER" \
  --ce-backend accelerate_fsdp2 \
  --online-layout phase_jobs \
  --resume-round "$ROUND0" \
  --group-size 2 \
  --queries-per-round "$SFT_QUERIES_PER_ROUND" \
  --states-per-trajectory -1 \
  --num-updates 200 \
  --batch-size "$SFT_BATCH_SIZE" \
  --global-batch-size "$SFT_BATCH_SIZE" \
  --micro-batch-size 1 \
  --save-every 1 \
  --learning-rate 5e-6 \
  --lora-rank 32 \
  --max-model-len "$SFT_MAX_MODEL_LEN" \
  --gradient-checkpointing "$SFT_CKPT_MODE" \
  --attn-implementation eager \
  --max-new-tokens 2048 \
  --seed 42 \
  --out "${OUT}"
