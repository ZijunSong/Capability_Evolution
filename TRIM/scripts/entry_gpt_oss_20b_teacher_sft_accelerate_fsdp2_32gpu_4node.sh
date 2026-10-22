#!/usr/bin/env bash
# === GPT-OSS-20B Teacher SFT — accelerate_fsdp2 train-only (2026-10-22) ===
#
# FSDP2 version of entry_gpt_oss_20b_teacher_sft_accelerate_ddp_32gpu_4node_20261021.sh:
# same sealed Teacher dataset and same common initial adapter, but the frozen
# BF16 base is sharded across the 32 gradient ranks (fully_shard per
# GptOssDecoderLayer) instead of replicated on every GPU. Checkpointing uses
# Distributed Checkpoint (sft_engine/sharded_checkpoint.py); the exported
# adapter is the aggregated LoRA only.
#
# Train-only: consumes DATASET.json + dataset.jsonl + SOURCE_CONTRACT.json +
# READY.json. It never collects or reseals.
#
# Smoke first: SFT_NUM_UPDATES=5 SFT_CKPT_MODE=off, verify load+5 steps+
# checkpoint round trip, then the formal 200-step run in a NEW OUT.

set -euo pipefail

# NOTE: image_command is executed via `bash -c` (no BASH_SOURCE); use the
# absolute lib path (same convention as the harness eval entries).
LIB=/mnt/songzijun/Capability_Evolution/TRIM/scripts/lib_sft_fsdp2_common.sh
test -f "$LIB"
# shellcheck source=lib_sft_fsdp2_common.sh
source "$LIB"

SEALED_DIR=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_sealed
test -f "$SEALED_DIR/DATASET.json"
test -f "$SEALED_DIR/READY.json"
test -f "$SEALED_DIR/dataset.jsonl"
test -f "$SEALED_DIR/SOURCE_CONTRACT.json"

INIT_ADAPTER=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current
test -f "$INIT_ADAPTER/adapter_config.json"
test -f "$INIT_ADAPTER/adapter_model.safetensors"
echo "INIT_ADAPTER_OK $INIT_ADAPTER"

SFT_CKPT_MODE="${SFT_CKPT_MODE:-off}"
SFT_NUM_UPDATES="${SFT_NUM_UPDATES:-200}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-10}"
SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-8192}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-32}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/fixed_runtime.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/backend.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/sharded_checkpoint.py

cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py \
  --nnodes "${NNODES}" \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas 1 \
  --tensor-parallel-size 1 \
  --rdzv-id "teacher-sft-fsdp2-${SFT_RUN_TAG}" \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --init-adapter "$INIT_ADAPTER" \
  --ce-backend accelerate_fsdp2 \
  --stage train \
  --sft-data "$SEALED_DIR/dataset.jsonl" \
  --dataset-manifest "$SEALED_DIR/DATASET.json" \
  --num-updates "$SFT_NUM_UPDATES" \
  --batch-size "$SFT_BATCH_SIZE" \
  --global-batch-size "$SFT_BATCH_SIZE" \
  --micro-batch-size 1 \
  --save-every "$SFT_SAVE_EVERY" \
  --learning-rate 5e-6 \
  --lora-rank 32 \
  --max-model-len "$SFT_MAX_MODEL_LEN" \
  --gradient-checkpointing "$SFT_CKPT_MODE" \
  --attn-implementation eager \
  --max-new-tokens 2048 \
  --seed 42 \
  --out "${OUT}"
