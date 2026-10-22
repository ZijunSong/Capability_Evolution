#!/usr/bin/env bash
# === GPT-OSS-20B Projected Teacher SFT — accelerate_fsdp2 train-only (2026-10-22) ===
#
# Consumes its OWN sealed dataset (experiment=projected_teacher_sft) produced by
# the Zero-replay prepare+seal chain. Never trains on Teacher rows: the Zero
# state replay is the definition of this ablation. Init adapter is the same
# common fresh LoRA as Teacher SFT — NOT a Teacher-trained checkpoint.
#
# Usage:
#   PROJECTED_DATA=/mnt/songzijun/outputs/.../projected_sealed/<id> \
#   SFT_NUM_UPDATES=5 OUT=... bash entry_..._projected_teacher_sft_accelerate_fsdp2_32gpu_4node.sh
#
# The prepare/seal stage is CPU work (prepare_projected_teacher_sft.py +
# seal_projected_teacher_sft_dataset.py); this entry only trains.

set -euo pipefail

# NOTE: image_command is executed via `bash -c` (no BASH_SOURCE); use the
# absolute lib path (same convention as the harness eval entries).
LIB=/mnt/songzijun/Capability_Evolution/TRIM/scripts/lib_sft_fsdp2_common.sh
test -f "$LIB"
# shellcheck source=lib_sft_fsdp2_common.sh
source "$LIB"

: "${PROJECTED_DATA:?set PROJECTED_DATA to the sealed projected_teacher_sft dataset directory}"
test -f "$PROJECTED_DATA/DATASET.json"
test -f "$PROJECTED_DATA/READY.json"
test -f "$PROJECTED_DATA/dataset.jsonl"
test -f "$PROJECTED_DATA/SOURCE_CONTRACT.json"
python3 - <<PY
import json
manifest = json.load(open("$PROJECTED_DATA/DATASET.json"))
assert manifest.get("experiment") == "projected_teacher_sft", manifest.get("experiment")
print("PROJECTED_DATASET_OK", {"experiment": manifest["experiment"], "n_rows": manifest.get("n_rows")}, flush=True)
PY

INIT_ADAPTER=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current
test -f "$INIT_ADAPTER/adapter_config.json"
test -f "$INIT_ADAPTER/adapter_model.safetensors"
echo "INIT_ADAPTER_OK $INIT_ADAPTER"

SFT_CKPT_MODE="${SFT_CKPT_MODE:-off}"
SFT_NUM_UPDATES="${SFT_NUM_UPDATES:-200}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-10}"
SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-8192}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-32}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_projected_teacher_sft_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/fixed_runtime.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/backend.py

cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_projected_teacher_sft_multinode.py \
  --nnodes "${NNODES}" \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas 1 \
  --tensor-parallel-size 1 \
  --rdzv-id "projected-sft-fsdp2-${SFT_RUN_TAG}" \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --init-adapter "$INIT_ADAPTER" \
  --ce-backend accelerate_fsdp2 \
  --stage train \
  --sft-data "$PROJECTED_DATA/dataset.jsonl" \
  --dataset-manifest "$PROJECTED_DATA/DATASET.json" \
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
