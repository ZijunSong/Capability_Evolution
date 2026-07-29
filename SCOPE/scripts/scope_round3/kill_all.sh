#!/usr/bin/env bash
pkill -f "training/scope_round3/hmin_v2_dup_rollout" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "train_sdi_dup" 2>/dev/null || true
echo "Killed round3 GPU processes"
