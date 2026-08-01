#!/usr/bin/env bash
pkill -f "training/scope_round3/hmin_v2_dup_rollout" 2>/dev/null || true
pkill -f "training/scope_round3/run_closed_loop_variant" 2>/dev/null || true
pkill -f "scripts/scope_round3/run_post_train_8gpu.sh" 2>/dev/null || true
pkill -f "scripts/scope_round3/run_all_8gpu.sh" 2>/dev/null || true
pkill -f "scripts/scope_round3/resume_post_train_8gpu.sh" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "vllm serve.*scope_round3" 2>/dev/null || true
pkill -f "vllm serve.*outputs/scope_round3" 2>/dev/null || true
sleep 2
echo "Killed round3 GPU processes"
