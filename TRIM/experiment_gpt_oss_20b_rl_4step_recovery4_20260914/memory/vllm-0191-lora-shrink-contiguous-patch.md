---
name: vllm-0191-lora-shrink-contiguous-patch
description: vLLM 0.19.1 LoRA shrink op assert inputs.is_contiguous() 失败；现成补丁 patch_vllm_lora_shrink_contiguous.py
metadata: 
  node_type: memory
  type: reference
  originSessionId: a7fdecfd-6210-4e0d-b28f-e63a8cf05218
  modified: 2026-09-11T14:54:05.605Z
---

vLLM 0.19.1 在 rollout 阶段加载 LoRA adapter 时，`vllm/lora/ops/triton_ops/lora_shrink_op.py:177` 的 `assert inputs.is_contiguous()` 会失败（多个 Worker TP rank 同时崩，栈：`column_parallel_linear.forward → apply → punica_wrapper.add_lora_linear → add_shrink → lora_shrink`），任务直接 failed。

**触发条件**：rollout 用 `lora=<adapter_path>`（非 None）。step0 从 base（lora=None）开始则不触发；step≥1 或复用旧 --out 目录导致 rollout0 就加载 theta0 时触发。

**现成补丁**：`/mnt/songzijun/Capability_Evolution/TRIM/scripts/patch_vllm_lora_shrink_contiguous.py`，把 `assert inputs.is_contiguous()` 改为 `inputs = inputs.contiguous()`（断言→强制连续化）。用法：依赖安装后、preflight 前执行 `python3 TRIM/scripts/patch_vllm_lora_shrink_contiguous.py`。

**How to apply**：在 `run_train_cloudml_8gpu_trim_smoke.sh` 依赖安装段后、preflight 段前已加入补丁调用（recovery3 `t-20260911225347-k9qvn`）。任何用 vLLM 0.19.1 + LoRA rollout 的 CloudML 任务都必须在启动脚本里打这个补丁，否则 step≥1 rollout 必崩。补丁幂等（找不到 assert 会报错退出，可作 preflight）。

关联 [[trim-gptoss20b-cispo-backward-oom]]、[[trim-vllm-rollout-worker-disable-custom-all-reduce]]、[[cloudml-trim-launch-heredoc-juicefs-reread]]。
