---
name: vllm-mxfp4-lora-marlin-ptx-gm402
description: vLLM 0.19.1 mxfp4 模型 + LoRA 在 gm402 上必崩 cudaErrorUnsupportedPtxVersion；解法=预 dequant mxfp4→bf16 checkpoint
metadata: 
  node_type: memory
  type: reference
  originSessionId: 5cd5c673-f50c-4fdf-81a3-91285e2d1ae0
  modified: 2026-09-12T10:11:44.662Z
---

GPT-OSS-20B（mxfp4 量化，config.json quantization_config.quant_method=mxfp4）在 vLLM 0.19.1 + gm402-80g 上，当 `enable_lora=True`（rollout 加载 LoRA adapter）时**必然崩溃**于 `cudaErrorUnsupportedPtxVersion`（the provided PTX was compiled with an unsupported toolchain）。

**根因（vLLM 0.19.1 已知架构限制）**：`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` 的 `select_mxfp4_moe_backend()` 在 LoRA 分支（`if config.is_lora_enabled:`）**硬编码**只选 `MARLIN` 或 `TRITON_UNFUSED`，提前 return，跳过 FlashInfer/CUTLASS/TRTLLM（后者才在 gm402/Hopper 上工作）。Marlin path 的 `marlin_utils_fp4.py:574 repack_weight: qweight = weight[i].view(torch.int32).T.contiguous()` 触发的 CUDA kernel PTX 与 gm402 driver 不兼容 → 8 卡 worker 同时崩。base 模型（lora=None）走非 LoRA path 选 FlashInfer backend，不触发，所以 step0 正常、step1+（lora=adapter）必崩。

**无可靠 runtime flag 禁用**（research agent 确认，2026-09-12）：
- `VLLM_MXFP4_USE_MARLIN=0` → 切 TRITON_UNFUSED，但需 `triton_kernels` 包（TRIM 环境未装），且 Triton mxfp4 LoRA path 用 `.tile::scatter4` PTX 也是 Hopper/SM10 特定，gm402 sm_90 未验证、脆弱。
- `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1` / `--moe-backend triton` → 只对非 LoRA path 生效，LoRA 分支提前 return 看不到。
- `--quantization None` → 不生效，模型 config.json 的 quantization_config 仍被检测。
- vLLM issue #34331 提的 `emulation_dequantize_weights` 被 maintainer mgoin 拒绝、未合并。
- 相关 issue：#30135（Marlin fallback）、#41477（scatter4 PTX）、#38022（gpt-oss+Marlin）。

**解法（推荐）**：预生成 bf16 checkpoint。用 HF transformers 4.57.6 `AutoModelForCausalLM.from_pretrained(mxfp4_path, dtype=torch.bfloat16, device_map="cuda:0")`（4.57.6 会 dequant mxfp4→bf16，log: "MXFP4 quantization requires Triton... we will default to dequantizing the model to bf16"），`model.save_pretrained(bf16_dir, safe_serialization=True)`，强制 drop config 的 quantization_config（设 None）。然后训练 `--model_name` 指向 bf16_dir：vLLM 走 bf16+LoRA 正常 path，无 mxfp4 Marlin，无 PTX 崩溃，rollout 仍快（vLLM batched）。bf16 20B ~40GB，8×80GB 充裕。脚本：`TRIM/scripts/dequant_gpt_oss_mxfp4_to_bf16.py`，dequant job YAML：`TRIM/cloudml_dequant_gpt_oss_20b_mxfp4_to_bf16_20260912_not_submitted.yaml`。

**Why:** step0（lora=None）总能过，step1+（lora=训练后 adapter）必崩——任何用 GPT-OSS-20B mxfp4 做 on-policy RL/trim 多步训练的任务，只要 step1+ 复用 vLLM+LoRA 就撞这个。不是 OOM/碎片，是确定性 PTX 不兼容。
**How to apply:** 见 [[trim-gptoss20b-cispo-backward-oom]]（OOM 是另一个独立问题，用 max_model_len 4096+mb=1 解；PTX 用本条 bf16 dequant 解）。两个修复叠加：dequant bf16 解 step1+ PTX，max_model_len 4096+mb=1 解 step0 backward OOM。`--rollout-backend hf` 也能绕 PTX 但 sequential generate 太慢不可行（~168min/chunk）。
