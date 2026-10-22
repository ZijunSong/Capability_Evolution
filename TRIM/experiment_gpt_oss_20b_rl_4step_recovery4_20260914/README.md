# GPT-OSS-20B RL 4-step 训练实验归档（2026-09-14）

## 最终成功任务
- **Job**: t-20260912184849-kjuu8 (recovery4)
- **state**: succeed, runTime 112126s (~31.1h)
- **输出**: outputs/trim_train_harness1_bcplus_full_gpt_oss_20b_8gpu_rl_4step_20260911_recovery4

## 三重修复
1. **bf16-dequant 模型** → 解 mxfp4 Marlin LoRA PTX（/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant, 41.8GB）
2. **max_model_len 4096 + mb=1** → 解 step0+跨 step backward OOM
3. **parse_generated_action wrapper fix**（four_cell_runtime.py:754）→ 解 step0 rollout TypeError
4. lora_shrink 补丁（patch_vllm_lora_shrink_contiguous.py）→ 防 vLLM LoRA assert

## 文件夹结构
- `yamls/` — 所有 YAML（original + recovery1-4 + dequant）
- `scripts/` — 提交脚本
- `patches/` — 代码补丁/脚本（dequant/lora_shrink/four_cell_runtime/parse_rollout_action/vllm_rollout_worker/rl_opd_types）
- `output_artifacts/` — 训练输出元数据 JSON（不含权重等大文件）
  - `vllm_sessions/` — 4 个 rollout 的 config/policy_digest/READY 等元数据
  - `adapters/` — theta0 + rl 的 adapter_config.json
- `logs/` — 恢复日志
- `memory/` — 相关记忆文件

## 不含的大文件（在原输出目录）
- adapters/rl/*.safetensors（训练后权重）
- adapters/theta0/*.safetensors
- vllm_sessions/*/result.json, job.json, vllm_worker.log
- /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/*.safetensors（bf16 模型 41.8GB）

## 失败任务（均保留）
- original t-20260911191937-y8uax: step0 backward OOM (mb=4)
- recovery1 t-20260911223014-skd0g: step0 backward OOM (mb=2)
- recovery2 t-20260912004216-cmcxl: step1 mxfp4 Marlin LoRA PTX
- recovery3 t-20260912181707-wnnhv: parse_generated_action TypeError
