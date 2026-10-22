# CloudML TRIM GPT-OSS-20B RL 4-step 训练 OOM 失败与恢复记录 (2026-09-11)

## 失败任务
- **Job**: `t-20260911191937-y8uax`（failed，runTime=7200s）
- job_name `trim-harness1-bcplus-full-gpt-oss-20b-8gpu-rl-4step-20260911`
- 队列 2334，8×gm402-80g，GUARANTEED_PUBLIC，priority 8，tag MemQA-General-V5
- 输出: `Capability_Evolution/TRIM/outputs/trim_train_harness1_bcplus_full_gpt_oss_20b_8gpu_rl_4step_20260911`
- **保留 failed attempt 不覆盖、不删除**，recovery 用全新 job name + 新 --out

## 失败根因（已定位）
- step0 rollout **成功完成**（19:42→21:19，~1h37min，256 episodes 多轮 agent，40 batches vLLM generate，output ~2488 toks/s）
- 21:19:44 `Loading checkpoint shards`（HF 加载 GPT-OSS-20B 做 optim）
- 21:19:49 `optim_step_start step=1 n_steps=1 cell=rl micro_batch_size=4 n_rl_datums=10058 n_opd_datums=0`
- 21:19:54–21:20:19 `hf_fb phase=cispo` done 1→16→24（total 10058）
- 21:20:23 `hf_rl_opd_client.py:191 _cispo_backward` → `hf_tool_opd.py:372 _response_position_logits` → `_forward` → peft → accelerate hooks → `gpt_oss/modeling_gpt_oss.py:130 torch.bmm(hidden_states, self.gate_up_proj)` → **OOM**
- GPU 1 撑满：79.15/79.18 GiB in use，76.30 GiB PyTorch allocated，1.99 GiB reserved unallocated，仅剩 20 MiB，Tried to allocate 858 MiB

**根因**: GPT-OSS-20B 是 MoE，cispo backward 在 expert `gate_up_proj` 的 `torch.bmm` 激活峰值 OOM。gradient checkpointing 默认已开（`--no-gradient-checkpointing` 才禁用），无法再靠激活重算省。micro_batch_size=4 + 长 context（max-model-len 8192，单 datum 长）下激活峰值超单卡 79GB。

**加重因素**: 日志无 vLLM stop 事件——rollout 完成后 vLLM TP8 引擎可能未及时释放显存，HF optim 又在同卡加载完整模型，双重占用。但 HF 模型用 device_map="auto" 分片，OOM 集中在 GPU 1 单卡（某 MoE 层激活集中）。

## 关键认知（修正之前的 ETA）
- `train_steps=min(...,1)` 只在 `--smoke` 时触发（line 2593），用户无 --smoke，**train_steps=4 不截断**
- 日志 `optim_step_start step=1 n_steps=1` 的 n_steps=1 是单次 optim 内部 sub-steps（line 2195 train_cell train_steps=1），不是总 train_steps
- 结构: 4 个 RL step（line 2153 n_train=args.train_steps=4, line 2170 for step in range(n_train)），每 step = 1 rollout + 1 optim(n_steps=1)
- 当前失败在第 1 个 optim（对应 rollout step0）

## Recovery 方案（保守、最小改动、不改 scorer/数据，符合 CLAUDE.md）
1. `--train-micro-batch-size 4` → **2**（降 cispo backward 激活峰值约一半，最直接）
2. `--gpu-memory-utilization 0.90` → **0.55**（大幅压 vLLM 占用，给 HF optim 留 ~35GB/卡）
3. 加 `PYTORCH_ALLOC_CONF=expandable_segments:True`（OOM 提示明确建议，减碎片；optim 阶段 vLLM 已停，不与 vLLM memory pool 冲突）
4. 其余全部不变（train_method=rl, train-data=sec, train-steps=4, train-env=local_legacy, max-turns 40, max-model-len 8192, 四件套版本, VLLM_DISABLE_CUSTOM_ALL_REDUCE=1, tiktoken, JVM, ANSERINI_CLASSPATH, rl_opd_types dataclass fix, local_legacy）
5. 全新 job name `...recovery1` + 新 --out `...rl_4step_20260911_recovery1`

**参数变化已分支记录**（micro-batch 4→2、vLLM util 0.90→0.55、+expandable_segments），与原 failed attempt 隔离，不混入。

## Recovery1 `t-20260911223014-skd0g` 结果：OOM 复现（2026-09-12 00:42）

- **mb=2 部分有效但不够**。step0 rollout 成功（~1h37min, 40 batches）→ 00:25 `optim_step_start mb=2 n_rl_datums=9884` → cispo backward **稳定推进到 done=424**（远超原崩点 24，elapsed 719s，student_fwd 339s 长序列区间）→ **00:40:48 OOM 复现**：Tried to allocate 1.93 GiB，GPU1 79.10/79.18GiB，PyTorch 67.70GiB allocated + **10.53GiB reserved-but-unallocated（碎片）**，仅剩 72MiB。
- mb=2 把 PyTorch 占用从 mb=4 的 76.75GiB 降到 67.70GiB（省 9GiB，越过 done=24 跑到 done=424），但**长 tail 序列 bucket + 10.53GiB 碎片**仍导致 1.93GiB 大块分配失败。与 smoke mb=1 OOM 模式完全一致（碎片化是主因，非峰值）。
- **结论**：调 micro_batch/gpu_mem 不足以解 GPT-OSS-20B MoE OOM，需结构性方案。
- recovery1 failed（runTime=7802），保留不覆盖。

## Recovery2 `t-20260912004216-cmcxl`：结构性修复（2026-09-12 00:42 提交）

对齐 smoke 成功的 recovery4 配置（mb=1 + max_model_len 4096 backward OK）：
1. `--train-micro-batch-size 2` → **1**
2. `--max-model-len 8192` → **4096**（结构性降 backward datum seq len：`four_cell_runtime.py:1749 max_full_tokens=max_model_len` → `hf_tool_opd.py:346 max_len`，降 MoE expert gate_up_proj bmm 的 hidden_states seq 维度）
3. 加 `python3 scripts/patch_vllm_lora_shrink_contiguous.py`（vLLM 0.19.1 step1+ LoRA assert is_contiguous 补丁，见 memory vllm-0191-lora-shrink-contiguous-patch）
4. `--gpu-memory-utilization 0.55`（保持）
5. 全新 job name `...recovery2` + 新 --out `...rl_4step_20260911_recovery2`
6. 其余不变（rl 4-step, local_legacy, train-data sec, max-turns 40, 四件套版本, tiktoken, JVM, ANSERINI_CLASSPATH, VLLM_DISABLE_CUSTOM_ALL_REDUCE=1, rl_opd_types dataclass fix）
- expandable_segments 不设（CLAUDE.md 警告 vLLM 兼容；max_model_len 降半应足够，对齐 smoke 成功配置）
- 00:42 提交，直接 running 无排队。高频每5分钟监控（cron 0093906f）。
- **关键验证点**：recovery2 step0 optim cispo backward 能否跑完 9884 datums。若仍 OOM，recovery3 加 max_model_len 3072 或 expandable_segments。


## Recovery2 `t-20260912004216-cmcxl` 结果：step0 完全闭合成功，step1 崩于 mxfp4 Marlin LoRA PTX（2026-09-12 09:05）

**step0 完全成功（OOM 修复配方有效！）**：
- step0 rollout 成功（01:15→02:54）→ 02:54:08 `optim_step_start mb=1 n_rl_datums=10169`
- **cispo backward 全量跑完无 OOM**：done=1→10169/10169（08:56:04，elapsed 21713s，~2.13s/datum）
- 08:56:04 `optim_step_done step=1 update_type=rl_only` ✓ —— **step0 完整闭合**
- **max_model_len 4096 + mb=1 + lora_shrink + gpu_mem 0.55 完全解决 step0 backward OOM**，10169 datums 全过。

**step1 vLLM 引擎重启失败（全新根因，非 OOM/碎片）**：
- 08:56:27 `cell=rl rollout step=1 policy=v1` → 08:56:33 `vLLM start tp=8 lora=adapters/rl`（step1 用训练后 LoRA，不同于 step0 lora=None）
- 08:57:42 `Using Marlin backend for mxfp4 lora`（vllm mxfp4.py:222）
- 08:57:45 8 卡 Worker_TP0-7 同时崩：`torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain` (cudaErrorUnsupportedPtxVersion)
- 失败栈：`mxfp4.py:339 process_weights_after_loading` → `marlin_utils_fp4.py:574 repack_weight: qweight = weight[i].view(int32).T.contiguous()` → PTX 不兼容
- **根因**：GPT-OSS-20B mxfp4 量化模型，vLLM 0.19.1 在加载 LoRA adapter 时走 mxfp4 Marlin LoRA backend，其 repack kernel 的 PTX 与 gm402 GPU CUDA driver 不兼容。step0 lora=None 走默认 path 不触发；step1+ lora=adapter 必触发 → 8 卡全崩。lora_shrink 补丁不解此（不同问题）。
- recovery2 failed（runTime=29850），保留不覆盖。

**Recovery3 待定**：需调研 vLLM 0.19.1 mxfp4 Marlin LoRA backend 禁用方法（强制 dequantize bf16？环境变量？`--quantization`？或 step1+ 改用 HF generate 绕过 vLLM LoRA）。已启动 research agent 调研。step0 的成功证明 OOM 修复配方正确，只需解决 step1+ 的 vLLM LoRA 加载问题。

## Recovery3 方案（用户批准：预生成 bf16 checkpoint）

research agent 确认 mxfp4+LoRA+gm402 是 vLLM 0.19.1 已知架构限制（select_mxfp4_moe_backend LoRA 分支硬编码走 Marlin，repack PTX 与 gm402 driver 不兼容），无可靠 runtime flag 禁用（VLLM_MXFP4_USE_MARLIN=0 需 triton_kernels 未装且 Triton path 也有 PTX 问题；VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 只对非 LoRA path；--quantization None 不生效）。--rollout-backend hf 可绕过但 sequential generate 太慢不可行。

**方案**：一次性 dequant mxfp4→bf16 checkpoint，--model_name 指向它，vLLM 走 bf16+LoRA 正常路径，避开 mxfp4 Marlin LoRA PTX 崩溃，rollout 仍快。step0 OOM 修复配方（mb=1+max_model_len 4096+lora_shrink+gpu_mem 0.55）保持有效（recovery2 已验证 step0 跑完 10169 datums 无 OOM）。

- **Dequant job** `t-20260912180631-zhptc`（1 GPU gm402）：脚本 `TRIM/scripts/dequant_gpt_oss_mxfp4_to_bf16.py`，HF transformers 4.57.6 from_pretrained(dtype=bf16) dequant（已确认 4.57.6 有 "we will default to dequantizing the model to bf16" 行为，recovery2 log 02:53:57），save_pretrained 到 `/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant`（~40GB bf16），强制 drop quantization_config。YAML: `cloudml_dequant_gpt_oss_20b_mxfp4_to_bf16_20260912_not_submitted.yaml`。
- **Recovery3 YAML** 已备好（待 dequant 成功后提交）：`cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_4step_20260911_recovery3_not_submitted.yaml`。--model_name → bf16-dequant dir，其余同 recovery2（mb=1, max_model_len 4096, lora_shrink, gpu_mem 0.55, rl 4-step, local_legacy）。8 GPU。
- 关键验证点：recovery3 step1 vLLM 加载 LoRA 时是否消除 mxfp4 Marlin PTX 崩溃（bf16 模型走正常 LoRA path）。step0 仍需跑（每 step 重新 rollout），但 step0 的 OOM 修复已验证。
- 失败任务保留：original t-20260911191937-y8uax、recovery1 t-20260911223014-skd0g、recovery2 t-20260912004216-cmcxl 均不覆盖。监控 cron 9ef06271（每10min）盯 dequant。

## Dequant 成功 + Recovery3 已提交（2026-09-12 18:17）

**Dequant job `t-20260912180631-zhptc` succeed**（runTime=333s, 1 GPU gm402）：
- preflight: torch 2.10.0+cu128, transformers 4.57.6 ✓
- 18:10:34 `MXFP4... we will default to dequantizing the model to bf16`（HF 4.57.6 确认 dequant）
- 18:11:45 加载完成 → 18:13:00 **`[dequant] saved config quantization_config: None`** + **`[dequant] OK: 9 safetensors, 41.8 GB`**
- 输出: `/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant`（9 safetensors, 41.8GB, config.json 无 quantization_config ✓——vLLM 不会检测 mxfp4）

**Recovery3 `t-20260912181707-wnnhv` 已提交**（8 GPU gm402, queue 2334, priority 8, tag MemQA-General-V5）：
- `--model_name /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant`（bf16, vLLM 走正常 LoRA path 避开 mxfp4 Marlin PTX）
- 保持 OOM 修复: `--train-micro-batch-size 1 --max-model-len 4096 --gpu-memory-utilization 0.55` + lora_shrink 补丁
- 全新 --out `...rl_4step_20260911_recovery3`，failed attempts（original/recovery1/recovery2/dequant-job succeed）保留不覆盖
- 高频每5min 监控（cron f405fc4f）。关键验证点: step1 vLLM+LoRA 是否消除 PTX 崩溃 + step0 OOM 仍解决。

## Recovery3 `t-20260912181707-wnnhv` 结果：bf16 解 step0 PTX 有效，但崩于 parse_generated_action wrapper（代码 bug）

**bf16 修复在 step0 vLLM 加载层面确认有效**：
- bf16-dequant 模型 vLLM 加载：`quantization=None` + `Using TRITON backend for Unquantized MoE`（非 mxfp4 Marlin）+ llm_ready。dequant job（t-20260912180631-zhptc, succeed, 41.8GB, config 无 quantization_config）产出正确。

**step0 rollout 崩于 parse_generated_action TypeError（代码 wrapper 不同步，非 mxfp4/OOM）**：
- `batched_env_rollout.py:128` 调 `parse_generated_action(text, ids, enc, harness_mask=, teacher_mode=, action_map=, finish_reason=)`
- 但 `batched_env_rollout.py:119` 导入 `four_cell_runtime.py:754` 的 wrapper，签名只 `harness_mask/teacher_mode`，**丢 action_map/finish_reason**
- wrapper 转发底层 `parse_rollout_action.parse_generated_action`（后者收 action_map）→ TypeError
- 根因：fae1557 引入 wrapper 只转 2 参数，caller 后加 action_map 未同步 wrapper
- recovery3 failed（runTime=1288），保留不覆盖。

## Recovery4 `t-20260912184849-kjuu8`（18:48 提交）：三重修复

修复 `four_cell_runtime.py:754` wrapper：签名加 `action_map=None, finish_reason=None` 并转发给 `_parse`（py_compile 过，已应用 JuiceFS 代码树）。

配置：同 recovery3（bf16-dequant + mb=1 + max_model_len 4096 + lora_shrink + gpu_mem 0.55, 8 GPU, rl 4-step, local_legacy）+ parse wrapper fix。全新 --out recovery4。failed attempts 保留不覆盖。高频每5min 监控（cron 5b46ca89）。

三重修复：①bf16 解 step1 PTX（recovery3 step0 验证 vLLM 加载不走 mxfp4 Marlin）②max_model_len 4096/mb=1 解 step0 OOM（recovery2 验证 10169 datums 无 OOM）③parse wrapper fix 解 step0 rollout TypeError。关键验证点：step0 rollout 不再 TypeError → step0 optim OOM 仍解决 → **step1 vLLM+LoRA 不再 PTX**。

## Recovery4 `t-20260912184849-kjuu8` 决定性突破：三重修复全部验证有效，step1 vLLM+LoRA 无 PTX 崩溃（2026-09-13 02:32）

**step0 完整闭合**：
- 02:08:33 `optim_step_done step=1 n_rl_datums=10034 update_type=rl_only`（cispo 10034 datums 全跑完无 OOM，elapsed 21384s ~5.9h）

**step1 vLLM+LoRA 成功启动（PTX 最终验证点通过！）**：
- 02:09:01 `cell=rl rollout step=1 policy=v1`
- 02:09:07 `vLLM start backend=vllm tp=8 lora=adapters/rl`（训练后 LoRA adapter）
- 02:10:50 `[rollout] chunk 1/1 episodes=256 prep=5.8s`（vLLM+LoRA 启动成功）
- 02:32 vLLM generate 持续推进（elapsed 90s/批，无报错）
- **对比 recovery2（mxfp4）**：同样 step1 vLLM+LoRA → `Marlin backend for mxfp4 lora` → 8 卡全崩 cudaErrorUnsupportedPtxVersion。**bf16 模型彻底消除 PTX 崩溃**。

**三重修复全部验证有效**：①parse wrapper fix（step0 rollout 无 TypeError）②bf16-dequant 模型（step1 vLLM+LoRA 无 mxfp4 Marlin PTX）③max_model_len 4096/mb=1（step0 backward 10034 datums 无 OOM）。

**结论**：GPT-OSS-20B mxfp4 在 vLLM 0.19.1+gm402 多步 RL 的两大 vLLM 兼容性死锁 + 一个代码 wrapper bug，通过 recovery4 三重修复完全解决。pre-train dequant mxfp4→bf16 checkpoint 是绕过 mxfp4+LoRA+vLLM PTX 限制的有效方案。

接下来：step1 optim（验证 step1+ 跨 step OOM/稳定性）→ step2-3。高频每5min 监控（cron 8a7763e9）。仍 running partial，未 4-step 全闭合前不作为 formal artifact。
