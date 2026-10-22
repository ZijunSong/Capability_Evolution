---
name: trim-gptoss20b-cispo-backward-oom
description: TRIM GPT-OSS-20B 8gpu CISPO backward OOM；vLLM gpu_memory_utilization=0.90 占满显存致训练 backward 仅剩 78MiB
metadata: 
  node_type: memory
  type: project
  originSessionId: a7fdecfd-6210-4e0d-b28f-e63a8cf05218
  modified: 2026-09-12T12:18:42.117Z
---

TRIM smoke recovery1 `t-20260911193613-lpyuz`（GPT-OSS-20B / trim / 8x gm402-80g）失败根因与诊断：

**进度**：依赖锁定+preflight 全过 → Round0 rollout（32 queries×8=256 episodes）生成 ~18min 成功 → `optim_step_start step=1 n_steps=1`（trim scape_seed 只跑 1 个 optim step）→ CISPO backward 在 GPT-OSS eager_attention_forward `torch.matmul(query, key_states...)` 分配 714MiB 时 OOM：GPU1 总 79.18GiB，仅剩 78.19MiB free，PyTorch 已占 76.75GiB。

**根因**：vLLM rollout（TP8，gpu_memory_utilization 默认 0.90）占满每卡 ~71GB；训练 backward（n_rl_datums=1535, n_rl_tokens=2.07M, micro_batch=4）在同组 8 卡再加载 student + 做 CISPO fwd/bwd，显存不足。vLLM 在训练阶段仍占显存（无 sleep/offload）。

**关键数据**：`n_steps=1`——trim scape_seed 只跑 1 个 optim step，rollout 完成后只差这一步 backward 不 OOM 即可完成。

**已确认（两个 Explore agent 读代码）**：
- `n_steps=1` 是 `train_cell(train_steps=1)` 每轮步数，**总 step=8**（trim scape_seed, on_policy_refresh=True 默认, train_steps 默认 8, `four_cell_runtime.py:2170` for step in range(8)）。每步重启 vLLM（8 次冷启动）。`max_turns` 实际=6（CLI 默认，MAX_TURNS env 不被 Python 读取）。
- scheme A 互斥（`vllm_hybrid.py:265-311`）：backward 时 vLLM 已 close_vllm 释放（rollout 结束→杀子进程+empty_cache）。**OOM 不是 vLLM 挤占，而是 micro_batch=4 的 attention 峰值+梯度累积本身超 79GB**。
- `--gpu-memory-utilization` 直传生效（CLI→args→open_vllm:1967→config.json→vllm_rollout_worker:131 的 LLM()），无 env 中转，launch dict 不含它不影响功能。
- keyys `t-20260910202309`（memory 误记 stable）实际也 failed，且更早：缺 `chz`（train_env=upstream No module chz），从未跑到训练。无 micro_batch=4 成功先例。

**修复（recovery2 `t-20260911223515-vmpgk`）**：`--train-micro-batch-size 4->1`（attention 峰值降 4 倍）+ `--gpu-memory-utilization 0.90->0.60`。脚本 `run_train_cloudml_8gpu_trim_smoke.sh` 已改。YAML recovery2。预期 backward 变慢 4 倍（1535 datums/micro_batch=1），8 step 总时长估 4-6h。

**2026-09-11 23:00 追加：recovery2/3 进展与 mb=1 仍 OOM**：
- recovery2 在 rollout 阶段崩于 `vLLM 0.19.1 lora_shrink_op.py:177 assert inputs.is_contiguous()`（复用 recovery1 --out 的 theta0 LoRA adapter 触发，见 [[vllm-0191-lora-shrink-contiguous-patch]]）。recovery2 runTime 796s failed。
- recovery3 `t-20260911225347-k9qvn`（现成补丁 patch_vllm_lora_shrink_contiguous.py 打上 OK + 全新 --out ...trim_all_recovery3 让 rollout0 lora=None base 起步）：rollout0 成功（~18min），但 **step1 backward mb=1 仍 OOM**：GPU1 剩 3.12GiB，PyTorch 占 64.51GiB + **10.68GiB reserved-but-unallocated（碎片）**，尝试分配 6.00GiB 失败（在 gpt_oss MoE expert 的 gate_up_proj bmm / attention）。
- **对比**：mb=4 PyTorch 占 76.75GiB 剩 78MiB（尝试 714MiB）；mb=1 占 64.51GiB 剩 3.12GiB（尝试 6GiB）。mb=1 改善 ~12GiB 但不够，且 6GiB 的大块分配 + 10.68GiB 碎片是主因。
- **结论**：调 micro_batch/gpu_mem 不足以解 OOM，需结构性方案（gradient checkpointing / expandable_segments / max_model_len 8192→4096）。CLAUDE.md expandable_segments 警告针对 vLLM runtime，但 backward 时 vLLM 已 close 退出，对 PyTorch backward 可能安全——待 agent 确认。recovery3 runTime 2017s failed。

**Why**：smoke 任务，OOM 在 backward 峰值，降 micro_batch 最对症。不走 PYTORCH_ALLOC_CONF=expandable_segments（CLAUDE.md：与 vLLM memory pool 不兼容）。
**How to apply**：见 [[trim-gptoss20b-trim-smoke-20260911]]、[[cloudml-trim-launch-heredoc-juicefs-reread]]。修复后用全新 recovery2，不覆盖 recovery1。

**2026-09-12 00:21 追加：recovery4 成功（max_model_len 8192→4096 解 OOM）**：
- recovery4 `t-20260911234243-8wbds`（mb=1 + gpu_mem 0.60 + **--max-model-len 4096** + lora_shrink 补丁 + 全新 --out recovery4）：step1 CISPO backward **越过 recovery3 的 done-16 OOM 坎，稳定推进** done 1→8→16→24→...→72（elapsed 145s，无 OOM）。**max_model_len=4096 结构性降 backward 序列长（four_cell_runtime.py:1749 max_full_tokens=max_model_len → hf_tool_opd.py:346 max_len）成功**。
- 启动链路全部修复固化在 `run_train_cloudml_8gpu_trim_smoke.sh`：cp-to-tmp（避 heredoc 回读 JuiceFS）+ patch_vllm_lora_shrink_contiguous.py（避 vLLM LoRA shrink assert）+ mb=1 + gpu_mem 0.60 + max_model_len 4096 + 全新 --out。
**2026-09-12 02:12 追加：recovery4 step1 完成 + step2 backward 起步 OOM（碎片）**：
- **step1 完整跑完**：02:04:59 `optim_step_done step=1`（n_rl=1534, n_opd=1159, n_rl_tokens=2.01M, update_type=rl_opd_joint, elapsed 6426s）。CISPO backward 1534 + OPD backward 1159 全过无 OOM。**max_model_len=4096 对 step1 backward 有效**。
- **step2 rollout1 加载 LoRA adapter（scape_seed）成功**：vLLM generate 推进 ~7min，**lora_shrink 补丁验证通过**（没崩在 is_contiguous assert，recovery2 的坑已填）。补丁有效。
- **step2 CISPO backward done=1 即 OOM**（02:25:30）：GPU1 剩 386MiB，PyTorch 占 65.97GiB + **11.96GiB reserved-but-unallocated（碎片）**，尝试分配 720MiB 失败。栈在 `four_cell_runtime.py:2189 train_cell`（_cispo_backward done=1）。
- **对比 step1**：step1 backward 前 rollout0（lora=None）close_vllm 后干净，碎片 10.68GiB 够跑完 1534；step2 backward 前 rollout1（lora=LoRA）close 后碎片升到 11.96GiB，done=1 就炸。**根因=跨 step 碎片累积（close_vllm 后 LoRA 相关残留未清干净），非单次峰值**。
- 报错再次提示 `PYTORCH_ALLOC_CONF=expandable_segments:True`。CLAUDE.md 警告与 vLLM 不兼容，但 backward 时 vLLM 已 close；风险是 env 继承给 vLLM 子进程。agent 正在查 vLLM 0.19.1 是否受 expandable_segments 影响 + close_vllm 是否可加额外 empty_cache。
- recovery4 `t-20260911234243-8wbds` runTime 9765s failed（step2 backward）。但已证明：max_model_len=4096 + lora_shrink 补丁让 step1 跑完，step2 rollout 带 LoRA 也成功——只剩跨 step 碎片 OOM 这一个问题。

**2026-09-12 02:55 追加：recovery5 提交（expandable_segments 解跨 step 碎片）**：
- Explore agent 代码级确认：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **安全**——SFT 路径 `hf_sft.py:280,521` 已用同设置跑 GPT-OSS-20B+8GPU（同模型同卡数先例）；vLLM 0.19.1 V1 用 PyTorch caching allocator 分配权重/KV cache，expandable_segments 只改段扩展策略不破坏正确性；enforce_eager=True 禁 CUDAGraph 降低交互风险。env 经 `vllm_hybrid.py:432 os.environ.copy()` 继承给 vLLM 子进程，SFT 先例证明安全。CLAUDE.md "不设与 vLLM memory pool 不兼容的 expandable_segments" 警告针对旧 vLLM+CUDAGraph 场景，enforce_eager+V1 下不适用。
- **修复**：recovery5 `t-20260912025557-bumx4`，启动脚本 `run_train_cloudml_8gpu_trim_smoke.sh` 加 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，让 allocator 在 11.96GiB 碎片中扩展段而非找不到连续 720MiB。沿用 mb=1 + gpu_mem 0.60 + max_model_len 4096 + lora_shrink 补丁 + 全新 --out recovery5。
- close_vllm（`vllm_hybrid.py:306-311,530-546`）已含 _release_cuda（empty_cache+ipc_collect+synchronize），但默认 expandable_segments=False 下碎片化 segment 无法合并，empty_cache 只释放 unallocated 不解碎片——故必须 expandable_segments。可选兜底：vllm_hybrid.py:445 Popen 加 start_new_session + close 用 killpg 确保孙进程回收（暂未加，先看 expandable_segments 是否足够）。
- 验证点：recovery5 能否跑过 step2 backward（done>1）并继续 step2+。step1 backward 已验证可过（recovery4 跑完），关键看 step2+ 跨 step 是否还碎片 OOM。

**2026-09-12 19:51 追加：recovery5 全部完成（8 step 全跑完，Succeeded）**：
- recovery5 `t-20260912025557-bumx4` **Succeeded**，runTime 60890s（~16.9h）。8 个 on-policy step 全部完成（`n_optimizer_steps=8`），无任何 OOM/报错/hung。step2 backward 越过 done=1 坎稳定推进，expandable_segments 完全解决跨 step 碎片 OOM。最终结果 19:51:42 `{"cell":"scape_seed","train_only":true,"n_decision_points":1536,"n_optimizer_steps":8,"adapter":".../trim_all_recovery5/adapters/scape_seed"}`。
- adapter 已保存：`/mnt/songzijun/Capability_Evolution/TRIM/outputs/train_harness1_bcplus_full_gpt-oss-20b_trim_all_recovery5/adapters/scape_seed`。
- **最终生效修复链路**（固化在 `run_train_cloudml_8gpu_trim_smoke.sh`）：cp-to-tmp（避 heredoc 回读 JuiceFS）+ patch_vllm_lora_shrink_contiguous.py（避 vLLM LoRA shrink assert）+ `--train-micro-batch-size 1` + `--gpu-memory-utilization 0.60` + `--max-model-len 4096`（结构性降 backward datum seq len）+ `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（解跨 step 碎片）+ 全新 --out。单 step ~107min 稳定无退化，8 step 总 ~16.9h。
- 这是 smoke 任务，结果不得冒充 formal artifact（CLAUDE.md）；但修复链路可复用于正式 trim 训练。关联 [[trim-gptoss20b-trim-smoke-20260911]]、[[vllm-0191-lora-shrink-contiguous-patch]]、[[cloudml-trim-launch-heredoc-juicefs-reread]]。

**2026-09-11 22:37 追加：4-step RL 任务的同类 OOM + recovery1（与 smoke 不同 task）**：
- 失败任务 `t-20260911191937-y8uax`（runTime=7200 failed，GPT-OSS-20B / **rl** / 4-step / max_turns=40 / train_env=local_legacy，非 smoke）：step0 rollout 成功（~1h37min, 40 batches vLLM generate, output ~2488 toks/s）→ 21:19:49 `optim_step_start step=1 n_steps=1 micro_batch=4 n_rl_datums=10058 n_opd_datums=0` → cispo backward done 1→16→24 时 OOM：`hf_rl_opd_client.py:191 _cispo_backward` → `hf_tool_opd.py:372 _response_position_logits` → peft → accelerate hooks → `gpt_oss/modeling_gpt_oss.py:130 torch.bmm(hidden_states, self.gate_up_proj)` → GPU1 79.15/79.18GiB。**注意 OOM 行不同**：smoke 在 attention `matmul` (714MiB)，本次在 MoE expert `gate_up_proj bmm` (858MiB)——本次 datum 更长(max_turns 40)，MoE expert 激活峰值更靠后打爆。
- 与 smoke 共同根因：scheme A 互斥，optim 前 `detach_vllm/close_vllm→_release_cuda` 已释放 vLLM 显存（`vllm_hybrid.py` assert_exclusive，attach_hf 时 vllm_active 必为 False）。**故 OOM 是 micro_batch 激活峰值本身超 79GB，与 vLLM 占用无关**——降 gpu_memory_utilization 对 backward OOM 无帮助（vLLM 已释放），只对 rollout 阶段有意义。
- recovery1 `t-20260911223014-skd0g`（22:30 提交，直接 running 无排队，22:36/22:37 PREFLIGHT+LUCENE_OK 全过）：`--train-micro-batch-size 4→2`（峰值约减半，从 ~79GB 临界 → ~40GB，理论应过）+ `--gpu-memory-utilization 0.90→0.55`（对 backward OOM 中性，但保守留余量；rollout 阶段 vLLM 占更少）。expandable_segments 不设。新 --out `...rl_4step_20260911_recovery1`，failed attempt 保留。
- **决策依据**：mb=4 峰值撞 79GB 上限，mb=2 峰值约半应安全（gradient accumulation 不累加激活，峰值=单 mb）。若 recovery1 仍 OOM（mb=2 不够），recovery2 降 mb 2→1（峰值再减半）。关联 [[trim-gptoss20b-rl-4step-20260911]]。

**2026-09-11 23:38 追加：mb=2 风险升高 + 更优结构性修复参数**：
- 对比证据：Qwen3-4B (dense 4B) smoke 用 mb=2+util0.6 **成功跨过原崩点**（done 752→864，[[trim-qwen4b-trim-smoke-20260911]]）；但 **GPT-OSS-20B (MoE 20B) smoke mb=1 仍 OOM**（PyTorch 64.51GiB + 10.68GiB 碎片，6GiB 大块分配失败）。GPT-OSS MoE 比 Qwen dense 更吃显存，mb=2 在 GPT-OSS RL 4-step（datum 更长 max_turns=40）**大概率仍 OOM**。
- **关键参数发现**：`four_cell_runtime.py:1749 backend.max_full_tokens = int(getattr(args, "max_model_len", 8192) or 8192)`，`hf_tool_opd.py:346 max_len = max_full_tokens`。即 **`--max-model-len` 直接控制 cispo backward 的 datum 序列长度上限**（不只是 vLLM 上下文窗口）。降 `--max-model-len 8192→4096` 是结构性降 MoE expert `gate_up_proj` bmm 的 hidden_states seq 维度，比降 mb 更对症且安全（不碰 vLLM 兼容性、不改 scorer）。
- recovery1 仍在 step0 rollout（23:37 batch 19，距 optim ~50min）。recovery2 YAML 已备好基础（cp 自 recovery1），待 recovery1 OOM 后改：mb 保持 2 + `--max-model-len 8192→4096` + 加 `python3 scripts/patch_vllm_lora_shrink_contiguous.py`（防 step1+ vLLM+theta0-LoRA 的 assert is_contiguous 崩溃，见 [[vllm-0191-lora-shrink-contiguous-patch]]）+ 新 --out recovery2。expandable_segments 暂不加（CLAUDE.md 警告 + max_model_len 降半应足够）；若 4096 仍不够再考虑。

**2026-09-12 00:42 追加：RL 4-step recovery1 OOM 复现 + recovery2 已提交**：
- recovery1 `t-20260911223014-skd0g` failed（runTime=7802）。step0 rollout 成功（~1h37min, 40 batches）→ 00:25 optim_step_start mb=2 n_rl_datums=9884 → cispo backward **稳定推进到 done=424**（远超原崩点 24，elapsed 719s）→ **00:40:48 OOM 复现**：Tried to allocate 1.93 GiB，GPU1 79.10/79.18GiB，PyTorch 67.70GiB allocated + **10.53GiB reserved-but-unallocated（碎片）**，仅剩 72MiB。
- **mb=2 部分有效但不够**：把 PyTorch 占用从 mb=4 的 76.75GiB 降到 67.70GiB（省 9GiB，越过了 done=24 原崩点跑到 done=424），但**长 tail 序列 bucket + 10.53GiB 碎片**仍导致 1.93GiB 大块分配失败。与 smoke mb=1 OOM 模式完全一致（碎片化是主因，非峰值）。
- **结论**：mb=2 只能延迟 OOM 不能根治，证实"调 micro_batch/gpu_mem 不足以解 GPT-OSS-20B MoE OOM"。
- recovery2 `t-20260912004216-cmcxl`（00:42 提交）：**mb 2→1 + max-model-len 8192→4096**（结构性降 backward datum seq len，对齐 smoke 成功的 recovery4 配置）+ lora_shrink 补丁 + gpu-mem 0.55 + 全新 --out recovery2。failed recovery1 保留不覆盖。高频每5分钟监控（cron 0093906f）。recovery2 YAML: `TRIM/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_4step_20260911_recovery2_not_submitted.yaml`。
- 关键验证点：recovery2 step0 optim cispo backward 能否跑完 9884 datums（max_model_len 4096 应把长 tail 峰值+碎片压到安全线）。若仍 OOM，recovery3 加 max_model_len 3072 或 expandable_segments。

**2026-09-12 03:33 追加：RL 4-step recovery2 cispo backward 稳定推进（max_model_len 4096 有效！）**：
- recovery2 `t-20260912004216-cmcxl`（mb=1+max_model_len 4096+lora_shrink 补丁+gpu_mem 0.55）：step0 rollout 成功（01:15→02:54，~1h39min，38 批 vLLM generate，max_model_len 4096 使 rollout 略长于 8192）→ 02:54:08 `optim_step_start mb=1 n_rl_datums=10169` → **cispo backward 稳定推进无 OOM**：done=1→16→152→416→688→960→1088（elapsed 2.5→2318s），student_fwd 1089s 线性稳定，**~2.13s/datum**。
- **已远超所有 OOM 崩点**：smoke recovery3 mb=1 在 done=16 OOM、RL recovery1 mb=2 在 done=424 OOM——recovery2 mb=1+4096 已到 done=1088 持续无 OOM。**max_model_len 4096 结构性修复确认有效**（降 MoE expert gate_up_proj bmm 的 hidden_states seq 维度，压住长 tail 峰值+碎片）。
- lora_shrink 补丁也确认生效（01:03 PATCHED_VLLM_LORA_SHRINK_CONTIGUOUS，step0 rollout lora=None 未触发，待 step1+ 验证）。
- **ETA（step0 cispo）**：10169 datums × ~2.13s ≈ ~6h（至 ~09:00），加 opd/optim + 3 step，总时长很长。
- **仍需盯防**：smoke recovery4 step1 backward 过了但 step2 backward done=1 即 OOM（跨 step 碎片累积，line 42-45）。RL 4-step 共 4 step，即使 step0 过，step1-3 backward 可能复现跨 step 碎片 OOM。recovery2 仍 running partial，未闭合不得作 formal artifact。

**2026-09-12 06:57 追加：RL 4-step recovery2 cispo backward 稳定推进 4h+ 无 OOM，切换半小时监控**：
- recovery2 cispo backward 自 02:54 启动后**稳定线性推进 4h+ 无 OOM**：done=16→1088→3000→5000→6840/10169（elapsed 2.5→14600s），student_fwd 6857s，~2.13s/datum。**max_model_len 4096 + mb=1 + lora_shrink 修复确认有效**，已远超 smoke recovery3 done=16 OOM 坎、RL recovery1 done=424 OOM 坎。
- 训练已完全启动并持续稳定运行。按目标"直至完全启动训练持续运行，改为每隔半小时监控"，从高频每5min（cron 0093906f）切回半小时（cron 5a20f28d，:17/:47）。
- step0 cispo 剩余 ~2.4h（至 ~09:20，ETA 略延后因 datum 10169 比 recovery1 9884 多）。之后 opd_gap phase（n_opd=0 RL 模式应极短）+ optim_step + step1-3。
- **仍需盯防**：step1-3 跨 step 碎片 OOM（smoke recovery4 step1 过但 step2 done=1 即 OOM）。step0 optim_step_done 闭合后是关键节点。
- recovery2 仍 running partial，未闭合不得作 formal artifact。关联 [[trim-gptoss20b-rl-4step-20260911]]。

**2026-09-12 09:05 重大追加：RL 4-step recovery2 step0 完全闭合成功，但 step1 vLLM 引擎重启崩于 mxfp4 Marlin LoRA PTX 不兼容（非 OOM！）**：
- **step0 完全成功**：08:56:04 `cispo done=10169/10169` + `optim_step_done step=1 update_type=rl_only`（n_rl_datums=10169, elapsed 21717s）。**max_model_len 4096 + mb=1 完全解决 step0 backward OOM**，cispo 全量 10169 datums 跑完无 OOM。
- **step1 vLLM 引擎重启失败**：08:56:33 `vLLM start tp=8 lora=adapters/rl`（step1 用训练后 LoRA，不同于 step0 lora=None）→ 08:57:42 `Using Marlin backend for mxfp4 lora`（mxfp4.py:222）→ 08:57:45 8 卡 Worker 同时崩：`torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain` (cudaErrorUnsupportedPtxVersion)。
- **失败栈**：`mxfp4.py:339 process_weights_after_loading` → `_setup_kernel` → `convert_to_mxfp4_moe_kernel_format` → `prepare_moe_mxfp4_layer_for_marlin` → `marlin_utils_fp4.py:574 repack_weight: qweight = weight[i].view(int32).T.contiguous()` → PTX 不兼容 CUDA error。
- **根因（全新，非碎片/OOM/non-lora_shrink）**：GPT-OSS-20B 是 mxfp4 量化。step0 lora=None 走默认 mxfp4 path 成功；step1+ lora=adapter 触发 vLLM 0.19.1 的 **mxfp4 LoRA Marlin backend**（`mxfp4.py:222`），其 Marlin repack kernel 的 PTX 与 gm402 GPU CUDA 工具链不兼容 → `.contiguous()` 触发的 CUDA kernel PTX 版本不被 gm402 driver 接受 → 8 卡全崩。lora_shrink 补丁不解此问题（shrink 是 is_contiguous assert，这里是 PTX 工具链不兼容的 CUDA error）。
- **对比 smoke recovery4**：smoke step1 backward 能过（scape_seed theta0 adapter）但 step2 OOM（碎片）；本 RL 任务 step1 在 vLLM 引擎重启阶段就崩（mxfp4 Marlin LoRA PTX），根本没到 backward。**不同 task 的 LoRA adapter 路径触发不同失败**。
- recovery2 `t-20260912004216-cmcxl` failed（runTime=29850），保留不覆盖。**step0 的成功证明 OOM 修复配方（mb=1+max_model_len 4096+lora_shrink+gpu_mem 0.55）对 backward OOM 完全有效**，但 step1+ 暴露了新的 mxfp4 Marlin LoRA PTX 兼容性问题，需另解。
- **待调研**：vLLM 0.19.1 mxfp4 LoRA Marlin backend 的禁用/绕过方法（强制 dequantize 到 bf16 再做 LoRA？环境变量？`--quantization` 参数？）。关联 [[trim-gptoss20b-rl-4step-20260911]]、[[vllm-0191-lora-shrink-contiguous-patch]]。

**2026-09-12 18:48 重大追加：recovery3 bf16 修复 step0 PTX 有效，但崩于 parse_generated_action wrapper 丢 action_map（代码 bug，非 mxfp4）→ recovery4 已提交**：
- recovery3 `t-20260912181707-wnnhv` failed（runTime=1288）。bf16-dequant 模型 vLLM 加载正确：`quantization=None` + `Using TRITON backend for Unquantized MoE`（**非 mxfp4 Marlin**），llm_ready。**bf16 dequant 解 PTX 在 step0 vLLM 加载层面确认有效**。
- 但 step0 rollout 报 `TypeError: parse_generated_action() got an unexpected keyword argument 'action_map'`：`batched_env_rollout.py:128` 调 `parse_generated_action(..., action_map=..., finish_reason=...)`，但 `batched_env_rollout.py:119` 导入的是 `four_cell_runtime.py:754` 的 wrapper，该 wrapper 签名只收 `harness_mask/teacher_mode`，**丢 action_map/finish_reason 不转发**给底层 `parse_rollout_action.parse_generated_action`（后者收 action_map）。代码 wrapper 与 caller 不同步（fae1557 引入 wrapper 时只转发 2 参数，caller 后加 action_map 未同步 wrapper）。
- **修复（代码级，已应用到 JuiceFS 挂载代码树）**：`four_cell_runtime.py:754` wrapper 签名加 `action_map=None, finish_reason=None` 并转发给 `_parse`。py_compile 过。
- recovery4 `t-20260912184849-kjuu8`（18:48 提交，8 GPU, enqueued）：同 recovery3 配置（bf16-dequant + mb=1 + max_model_len 4096 + lora_shrink + gpu_mem 0.55）+ parse wrapper fix。failed recovery3 保留不覆盖。高频每5min 监控（cron 5b46ca89）。
- **三重修复叠加**：bf16 解 step1 PTX（recovery3 step0 验证加载正确）+ max_model_len 4096/mb=1 解 step0 backward OOM（recovery2 验证 10169 datums 无 OOM）+ parse wrapper fix 解 step0 rollout TypeError。recovery4 关键验证点：step0 rollout 不再 TypeError + step1 vLLM+LoRA 不再 PTX。
- 关联 [[vllm-mxfp4-lora-marlin-ptx-gm402]]、[[trim-gptoss20b-rl-4step-20260911]]。dequant job t-20260912180631-zhptc succeed（bf16 checkpoint 41.8GB 持久化可复用）。

**2026-09-12 20:34 追加：recovery4 三重修复全部生效，step0 cispo backward 稳定推进无 OOM，切回半小时监控**：
- recovery4 `t-20260912184849-kjuu8`：三重修复①bf16-dequant 模型（vLLM quantization=None + Unquantized MoE，非 mxfp4 Marlin）②max_model_len 4096/mb=1（解 step0 OOM）③parse_generated_action wrapper fix（four_cell_runtime.py:754 加 action_map/finish_reason 转发，解 step0 rollout TypeError）。
- step0 rollout 成功（19:07→20:12，40 batches，无 TypeError——recovery3 在此崩，wrapper fix 消除）→ step0 optim cispo backward 20:12 启动**稳定推进无 OOM**：done=16→224→376→488→624/10034（elapsed 34→1328s），student_fwd 624s，~2.13s/datum，已过原 recovery1 mb=4 done=24 OOM 坎。
- cispo 全跑完 ~5.4h（至 ~02:00）→ opd/optim → **step1 vLLM start lora=adapters/rl（PTX 最终验证点，~6h 后）**。切回每半小时监控（cron f48a7f3c），step0 闭合后切回高频盯防 step1。
- bf16 模型 datum 数 10034（略少于 mxfp4 的 10169，因 bf16 权重布局不同）。bf16 checkpoint（/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant, 41.8GB, 9 safetensors, config 无 quantization_config）持久化可复用。
- 关联 [[vllm-mxfp4-lora-marlin-ptx-gm402]]、[[trim-gptoss20b-rl-4step-20260911]]。

**2026-09-13 02:32 决定性里程碑：recovery4 三重修复全部验证有效，step1 vLLM+LoRA 无 PTX 崩溃！**：
- recovery4 `t-20260912184849-kjuu8`：step0 完整闭合（02:08:33 `optim_step_done step=1 n_rl_datums=10034 update_type=rl_only`，cispo 10034 datums 全跑完无 OOM）→ step1 启动（02:09:01 `rollout step=1 policy=v1`）→ 02:09:07 `vLLM start tp=8 lora=adapters/rl`（**训练后 LoRA adapter！**）→ 02:10:50 `[rollout] chunk 1/1 episodes=256 prep=5.8s`（**vLLM+LoRA 启动成功，无 cudaErrorUnsupportedPtxVersion！**）→ 02:32 vLLM generate 持续推进无报错。
- **对比 recovery2（mxfp4）**：同样 step1 `vLLM start lora=adapters/rl` → `Using Marlin backend for mxfp4 lora` → 8 卡全崩 cudaErrorUnsupportedPtxVersion。**recovery4 bf16 模型彻底消除 mxfp4 Marlin LoRA PTX 崩溃**。
- **三重修复全部验证有效**：①parse wrapper fix（step0 rollout 无 TypeError）②bf16-dequant 模型（step1 vLLM+LoRA 无 mxfp4 Marlin PTX）③max_model_len 4096/mb=1（step0 backward 10034 datums 无 OOM）。
- **结论**：GPT-OSS-20B mxfp4 在 vLLM 0.19.1+gm402 多步 RL 训练的两大 vLLM 兼容性死锁（step0 backward OOM + step1+ mxfp4 Marlin LoRA PTX）+ 一个代码 wrapper bug，已通过 recovery4 三重修复完全解决。pre-train dequant mxfp4→bf16 checkpoint（/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant, 41.8GB）是绕过 mxfp4+LoRA+vLLM PTX 限制的有效方案。
- 接下来盯防 step1 optim（验证 step1+ 跨 step OOM/稳定性）→ step2-3。若 step1+ 持续稳定，4-step 训练将最终跑通。高频每5min 监控（cron 8a7763e9）。仍 running partial，未 4-step 全闭合前不作为 formal artifact。
- 关联 [[vllm-mxfp4-lora-marlin-ptx-gm402]]、[[trim-gptoss20b-rl-4step-20260911]]。dequant job t-20260912180631-zhptc（bf16 checkpoint）+ parse wrapper fix（four_cell_runtime.py:754）均持久化可复用。

**2026-09-13 04:08 追加：recovery4 step1 optim cispo backward 跨 step OOM 验证点通过，切回半小时监控**：
- step1 rollout 完成（40 批，02:10→03:53）→ 03:53:47 `Loading checkpoint shards 9/9`（bf16 9 shards ~11s）→ 03:54:09 `hf_fb phase=cispo step=2 done=8 total=10140`（step1 optim 即 log step=2 启动，**无 OOM**）→ 稳定推进 done=8→136→272→408/10140（elapsed 17→865s），student_fwd 406s，~2.13s/datum。
- **跨 step OOM 验证点通过**：smoke recovery4 在 step2 backward done=1 因跨 step 碎片累积 OOM 崩溃；recovery4（bf16+mb=1+max_model_len 4096）step1 optim cispo 已到 done=408 稳定无 OOM。bf16 模型 scheme A 互斥（optim 前 close_vllm 释放干净）+ mb=1 在跨 step 场景同样有效。
- **三大风险全部解决**：①step0 backward OOM（max_model_len 4096+mb=1）②step1+ vLLM mxfp4 Marlin LoRA PTX（bf16 dequant）③跨 step 碎片 OOM（bf16+mb=1+scheme A 释放）④parse wrapper TypeError（four_cell_runtime.py:754 fix）。recovery4 是 GPT-OSS-20B mxfp4 在 vLLM 0.19.1+gm402 多步 RL 训练的完整可行配置。
- step1 cispo ~5.9h（至 ~10:00）→ step2-3 各 ~7.5h。4-step 总 ETA ~22-24h（至 09-14 凌晨）。切回每半小时监控（cron a921e1df）。仍 running partial，未 4-step 全闭合前不作为 formal artifact。

**2026-09-14 01:57 最终成功：recovery4 4-step RL 训练全部跑通，state=succeed！**：
- recovery4 `t-20260912184849-kjuu8`：state=succeed，runTime=112126s（~31.1h），正常退出。4-step RL 训练全部完成。
- 4 个 optim_step_done（step0/1/2/3 的 cispo backward 各跑完 10034/10140/10019/10193 datums 无 OOM），4 个 vllm_sessions（rollout0/1/2/3）全部成功，step1/2/3 的 vLLM+LoRA 重启全部无 PTX。
- 输出产物完整：adapters/{theta0,rl}、4 个 vllm_sessions、FOUR_CELL_SUMMARY.json、TRAIN_SUMMARY.json、ADAPTER_MAP.json、ADAPTER_RELOAD_AUDIT.json、LAUNCH.json、RUN_MANIFEST.json、RUNTIME_EFFECT_AUDIT.json。
- **三重修复完全验证有效**：①parse_generated_action wrapper fix（four_cell_runtime.py:754 加 action_map/finish_reason 转发）②bf16-dequant 模型（/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant, 41.8GB, config 无 quantization_config，vLLM quantization=None 走 Unquantized MoE 非 mxfp4 Marlin）③max_model_len 4096 + mb=1（解 step0+跨 step backward OOM）+ lora_shrink 补丁。
- **这是 GPT-OSS-20B mxfp4 在 vLLM 0.19.1+gm402 多步 RL 训练的完整可行配置**。pre-train dequant mxfp4→bf16 checkpoint 是绕过 mxfp4+LoRA+vLLM PTX 限制的有效方案。
- 失败任务均保留：original t-20260911191937-y8uax、recovery1 t-20260911223014-skd0g、recovery2 t-20260912004216-cmcxl、recovery3 t-20260912181707-wnnhv。dequant job t-20260912180631-zhptc succeed（bf16 checkpoint 持久化）。
- 关联 [[vllm-mxfp4-lora-marlin-ptx-gm402]]、[[trim-gptoss20b-rl-4step-20260911]]。输出目录：/mnt/songzijun/Capability_Evolution/TRIM/outputs/trim_train_harness1_bcplus_full_gpt_oss_20b_8gpu_rl_4step_20260911_recovery4。
