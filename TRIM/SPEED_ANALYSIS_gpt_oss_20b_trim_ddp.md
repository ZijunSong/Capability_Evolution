# GPT-OSS-20B mxfp4 trim DDP 100step 训练速度分析

> 任务：`t-20260919095305-qxmz5`（recov2）
> 配置：GPT-OSS-20B mxfp4 原版 + torch_ddp_lora + train_method trim + 8×gm402-80g
> 分析时间：2026-09-24（已完成 36/100 step，runTime 121h）

## 结论先行

**单 step 实测 ~3.3h，100 step 预计 ~330h（13.7 天）完成。** 比同框架 Qwen3-4B trim DDP（44min/step）慢约 **4.5 倍**。速度慢是 **20B mxfp4 的固有代价 + mb1 致 microbatch 数膨胀 + max-turns 40 深轨迹 + rollout/backward 串行** 四因叠加的结果，**非配置问题、非卡顿、非资源未充分利用**——任务全程零报错零 OOM，8 卡显存峰值 74.8GB/80GB（93.5%）。

## 实测节奏（稳态，无退化）

| step | STEP_COMPLETE 时间 | 本 step 耗时 |
|---|---|---|
| 1 | 09-19 13:22 | ~3.3h（含启动开销） |
| 7 | 09-20 09:39 | 3h20m（step6→7） |
| 10 | 09-20 19:54 | 3h22m |
| 14 | 09-21 09:45 | 3h24m |
| 18 | 09-21 23:30 | 3h25m |
| 22 | 09-22 13:00 | 3h29m |
| 28 | 09-23 09:05 | 3h19m |
| 32 | 09-23 22:35 | 3h22m |
| 35 | 09-24 08:41 | 3h17m |

- 稳态 ~3.3h/step，从 step1 到 step22 **节奏无加速退化**，极其稳定。
- backward `update_wall_s` 稳定在 7863~8045s（~2.19h），rollout 约 ~1.1h/step。

## 速度慢的四大根因（按影响排序）

### 根因 1：mb1 致 backward microbatch 数膨胀（主因，~2.2h/step）

- `--train-micro-batch-size 1` 是为规避 20B mxfp4 backward OOM 的必要约束（[[trim-gptoss20b-cispo-backward-oom]] 实证 mb≥2 必 OOM）。
- 每 step 的 OPD datum 数 ≈ 256 episodes × ~5 action points ≈ 1275，mb=1 → **每 step 1273~1280 个 microbatch 必须串行 backward**（DDP 路径无跨卡切分，见 [[capability-evolution-sync-20260918]] 多节点探索结论）。
- 每 microbatch = student_fwd + teacher_fwd + loss.backward()，实测 ~6.3s/microbatch → 1275 × 6.3s ≈ **8025s ≈ 2.23h/step 纯 backward**。
- 对比 Qwen3-4B：同样 mb=1 但 action point 更少（303 microbatch），backward 仅 30.8min。20B 的 1275 microbatch 是 4B 的 4.2 倍，这是 backward 慢 4 倍的直接来源。

### 根因 2：20B mxfp4 rollout 生成慢（次因，~1.1h/step）

- 每 step rollout = 256 episodes × 最多 40 轮 agentic 工具调用（max-turns 40），每轮提交 ~244 prompts 给 vLLM。
- 20B mxfp4 用 TRITON MoE backend（非 Marlin，worker.log `mxfp4.py:352 Using 'TRITON' Mxfp4 MoE backend`），生成吞吐 ~1900 toks/s（8 卡 per-rank TP=1 并行）。
- 相比 Qwen3-4B（4B bf16，生成快），20B mxfp4 单 token 生成开销高 ~5 倍，深轨迹（40 turns）放大该差距 → rollout ~1.1h/step（4B 约 15min）。
- mxfp4 量化的 TRITON MoE kernel 本身也有额外反量化开销。

### 根因 3：max-turns 40 深轨迹（放大根因 1+2）

- `--max-turns 40` 让每个 episode 最多走 40 轮工具调用，轨迹深度大 → 单 episode 的 action points 多 → OPD datum 多 → microbatch 多（根因 1 放大）+ rollout 轮次多（根因 2 放大）。
- 降 max-turns 可直接降 microbatch 数与 rollout 时长，但会牺牲训练轨迹的探索深度（影响 trim 训练效果，不宜轻易降）。

### 根因 4：rollout/backward 严格串行（Scheme A 时分复用）

- DDP 路径用 Scheme A：vLLM-rollout 与 HF-train 交替占用同一 8 卡 GPU set（`four_cell_runtime` 单进程 asyncio 循环）。
- step N 的 rollout 完成后才开始 step N 的 backward，backward 完成存 adapter 重启 vLLM 才进 step N+1。
- 无 Scheme B（vLLM 8 卡 + HF 8 卡并行流水线，step N backward 与 step N+1 rollout 重叠），框架当前不支持（[[capability-evolution-sync-20260918]] 多节点探索结论：Scheme B 需改代码开发）。
- 故 rollout 1.1h + backward 2.2h 必须串行相加 = 3.3h/step，无法通过并行隐藏任一段。

## 已验证排除的非根因

| 怀疑点 | 实测 | 结论 |
|---|---|---|
| 显存未充分利用 | peak_mem 74774MB/80GB（93.5%），KV cache 40.31GiB/卡 | ✅ 已充分利用，gpu0.70 高水位 |
| GPU 闲置/卡死 | runTime 持续涨，actor_heartbeat 每 25s+8 microbatch | ✅ 非 hung，持续计算 |
| OOM 拖慢 | 全程 0 OOM，peak 留 5.2GB 余量 | ✅ 无 OOM |
| JuiceFS I/O 拖慢 | 模型已 cp 到 /tmp 本地盘，rollout/backward 不读 JuiceFS | ✅ 已规避（recov1 修复） |
| DDP 通信开销 | backward 是串行梯度累积，8 rank all-reduce 占比小 | ✅ 非 DDP 通信瓶颈 |
| PTX 崩溃重试 | rollout 用 TRITON MoE backend，全程 0 PTX 错误 | ✅ 无 PTX 问题 |

## 量化对比

| 项 | Qwen3-4B trim DDP（muhdh） | GPT-OSS-20B trim DDP（本任务） | 倍数 |
|---|---|---|---|
| 模型大小 | 4B bf16 | 20B mxfp4 | 5× |
| 单 step 耗时 | 44min | 3.3h（198min） | 4.5× |
| backward microbatch | 303 | 1273 | 4.2× |
| backward 耗时 | 30.8min | 132min（2.2h） | 4.3× |
| rollout 耗时 | 15min | 66min（1.1h） | 4.4× |
| backward 峰值显存 | 18.9GB/卡 | 74.8GB/卡 | 4.0× |

各维度慢 4~4.5 倍，与 20B vs 4B 的模型规模差（5×）基本匹配，**符合预期，无异常**。

## 可加速方向（需权衡，不建议轻易改）

1. **提高 mb**（如 mb=2/4）：backward microbatch 数减半/四分之一，但 20B mxfp4 mb≥2 必 OOM（[[trim-gptoss20b-cispo-backward-oom]] 实证），**不可行除非换更大显存或降 gpu-mem 留更多 backward 空间（但会降 KV cache 致 rollout 更慢）**。
2. **降 max-turns**（如 40→20）：直接减半 microbatch 与 rollout，但牺牲轨迹深度，影响 trim 训练效果。
3. **实现 Scheme B 流水线**（框架开发）：让 step N backward 与 step N+1 rollout 重叠，理论上可隐藏 backward 或 rollout 中较短的一段（~1.1h），step 降到 ~2.2h。需改 `four_cell_runtime` + 新增 `SchemeBRuntime`，工作量/风险高（[[capability-evolution-sync-20260918]] 评估）。
4. **换 bf16 反量化模型**：避开 mxfp4 TRITON MoE kernel 开销，生成更快。但用户明确要求用原版 mxfp4 非反量化，**不改**。

**当前 ~3.3h/step 是在"原版 mxfp4 + mb1 不 OOM + max-turns 40 深轨迹 + 不改框架代码"约束下的最优稳态速度，无加速空间。100 step 预计 ~330h（13.7 天）完成，符合预期。**

## 关键日志证据

- backward：`ddp_cispo n_microbatches=1273 update_wall_s=7868 peak_mem_mb=74774.9`（每 step）
- rollout：`Processed prompts 100% 32/32` ×多轮，`vllm_wait / vLLM generate` 心跳
- 节奏：`STEP_COMPLETE` 每 ~3.3h 落盘一次，22 个 step 时间戳见上表
- 健康判据：runTime 持续涨 + actor_heartbeat 推进 + STEP_COMPLETE 增长，三者任一前进即非 hung
