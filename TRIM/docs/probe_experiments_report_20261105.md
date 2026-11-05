# Probe A/B/C 探针实验报告：组件能力增益、策略改变与可内化性

> 模型：Qwen3.5-9B（dense bf16，H0=瘦 harness）。实验窗口：2026-11-01 ~ 11-05。
> 分支 `rsi-implementation`；探针实现基线 `dc4b2c9`，测量修复（measurement_v4）见本报告 §6。
> 组件：esp=Evidence Selection & Prioritization、ccso=Context Consolidation、trieve=Retrieval Initiation（auto_populate_first_search）、msc=Metacognitive Search Control。
> 口径：Probe A=Δadd（J(H_m)−J(H0)，同面板同 seed）；Probe B=同状态 K(s) 上 H_m Teacher vs H0 Student 的分布影响（NLL/D_K）；Probe C=训练后 H0 部署收益（G_m^8/E_m^8 paired bootstrap）。

---

## 1. 执行总览（R0→R3 四阶段）

| 阶段 | 内容 | 任务 | 结果 |
|---|---|---|---|
| R0 | 离线重分类：旧 r3 漏斗按 v4 口径重算 + GATE_V4 重判 | CPU（diagnose/rejudge CLI） | trieve 4 行全 DEPRIORITIZE→BORDERLINE；ccso 91 条=direct 尝试失败非成功 |
| R1 | 测量 smoke：16+16 题、双单成员、5 项验收 | 7ahc1（8gpu, 10min） | **gate 7/7 PASS** |
| R2 | 修复后 C1：双 seed 4→8，真实 checkpoint | r5aie（8gpu, 34min） | **双 seed BORDERLINE**（NLL 可靠改善、D_K 不动） |
| R3 | C2 三条件：rl_only/harnessopd_m 训练 + 4 条件评测 | 0g5aj+6zfmd+sharq | **PROBE_C_C2.json 出全量 G/E** |

## 2. Probe A：哪些组件能提高能力（Δadd，pp）

基线 = exp_b 瘦 harness base model（bcplus 16.68 / longsealqa 66.36 / frames 73.41 / hotpotqa 89.69），36 个 formal eval（3 seed×4 bench×4 联盟，10-26/27 批次）：

| 联盟 | BrowseComp+ | LongSealQA | FRAMES | HotpotQA | 判读 |
|---|---|---|---|---|---|
| **esp** | **+3.54** | **+2.54** | +1.13 | -0.95 | **唯一稳定正贡献**（难检索基准显著增益，与 27B 批次"esp 最强"跨规模复现） |
| ccso | -0.04 | +0.68 | -1.18 | -1.29 | 无稳定增益 |
| trieve | -0.78 | **-3.15** | +0.85 | -2.54 | 总体负（longsealqa 明确有害） |
| msc | +0.37 | +0.04 | +1.21 | +0.03 | 中性 |

**结论：esp 能提高能力；msc 中性；ccso/trieve 不提高。**

## 3. Probe B：哪些组件能改变策略（状态级）

C0 漏斗（v4 attempted/passed 口径，128 状态/联盟）+ R2 C1（双 seed 4→8step，真实 checkpoint，GATE_V4）：

| 联盟 | C0 verdict | 关键漏斗数 | C1 策略层（D_K） |
|---|---|---|---|
| trieve | **READY** | 32/32 可投影、6531 监督 token、全 query 覆盖 | **D_K 不动**（R_D −0.002~−0.004，容差内，双 seed 复现）；NLL 可靠改善（R_N@8≈+0.104，剂量响应 +0.045→+0.109） |
| ccso | NO_SUPPORT | 95 触发但 91 卡投影（`projection_direct attempted=95/passed=4`） | 被 C0 拦下（监督 2-3 状态 < 门槛） |
| esp | NO_TRIGGER | 128/128 未触发（adaptive_rerank 是 synthetic 启发式） | 无通道 |
| msc | NO_TRIGGER | 128/128 未触发（verify_tool 无真实 verifier） | 无通道 |

**结论：只有 trieve 的 auto_populate 有策略层作用通道；其策略改变幅度（D_K）在 8step 内未偏离噪声。esp/ccso/msc 在状态级无可测的策略改变。**

关键测量修复（R1 实锤）：旧 C1 的 invalid 0.70-0.75 中**截断只贡献 ~1/8**（128→2048 token 预算对照：截断 6/8→0/8，invalid 0.875→0.75）——主失败是真实行为失败（inaccessible_reference/schema），非测量伪影。

## 4. Probe C：哪些组件可以被内化（8step 预算）

C2 三条件（auto_populate，从 θ0 训 8step，正式 HarnessOPD 语义；C_select 64 题 full episode，paired bootstrap 2000）：

| 条件 | J | 说明 |
|---|---|---|
| no_update（θ0） | 0.0731 | 基线 |
| runtime_m（θ0+H_m 外挂） | 0.0921 | 外挂 +1.9pp（CI 跨零） |
| harnessopd_m（组件训练） | 0.0674 | **G_m=−0.006，CI [−0.047,+0.026] 跨零** |
| rl_only（纯 RL） | 0.0575 | rl_gain=−0.016（短预算 RL 扰动） |

**E_m^8 = +0.010，CI [0.000, +0.030]（贴零不严格）**：组件指导相对纯 RL 方向为正但统计不显著。61-62/64 题平局——效应量低于 panel 噪声地板。

## 5. 三探针关联：明显且自洽

| 组件 | A 能力 | B 策略 | C 内化 | 一致性解读 |
|---|---|---|---|---|
| esp | **正** | NO_TRIGGER（无策略通道） | 未单独测 | A 增益走 **harness 结构辅助**通道（rerank 指令改写证据排序），对策略探针不可见是预期行为 |
| trieve | 负 | **有通道**但 D_K 不动 | E_m 贴零 | **B 的 D_K 不动精准预测了 C2 null**——因果链闭合（分布未变→行为未变→任务无收益） |
| ccso | 无 | 触发但投影失败 | 进不了 C | B 的投影瓶颈解释了 C 的缺席 |
| msc | 中性 | NO_TRIGGER | 未测 | 无通道无收益，一致 |

三条规律：
1. **B 是 C 的必要前置**：C0 READY 是内化实验的入场券（四联盟中只有 trieve 走通全链）。
2. **B 的 D_K 是 C 的先行指标（已验证一次）**：R2 BORDERLINE（FIT_IMPROVED_POLICY_UNRESOLVED）→ R3 null-ish，探针的"否决"预测链成功；"确认"方向（D_K 动→任务有收益）尚无正样本。
3. **A 正 ≠ 可内化**：esp 的增益是便宜的 harness 辅助（部署侧保留），trieve 才是潜在可内化对象（训练侧聚焦）——两列排序不可混用。

## 6. 证据边界（诚实表述）

**当前证据支持**："8-step 预算、64 题 panel、auto_populate 单组件条件下，未检测到可内化证据（D_K 不动 + E_m CI 贴零 + G_m 跨零，三层一致）"。**不是**"所有能力都不能被内化"。

三个局限：
1. **功效不足 ≠ 效应为零**：E_m 方向为正、CI 下界贴 0 而非明确负值；64 题 panel ±2.5pp 噪声可能淹没 ~1pp 的真实效应。
2. **预算只探了 8step**：16step 是预注册的 borderline 扩展（未跑）；跨规模证据（GPT-OSS-20B 100step 消融 +0.94~+3.37pp）来自"RL+OPD 联合 vs 纯 OPD"对照，缺 no_update 基线，未隔离蒸馏的独立贡献；9B 50step HarnessOPD 批（本报告同期）四联盟 vs θ0 = ±1pp 噪声级，与 R3 互证。
3. **通道覆盖不全**：观测级组件（sentence_compress）需 Commit D 的 transform_then_policy 通道才有参测资格——当前探针对其"未开放"而非"判死刑"。

**探针的实用性定位**：把"训练是否带来任务收益"的验证成本从 50-200step 大训练降到 8step 探针+64 题面板（~1/6 成本），且在 9B 上提前给出了与 50step 训练一致的答案。它当前能可靠**否决**（D_K 不动→别指望收益），确认能力待积累。

## 7. 执行损耗与工程沉淀

- **8 轮重提**：4 轮 entry 脚本层（heredoc TRAIN_RC 提前展开 / env 前缀传递丢失+残留块 / 续行符字面双反斜杠 / submit OUT 硬赋值覆盖环境变量致数据竞争）+ 4 轮 flashinfer gdn 坏节点（32764-32767，与 9B teacher 训练同款，重提解）。
- **wrapper 收尾 find maxdepth=3 漏 depth4 adapter**：两训练本体全部成功但 READY_MANIFEST 未写，手工补齐（已修 maxdepth→5）。
- 产物台账：两训练 adapter 均在 `outputs/probe-c-c2-9b-*-8step-16gpu_20261104_r6/checkpoints/{scape_seed,rl}/step_000008/adapter`；eval 终版 OUT=`probe-c-r3-c2-eval-8gpu_20261104_r10/c2/`。

## 8. 后续（按 guide R4/R5）

1. **C3 校准**：用 rieve 50step checkpoint 校准 C1 预测力；同 checkpoint 条件补 H0 base/RL-only/HarnessOPD 同题配对。
2. **提功效**：panel 64→256 题 + 预算 16/32/50-step 曲线，回答"效应是零还是被噪声淹没"。
3. **补通道**：sentence_compress 的 transform_then_policy 试点（guide Commit D）。
4. 部署建议不变：**esp 保留在 harness（无需训练）；trieve 是唯一值得继续投入训练的内化候选**。

## 附：产物与 job 索引

| 阶段 | job | OUT |
|---|---|---|
| 旧 C0/C1（r3） | t-20261101145740-p06y6 | `outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3` |
| R1 smoke | 7ahc1（r1 koxph/r2 8oiig 死于 pair revision/校验签名） | `.../probe-c-r1-smoke-8gpu_20261104_r3/Qwen_Qwen3.5-9B` |
| R2 C1 | r5aie（r1 suxd1 死于 StepRecord 下标） | `.../probe-c-r2-c1-8gpu_20261104_r2/Qwen_Qwen3.5-9B` |
| R3 训练 | 0g5aj（rl_only）/6zfmd（harnessopd_m） | `.../probe-c-c2-9b-*-8step-16gpu_20261104_r6` |
| R3 eval | sharq（终版） | `.../probe-c-r3-c2-eval-8gpu_20261104_r10/c2` |
