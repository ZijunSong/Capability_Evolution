# Probe A/B/C 三探针综合结论（Qwen3.5-9B，2026-11-05）

> 数据源：Probe A = 9B 四联盟 36 eval（10-26/27 批次，瘦 harness base model 行，Δadd 口径）；
> Probe B = R2 修复后 C1（r5aie，seed 11/23，NLL/D_K/invalid）+ R1 C0 漏斗（7ahc1）；
> Probe C = R3 C2 三条件（sharq，PROBE_C_C2.json，n=64 paired bootstrap）。
> 组件清单：esp=Evidence Selection、ccso=Context Consolidation、trieve=Retrieval Initiation、msc=Metacognitive Control。

## 一、Probe A：哪些组件能提高能力（Δadd，pp vs exp_b 瘦 harness 16.68/66.36/73.41/89.69）

| 联盟 | BrowseComp+ | LongSealQA | FRAMES | HotpotQA | 判读 |
|---|---|---|---|---|---|
| **esp** | **+3.54** | **+2.54** | +1.13 | -0.95 | **难基准显著增益，唯一稳定正贡献** |
| ccso | -0.04 | +0.68 | -1.18 | -1.29 | 无稳定增益 |
| trieve | -0.78 | **-3.15** | +0.85 | -2.54 | 总体负贡献（longsealqa 显著伤） |
| msc | +0.37 | +0.04 | +1.21 | +0.03 | 持平（无收益无伤害） |

**Probe A 结论**：**esp 能提高能力**（难检索基准 +2.5~3.5pp，与 27B 同款批次"esp 最强"跨规模复现）；
msc 中性；ccso/trieve 不提高（trieve 在长文档基准明确有害）。

## 二、Probe B：哪些组件能改变策略（状态级分布影响，同状态 K(s) 上 H_m Teacher vs H0 Student）

| 联盟 | C0 漏斗 | C1 策略层（D_K） | 判读 |
|---|---|---|---|
| trieve (auto_populate) | **READY**（32/32 可投影） | **D_K 不动**（R_D -0.002~-0.004 容差内，8step） | 动作可投影但**候选分布未改变** |
| ccso (sentence_compress) | NO_SUPPORT（95 触发仅 3-4 投影） | —（被 C0 拦下） | 触发正常但变换出不了学生动作 |
| esp | NO_TRIGGER（真实轨迹 0 触发） | — | 策略层无影响通道 |
| msc | NO_TRIGGER | — | 策略层无影响通道 |

**Probe B 结论**：只有 **trieve 的 auto_populate 有策略层作用通道**（能改变 Teacher 视角并投影出合法动作），
但其策略改变量（D_K）在 8step 内未偏离噪声；esp/ccso/msc 在状态级**没有可测的策略改变**——
esp 的 Probe A 增益不来自改变单步策略分布，而来自 harness 层的结构性辅助（rerank 指令改写证据排序）。

## 三、Probe C：哪些组件可以被内化（训练后 H0 部署收益，8step 预算）

| 条件 | J | 说明 |
|---|---|---|
| no_update（θ0） | 0.0731 | 基线 |
| runtime_m（外挂） | 0.0921 | 外挂 +1.9pp（CI 跨零） |
| harnessopd_m（组件训练） | 0.0674 | **G_m=-0.006，CI 跨零** |
| rl_only（纯 RL） | 0.0575 | rl_gain=-0.016（短预算 RL 扰动） |

**E_m^8=+0.010（CI [0.000,+0.030]，贴零不严格）**：组件指导方向为正但统计不显著。
R2 C1 的 NLL 剂量响应（+0.045→+0.109）证明**监督目标可被拟合**，但 D_K 不动 + C2 任务级 null
共同说明：**8step 预算内没有组件被有效内化**——"学得会目标"≠"内化成能力"。

## 四、三探针关联：明显且自洽

| 组件 | A（能力） | B（策略） | C（内化） | 一致性 |
|---|---|---|---|---|
| esp | **正**（+2.5~3.5pp） | NO_TRIGGER（无策略通道） | 未单独测（联盟级） | **A 正但 B 无通道** → 增益是 harness 结构性辅助，非策略改变，**C1/C2 探针测不到它是预期行为**（探针测的是策略内化，不是 harness 辅助） |
| trieve | 负 | **有通道**（READY）但 D_K 不动 | E_m=+0.010 贴零 | **B 有通道但 D_K 不动 → C 无收益**，三层严格一致 |
| ccso | 无 | 触发但投影失败 | 未测 | B 的投影瓶颈解释了为何不能进 C |
| msc | 中性 | NO_TRIGGER | 未测 | 无通道无收益，一致 |

**三条关联规律**：
1. **B 是 C 的必要前置**：C0 READY 是 C1/C2 的入场券（trieve 唯一走通全链）；投影失败/无触发的组件连 C 的门槛都进不了。
2. **B 的 D_K 是 C 的先行指标但非充分**：trieve D_K 不动精准预测了 C2 任务级 null——"候选分布未改变→部署行为未改变→任务无收益"的因果链成立。R2 BORDERLINE（FIT_IMPROVED_POLICY_UNRESOLVED）对 C2 null-ish 的预测力得到验证，**探针链路有校准效力**。
3. **A 与 B/C 测的是两种不同收益**：esp 的 A 增益走 harness 结构辅助通道（对探针不可见），trieve 的潜在收益走策略内化通道（探针可见但目前为零）——**A 正 ≠ 可内化，A 零 ≠ 无价值**，两列排序不可混用（这正是任务书 §2.1 反复强调的口径分离）。

## 五、行动含义
- **部署侧**：esp 值得保留在 harness（便宜、正增益、无需训练）；trieve/ccso 外挂无依据。
- **训练侧**：继续内化研究应聚焦 **auto_populate（唯一有投影通道的组件）**，需扩预算（16/32 step）或扩 panel 提高功效；sentence_compress 需先解决投影瓶颈（Commit D 观测级通道）。
- **探针侧**：C1 BORDERLINE→C2 null 的预测链已验证一次（n=1 组件），C3 校准需再积累 2-4 个机制不同的组件才能给出选择器的预测准确率。
