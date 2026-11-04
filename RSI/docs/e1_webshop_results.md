# E1-WebShop：9B 进化结果（champion 出现，待独立确认）

> **artifact**：`runs/e1_webshop_9b_r9/evolve/evolve_result.json`（job bojde=webshop-9b r8 链，
> 2026-11-04 17:19 落盘；11 候选评估，PLATEAU 收官）。
> **基线对照**：E0 口径（guide §12）——H0-ReAct 空 bank，manifest 12+12 asin 组（digest 4a46047d）。

## 核心结果

| 项 | 值 |
|---|---|
| 评估候选 | 11（全部 legal，0 组件错误） |
| 候选 macro 分布 | 0.006 ~ **0.589**（中位 ~0.34） |
| **champion** | `accbce6a59…`（bank_digest 见 manifest），gen2 当选 macro **0.45** |
| champion 组件 | **GoalCompletionGuard + ActionRecoveryGuard**（双组件组合） |
| gen3 复测 | champion bank 作为 parent 在 panel 上得分 **≥0.589**（压过单候选 0.589 的 bd8ecc8aa6——组件组合有互补效应） |
| baseline（空 bank） | < 0.028（gen0 候选 0.0278 即夺冠） |

## 机制层里程碑

1. **proposer 格式修复实锤**：11/11 候选 ACCEPTED（对比 r2 时代 18/18 拒绝）。
2. **dedup 生效**：`near_dup` 标记出现在多个重复提议上（缓存命中省 panel）。
3. **PR-5 baseline 保留持续生效**：gen1/3/4/5 都是 `incumbent retained`——包括 gen3 拒绝 0.589 单候选（champion 组合 ≥0.589）。
4. **行为解释（初步）**：GoalCompletionGuard（veto None 动作 + last-obs 非法时 block finish + 注入"不要提前 finish"指令）+ ActionRecoveryGuard——直接对症 webshop 的"94% finish 买不对"失败模式（H0 formal 链结论）。

## 警示（未完成验证前不声称增益）

1. **gen1/3-5 的 parent 复测分数跨度大（0.19→0.59→…）**，同一 bank 相邻代分数差 >30pp——panel 方差极大（12 任务、temp 1.0）。champion 的 0.45 vs baseline <0.028 的 Δ 在这个方差下**不足以上下结论**。
2. **独立 confirm 未做**：champion 需冻结后在 fresh confirm 池（webshop 官方 500-goal 面板的 asin 隔离子集）复测，预注册阈值后才能称 confirmed。
3. 11 个候选里 6 个被标 `near_dup`（换描述重提）——proposer 多样性不足，与 webshop-27b r3 的"6/6 同名 error_tracker"一致。

## 下一步

1. **E3 前置 confirm**：champion bank 冻结 → confirm 池（从未参与开发的 asin 组）× 24 任务 → 预注册 Δ 判据。
2. 方差控制：panel 12 任务 temp 1.0 的配对方差需实测（同 bank 复跑 2-3 次）后再定 confirm 规模。
3. AppWorld-9B r3（3gwab）仍在跑；alfworld 臂待 webshop 流程稳定后复制。
