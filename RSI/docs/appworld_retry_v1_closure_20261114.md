# AppWorld 有界重试 v1 结案（2026-11-14）

## 结果（guide §4.5 诚实标注）

- **end state: `CANDIDATES_REPEATEDLY_INVALID` → honest_label = NO_TEACHER**
- job `t-20261113173743-lhipv`（v6）正常退出（非 crash 非 hang）——job success 但方法无产出的合法案例
- H0 parent panel 串行 12 题完整跑完（~36min，JuiceFS scratch 修复生效）
- proposer 2 代 × 2 slots × ≤2 attempts = 8 次提议，全部被 PR-04 证据契约拒绝：
  `hypothesis too short to name a mechanism` + `no failure_evidence_refs cited`
- evaluated_candidates = 0，champion = None，预算未浪费在低质量 panel 上（契约前置拦截）

## 结论

**9B self-proposer 在证据契约下无法产出合规候选。** 与 e1 r4/r5 时代对比：当时
proposer 输出被宽松接受（16 候选全 ACCEPTED 但全是 no-op/guard 家族），PR-04
契约把"看起来像组件"和"有机制有证据"区分开后，9B 的提议能力显出真实水位。

## 与 guide §4.5 的对照

- §4.5 #6：确认失败即停止教师路线，不看确认集再改 bank 重测 → **遵守**
- §4.5 结束状态区分：`PROPOSAL_STALLED/CANDIDATES_REPEATEDLY_INVALID` ≠ RUNTIME_INVALID
  → 本轮是前者（提议能力问题），不是基础设施问题（五连 hang 已修复，链路全通）
- §9.5：AppWorld 作为"教师生成/可实现性边界"的案例纳入论文叙事，不硬塞三域 OPD

## 下一步选项（不在本轮执行）

1. **ALFWorld 优先推进**（预冻结 622625de + v2 manifest 就绪）：fresh confirm → probe →
   readiness → pilot——AppWorld 失败不阻塞 ALFWorld/WebShop 内化线
2. AppWorld teacher 路线暂停；如重启：(a) assisted proposer 上界对照（guide §4.5 #5，
   不计入 9B 自进化成功）；(b) 专门的工具使用 SFT 分支（guide §4.5 末段）
3. 五连 hang 修复（JuiceFS scratch 重定向）是本轮最有价值的副产品——所有后续
   AppWorld job（包括 ALFWorld/WebShop 的同型 job）受益
