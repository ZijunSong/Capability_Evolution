# HarnessOPD 方法语义迁移对照

## 训练数据（任务书 §8.1）

| 数据 | 动作来源 | 上下文 | 用途 |
|---|---|---|---|
| RL row | Student 真实采样动作 | 当时真实 H0 token prefix（DecisionRecord 记录原样 ids） | CISPO/outcome RL |
| OPD row | Teacher 事件经合法对齐得到的 Student 动作 | Student H0 prefix + 私有 Teacher prefix 分别保存 | gated projected-gap |

Teacher 动作不替换 RL row 的 sampled action；old logprobs 不混用
（`opd/records.py::build_rl_row/build_opd_row`）。

## A 路（explicit_map）与 B 路（native_anchor）

- A 路 recipe = **带先决条件的基础动作程序**（`opd/recipes.py`）。每决策默认只
  出一个可执行 next action；后续步只在其观测进入 Student 可见历史后才监督；
  不把 Teacher 未来读到的路径/行号/ID 塞进当前动作。
- B 路 skip-to-anchor（`opd/anchors.py`）只跳内部事件，不跳过获取必要环境信息。
  三层检查（`opd/realizability.py`）：语法合法 / 环境可执行（precondition 注册
  表）/ 信息可实现（事实引用在 Student 可见历史，或可经合法基础动作重新获取 →
  acquire-then-supervise）。Teacher 对公开输入的更好推理**不算泄漏**；
  隐藏标签、未公开工具结果、fork-only 对象算（`opd/provenance.py`）。

## 损失（任务书 §9.1，逐行对照旧 `sr_opd_loss.py`）

```
g_t = σ(β·sg[ℓT_t − ℓS_t])
L_OPD = Σ_t w_t m_t g_t (sg[ℓT_t] − ℓS_t) / Σ_t w_t m_t
L = L_CISPO + λ·L_OPD
```

CISPO：`ratio = exp(new_lp − old_lp)`，`weight = sg[clamp(ratio, lo, hi)]`，
token 项 `−weight·advantage·new_lp·mask`，按全局有效 action-token 归一。
**不是**完整词表 reverse KL（`compute_sr_opd_reverse_kl` 是独立路径，不混称）；
**不是** PPO surrogate / GRPO 默认目标。`beta=0` 得 0.5，不等于 no_gate。
无监督 token → 零贡献 + 记录原因，不造 dummy target。

数值对照测试：`tests/unit/test_loss_and_grad_parity.py`（legacy reference 内嵌，
loss 与梯度 1e-6 容差，Teacher/gate 无梯度）。

## 方法开关（任务书 §9.2，`training/joint.py::MethodSpec`）

| method | RL | OPD | explicit_map | native_anchor | gate |
|---|---|---|---|---|---|
| harness_opd_rl | ✓ | ✓ | ✓ | ✓ | σ |
| rl_only | ✓ | ✗ | – | – | – |
| opd_only | ✗ | ✓ | ✓ | ✓ | σ |
| no_project | ✓ | ✓ | ✗ | ✓ | σ |
| no_gate | ✓ | ✓ | ✓ | ✓ | 恒 1 |
| aligned_teacher_sft | ✗ | CE | 可建 SFT 数据 | 可用 | n/a |

`no_project` ≠ 禁止全部对齐；`no_gate` 保留合法性/来源/隔离检查。
`aligned_teacher_sft` 是独立 SFT 管线，拒绝被当作 HarnessOPD 改名。

## 一次联合更新（`training/joint.py::joint_update`）

zero_grad → backward_rl（CISPO）→ backward_opd（gated gap，λ 显式传入后端）→
**一次** optimizer_step（test_joint_step_count）。分布式用一次全局有效 token/weight
统计（`training/backend.py::GlobalNormalizers` + `opd.losses.scale_projected_gap_
loss` 的 world-size 约定，不重复 all-reduce 全局分母）。

## Teacher 打分（任务书 §8.8，`opd/scoring.py`）

Teacher/Student 对同一 target tokens 各自 teacher-forcing；Teacher prefix 取目标
生成前的私有上下文（不含目标答案/未来输出/提示内复制目标）。recipe 投影目标标
`teacher_scored_projected_target`（≠ 原高级动作的采样概率）。Teacher scoring view
渲染规则版本化（`TEACHER_SCORING_VIEW_VERSION`），缺失合法 view 即 skip，不用
空 prompt/Student 同 prompt 凑数。

## 真实可蒸馏性 probe（任务书 §8.6，`opd/probe.py`）

真实任务/真实决策点统计：总决策/触发/fork 成功/显式映射/anchor/合法目标/有效
token/覆盖簇/梯度贡献/额外成本；三条覆盖率比率分别报告。**不沿用 10% 门槛**；
第一版工程准入=非零真实样本+非零有效 token+非零有限 OPD 梯度（只证链路，不证收益）。
