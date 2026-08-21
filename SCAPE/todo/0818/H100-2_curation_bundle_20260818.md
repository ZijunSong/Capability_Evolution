# H100-2：`importance_tagging + subtractive_curation` 联合内化

## 0. 核心假设

不要继续单独救 `importance_tagging`。

最新 proper same-`xi_t` K4/K8 fork 已经表明，`importance_tagging` 单组件平均 teacher-minus-student 为负，正式 gate 失败。旧 approximate positive signal 作废。

但这不等于 `importance_tagging` 对真正的 curation 行为没有价值。根据组件语义：

- `importance_tagging` 给 `curate` 提供 importance；
- `subtractive_curation` 在 curated set 满时使用 importance 自动驱逐低价值文档并加入新文档；
- Student 没有 importance 接口，但有原生 `curate(add_ids, remove_ids)`。

因此新的机制假设是：

> importance 本身不是要被 Student 复现的输出；它应当与 subtractive curation 联合，让 harness 把“重要性判断”兑现成真实的 add/remove state transition，然后再把这一 state transition 投影成 Student 原生可执行的 `curate(add_ids, remove_ids)`。

内部实验名：

```text
PROJECTED_CURATION_BUNDLE
```

---

## 1. 强制规则

1. 8×H100。
2. actual LoRA / actual model only。
3. Student inference 无 `importance`、无 `subtractive_curation`、无其他训练期 privilege。
4. 不允许使用旧 subtractive all-zero evaluator 直接训练。
5. 必须先修复 terminal gold/reference contract，并重新收集有效 curation-event 数据。
6. 不要求 Student 超过 Full Harness Teacher；主目标：
   - Student_after > Student_before
   - 再超过其他 distillation baselines。
7. `importance_tagging` 单组件失败结果必须保留，不能删除或改写。
8. 如果联合组件也没有正 value / real closed-loop gain：
   - redesign 一次或 discard；
   - 禁止缩 claim。

输出目录：

```text
outputs/0818_projected_curation_bundle/
```

---

## 2. 第一阶段：修复 subtractive 数据与 evaluator contract

先审计现有结果所指出的问题：

```text
terminal_reward_available = 0
gold/reference = 0/256
valid_remove_ids = 0
```

必须从真实 BrowseComp/Search evaluator 重新建立可评分 row。

每个 row 至少包含：

```text
query_id
state_hash
documents
curated_ids_pre
curated_importance_pre
teacher curate action / incoming docs
gold/reference evidence ids
qrel / terminal scoring fields
component mask
tool history
remaining budget
```

### 2.1 事件采样必须聚焦“接近容量上限”

不要随机收一堆永远不会触发 subtractive 的状态。

优先采样：

```text
curated_count >= 24
```

并重点 oversample：

```text
curated_count == 30
新候选 evidence 到达
subtractive curation 实际触发
```

但最终报告 unique state/query 数量，不能把 oversampling 伪装成自然频率。

### 2.2 oracle sanity

修复后必须通过：

- synthetic oracle curated recall > base；
- terminal reward 可非零；
- final answer recall 可非零；
- `valid_add_ids > 0`；
- `valid_remove_ids > 0`；
- remove id 必须来自当前 curated set；
- add id 必须来自当前可见 documents。

任何一项失败，不允许开始训练。

---

## 3. 第二阶段：联合组件的 executable-effect fork

不要复用“importance ON vs OFF 的第一动作”作为 value gate。

新的比较对象是“联合 harness effect 是否能被 Student 原生动作复现”。

同一 `xi_t` 建两条分支：

### Full branch

```text
importance_tagging = ON
subtractive_curation = ON
```

执行真实 curation event，记录：

```text
curated_ids_before
curated_ids_after
removed_ids = before - after
added_ids   = after - before
```

### Reduced/projected branch

关闭两个组件，只允许 Student/native runtime。

将上面的 state delta 编译为：

```json
{
  "tool": "curate",
  "arguments": {
    "add_ids": [...],
    "remove_ids": [...]
  }
}
```

执行这个 projected action 后，两条分支都使用 reduced continuation policy 跑 K4/K8。

### Gate

需要正式回答：

```text
Does executing the projected native curate action preserve the positive downstream effect
of the full importance+subtractive intervention?
```

至少：

```text
512 states × 2 seeds × K4/K8
```

报告：
- mean T(projected/full effect) - reduced baseline
- CI
- tool cost
- curated evidence recall delta
- terminal reward delta

如果联合 projected effect 本身都不正，直接 `DISCARD_CURATION_BUNDLE`，不要训练 LoRA。

---

## 4. 第三阶段：训练数据

只使用 Student 已观察状态中的合法 ids。

目标 action：

```text
curate(add_ids=<full harness ultimately adds>,
       remove_ids=<full harness ultimately removes>)
```

不要监督 importance 数值本身。

至少构造：

```text
CURATION_BUNDLE_TRAIN
CURATION_BUNDLE_VALID
CURATION_BUNDLE_TEST
```

并生成两个关键 controls：

### Control A：`SUBTRACTIVE_ONLY_PROJECTED`

关闭 importance_tagging，仅使用 subtractive 自身产生的 state delta。

目的：判断 importance 是否真的通过组合给 subtractive 提供额外价值。

### Control B：`SHUFFLED_CURATION_DELTA`

保持同一 state、相同 add/remove count budget，但打乱合法的 remove/add target 对应。

目的：排除“学会更频繁 curate”而不是学会状态相关取舍。

---

## 5. 第四阶段：8 GPU 训练矩阵

第一轮：

| GPU | Variant | Seed |
|---|---|---:|
| 0 | `COMBINED_PROJECTED_ACTION_CE` | 42 |
| 1 | `COMBINED_PROJECTED_ACTION_CE` | 43 |
| 2 | `COMBINED_PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL` | 42 |
| 3 | `COMBINED_PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL` | 43 |
| 4 | `SUBTRACTIVE_ONLY_PROJECTED_ACTION_CE` | 42 |
| 5 | `SUBTRACTIVE_ONLY_PROJECTED_ACTION_CE` | 43 |
| 6 | `SHUFFLED_CURATION_DELTA_CE` | 42 |
| 7 | `SHUFFLED_CURATION_DELTA_CE` | 43 |

不要加入 importance 单组件 LoRA；它已经被 proper fork gate 否决。

### Loss 必须覆盖 arguments

训练 mask 至少覆盖：

```text
tool=curate
add_ids
remove_ids
```

如果 `remove_ids` token mask/serialization 不可解析，先修 trainer。

---

## 6. 第五阶段：real closed-loop

必须从 initial state 跑真实 multi-step Search。

比较：

```text
BASE_STUDENT
COMBINED_PROJECTED
SUBTRACTIVE_ONLY_PROJECTED
SHUFFLED_CURATION_DELTA
FULL_HARNESS_REFERENCE
```

主指标：

```text
overall_reward
curated_evidence_recall
trajectory_recall
final_answer_recall
invalid_tool_rate
curate_event_rate
valid_add_rate
valid_remove_rate
relevant_added_rate
irrelevant_removed_rate
curated_set_churn
tool_calls / turns
```

### GO gate

```text
1. Combined Student > Base Student
2. Combined > Subtractive-only
3. Combined > shuffled
4. >=2 seeds same positive direction
5. paired bootstrap CI(Student-Base) > 0
6. valid remove/add 行为机制成立
```

不要求超过 Full Harness。

如果 Combined > Base 但不 > Subtractive-only：
- importance 没有提供额外组合价值；
- 从方法中 discard importance，不允许把“联合”包装成贡献。

如果 Combined 连 Base 都不过：
- 读 cases；
- 允许一次 redesign；
- 再失败则整个 curation bundle discard。

---

## 7. Case analysis 必做

至少 60 个 paired cases，分层：

```text
Full harness success / Base fail
Combined Student success / Base fail
Base success / Combined fail
Wrong removal
Wrong addition
Correct removal but premature stop
Curated set oscillation
Repeated low-value curation
```

特别判断：

1. importance 是否真正改变了“谁被删”；
2. subtractive 是否只是在容量满时机械触发；
3. Student 是否学到了“容量管理”而不是单纯增加 curate 次数；
4. 更好的 curated set 是否真正传到 final answer。

输出机制结论，不允许只给 aggregate。

---

## 8. Artifacts

```text
RUN_MANIFEST.json
STATUS_LIVE.md

CURATION_EVALUATOR_REPAIR.md
CURATION_ORACLE_SANITY.json
CURATION_EVENT_COVERAGE.csv
CURATION_BUNDLE_SCHEMA.md

CURATION_BUNDLE_VALUE_PER_STATE.jsonl
CURATION_BUNDLE_K4_K8_GATE.json

CURATION_BUNDLE_TRAIN.jsonl
CURATION_BUNDLE_VALID.jsonl
CURATION_BUNDLE_TEST.jsonl
SHUFFLED_CURATION_DELTA.jsonl

TRAINING_CELLS.csv
DEV_REAL_CLOSED_LOOP.csv
TEST_REAL_CLOSED_LOOP.csv
PAIRED_BOOTSTRAP.csv
CURATION_MECHANISM_METRICS.csv
CURATION_CASE_ANALYSIS.md

H1002_0818_HANDOFF.json
SHA256SUMS
```

最终只能给：

```text
GO_CURATION_BUNDLE
REDESIGN_ONCE_CURATION_BUNDLE
DISCARD_IMPORTANCE_KEEP_SUBTRACTIVE
DISCARD_CURATION_BUNDLE
```
