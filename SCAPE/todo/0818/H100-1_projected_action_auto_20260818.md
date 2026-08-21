# H100-1：`auto_populate_first_search` 的 Projected-Action OPD 重构

## 0. 本轮目标

不要继续扩旧的 `AUTO + reverse 8-way Route-KL` recipe。最新实验已经说明：

- `auto_populate_first_search` 的 same-state/value signal 可以是正的；
- route proxy 可以显著改善；
- 但 actual-LoRA、no-privilege、real multi-step closed-loop 下 Student 没有超过训练前 Student，且没有超过 shuffle control；
- H20 clean-init 修复 evaluator/load contract 后，已完成的 unshuffled DEV seeds 仍低于 CLEAN_BASE。

本轮要验证一个更根本的机制假设：

> `auto_populate_first_search` 真正需要被内化的行为，不是“第一次搜索时选择哪个 route”，而是“第一次成功 search 之后，模型自己执行一次 `curate(add_ids=...)`，复现 harness 自动写入 curated set 的状态变化”。

旧 AUTO 实验只蒸馏 8-way route/tool-name，不蒸馏 query、doc ids、JSON args，也没有把 harness 自动 side-effect 映射成 Student 可以执行的动作。因此本轮必须改变 **distillation target**，而不是继续扫 loss。

内部实验名统一使用：

```text
PROJECTED_ACTION_AUTO
```

暂时不要把它当论文方法名。

---

## 1. 强制约束

1. 使用 8×H100。
2. Student 推理时必须 `student_inference_privilege=false`。
3. 必须训练 actual LLM weights / PEFT LoRA；禁止用 `route_head.pt` 代替。
4. 必须运行真实 multi-step Search closed-loop；route proxy 只能做诊断。
5. 训练后的 Student 的第一目标是：
   - `Student_after > Student_before`
   - 然后再比较 Matched Text / OPSD / OPHSD / SEED 等 baseline。
6. **不要求 Student 超过带 Harness 的 Teacher。** Teacher 只作为 upper/reference，不作为 GO gate。
7. 禁止 mock、伪造 teacher action、伪造 gold、重复样本冒充 unique states。
8. 如果 projected-action 方法仍不能让 Student 超过训练前 Student：
   - 先做 case/root-cause analysis；
   - 允许一次 substantive redesign；
   - 第二次仍失败则 discard 该方法；
   - 不允许通过缩窄 claim 来把失败包装成主贡献。
9. 先读：
   - `/mnt/songzijun/CLAUDE.md`
   - 最新 `result-simplified*.md`
   - AUTO 相关现有脚本、训练器、real closed-loop evaluator。
10. 新代码、新输出放独立目录，禁止覆盖旧 paper-grade artifact。

建议输出根目录：

```text
outputs/0818_projected_action_auto/
```

---

## 2. 第一阶段：代码审核——确认旧 AUTO target mismatch

先不要训练。

必须定位并给出代码级证据，回答：

1. `auto_populate_first_search` 在真实 runtime 中：
   - 何时触发；
   - 触发前 curated set；
   - search 返回哪些 doc ids；
   - 触发后 harness 自动加入哪些 ids；
   - 是否产生模型可见的显式 `curate` tool call。
2. 旧 AUTO OPD 数据中的 target 到底是什么：
   - 仅 route distribution？
   - tool name？
   - 是否包含 `curate(add_ids=...)`？
   - 是否包含真实 doc ids / args？
3. 旧训练 sample 是在：
   - search 前；
   - search 当步；
   - 还是 search 完成后的下一决策点？
4. 旧 same-state proxy 为什么可以大涨但 closed-loop 不涨：
   - 至少抽取 30 个 `route proxy improved but real behavior unchanged/worse` cases；
   - 检查 Student 是否学会“搜完立即 curate”；
   - 统计首次成功 search 后 1/2 个 turn 内的 curate rate、add_ids 合法率、加入 relevant evidence 的比例。

输出：

```text
AUTO_TARGET_CONTRACT_AUDIT.md
AUTO_TARGET_CONTRACT_AUDIT.json
AUTO_FAILURE_CASES.jsonl
AUTO_FAILURE_CASE_ANALYSIS.md
```

若发现旧实验实际上已经完整蒸馏了 post-search `curate(add_ids)`，则停止本方案并根据真实代码重做机制判断；禁止为了迎合假设修改证据。

---

## 3. 第二阶段：实现 Harness Intervention → Student Action Projection

### 3.1 定义 projection

对 Student 自己的 on-policy trajectory，在第一次成功 search 后建立 fork。

记：

```text
s_pre   = search observation 已返回、但 AUTO side-effect 尚未应用的状态
s_full  = full harness 对同一状态应用 AUTO 后的状态
ΔH      = curated_ids(s_full) - curated_ids(s_pre)
```

若：

```text
ΔH.add_ids != []
```

则生成 Student-native projected action：

```json
{
  "tool": "curate",
  "arguments": {
    "add_ids": ["...真实合法doc ids..."],
    "remove_ids": []
  }
}
```

要求：

- 所有 `add_ids` 必须来自 Student 在 `s_pre` 已经观察到的 documents/search results；
- 不允许把 Student 不可见的 document id 投影进去；
- projection 必须由 deterministic runtime state delta 产生，不允许 LLM 自己“猜一个 target”；
- 记录 provenance：query_id、state_hash、search result ids、pre/post curated ids、projected tool call。

对于 `ΔH.add_ids=[]` 的状态不强行生成正例。

### 3.2 数据集

重新从真实 on-policy Student 收集，不复用旧的“first-search route”数据作为主训练数据。

最低要求：

```text
>= 1000 unique projected-action-positive states
>= 300 unique query ids
query-disjoint train/valid/test
```

如果真实支持不足 1000：
- 可以降低到实际可获得规模；
- 必须报告 unique/support；
- 禁止 resample 后宣称 unique 数满足要求。

建议 split：

```text
train 70%
valid 15%
test 15%
```

另存一份 matched shuffled projection：
- same states
- same update budget
- 保持 add-count / tool marginal 尽量一致
- 打乱 state ↔ projected add_ids 对应关系
- 必须保证打乱后 ids 在当前 state 中仍合法，否则该 control 无效。

---

## 4. 第三阶段：8 GPU 第一轮训练矩阵

不要再做“4 种普通 route loss”的无效 sweep。

8 卡运行 4 个 substantive target variants × 2 seeds：

| GPU | Variant | Seed |
|---|---|---:|
| 0 | `OLD_REVERSE_ROUTE_KL`（旧 recipe 对照） | 42 |
| 1 | `OLD_REVERSE_ROUTE_KL` | 43 |
| 2 | `PROJECTED_ACTION_CE` | 42 |
| 3 | `PROJECTED_ACTION_CE` | 43 |
| 4 | `PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL` | 42 |
| 5 | `PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL` | 43 |
| 6 | `SHUFFLED_PROJECTED_ACTION_CE` | 42 |
| 7 | `SHUFFLED_PROJECTED_ACTION_CE` | 43 |

### 4.1 `PROJECTED_ACTION_CE`

只在 canonical tool-call span 上训练：

- tool name `curate`
- JSON key structure
- `add_ids`
- `remove_ids`

不要训练无关自然语言 token。

### 4.2 `PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL`

先监督 projected `curate` action，然后执行该 action 得到 projected next state；在下一决策点再做一次 teacher/student KL。

目的不是“多加一个 loss”，而是测试：

> 学会 harness side-effect 的可执行动作后，是否还需要 continuation-level teacher guidance 才能闭合成真正的策略收益。

### 4.3 Anchor

使用当前稳定 anchor 配置作为起点，不要为了追结果每个 variant 单独调参。

如果发现 projected action CE 梯度或 token mask 不正确，先修 trainer，不要换 objective。

---

## 5. 第四阶段：真实 closed-loop GO gate

先 DEV128，再 TEST256（若当前 manifest 实际少于 256，使用全部 unique test queries并报告数量）。

必须比较：

```text
BASE_STUDENT
OLD_REVERSE_ROUTE_KL
PROJECTED_ACTION_CE
PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL
SHUFFLED_PROJECTED_ACTION_CE
FULL_HARNESS_REFERENCE   # 只做参考，不要求超越
```

主指标：

1. overall/task reward
2. trajectory evidence recall
3. final answer recall
4. curated evidence recall
5. invalid tool rate
6. 首次成功 search 后 1 turn 内 `curate` rate
7. projected add_ids 的 relevant/qrel hit rate
8. turns / tool cost

### GO 条件

不要求超过 Harness Teacher。

必须满足：

```text
A. best projected Student > BASE_STUDENT
B. projected > shuffled projected control
C. 至少两个独立 seed 同方向
D. pooled paired bootstrap 95% CI(Student - Base) > 0
E. invalid_tool 无实质退化
F. 行为机制成立：search 后及时 curate 的比例明显提高
```

若 A/B 不成立，不能进入 paper main line。

---

## 6. 失败后的唯一允许 redesign

如果 projected action 明显学会、行为指标提升，但最终 reward 仍不提升，必须先读 50 个 paired cases：

```text
Base success / Student fail
Base fail / Student success
Both fail but Student curates earlier
Student adds wrong top-K
Student over-curates and loses search diversity
```

只允许依据 cases 做一次 redesign，例如：

- `curate` 时机 gate；
- top-K projection 改成 relevance-aware subset；
- 把 AUTO 从“一律自动收 top-K”改成“只投影 teacher 最终保留的 ids”；
- projected action 加 outcome/value weighting。

禁止把失败解释成“只要 route 学到了就算成功”。

第二次 real closed-loop 仍不超过 Base：`DISCARD PROJECTED_ACTION_AUTO`。

---

## 7. 必须输出的 artifacts

```text
RUN_MANIFEST.json
STATUS_LIVE.md

AUTO_TARGET_CONTRACT_AUDIT.md
AUTO_TARGET_CONTRACT_AUDIT.json
AUTO_FAILURE_CASES.jsonl
AUTO_FAILURE_CASE_ANALYSIS.md

PROJECTED_ACTION_SCHEMA.md
PROJECTED_ACTION_DATA_AUDIT.md
PROJECTED_ACTION_TRAIN.jsonl
PROJECTED_ACTION_VALID.jsonl
PROJECTED_ACTION_TEST.jsonl
SHUFFLED_PROJECTED_ACTION_TRAIN.jsonl

TRAINING_CELLS.csv
DEV_REAL_CLOSED_LOOP.csv
TEST_REAL_CLOSED_LOOP.csv
PAIRED_BOOTSTRAP.csv
MECHANISM_METRICS.csv
CASE_ANALYSIS.md

H1001_0818_HANDOFF.json
SHA256SUMS
```

最终 handoff 只能给出以下之一：

```text
GO_PROJECTED_ACTION_AUTO
REDESIGN_ONCE_PROJECTED_ACTION_AUTO
DISCARD_PROJECTED_ACTION_AUTO
```

不要输出模糊的“proxy positive / promising”作为主结论。
