# SCAPE H100-3 指令：Same-Environment-State Policy Influence Map

## 任务定位

本机专门实现 SCAPE Pre-stage 的第二轴：

```text
Same-State Policy Influence I_m
```

目标不是看 module removal 后最终分数，而是直接测：

```text
同一个真实 environment state xi_t
Full Harness view 与 -m view
到底让同一个 policy 的下一次 tool decision 改变多少？
```

**不训练。**

---

# 1. 工作目录

```text
/mnt/songzijun/Capability_Evolution/SCAPE
```

分支：

```text
exp/h1003-same-state-influence
```

输出：

```text
outputs/h100_3_influence/
```

必须生成：

```text
RUN_MANIFEST.json
STATUS_LIVE.md
SNAPSHOT_SCHEMA.md
DUAL_VIEW_PARITY.md
INFLUENCE_PER_STATE.jsonl
INFLUENCE_BY_COMPONENT.csv
INFLUENCE_BY_COMPONENT.md
NULL_CONTROL_REPORT.md
SHA256SUMS
```

---

# 2. 必须先实现的基础设施

## 2.1 Environment Snapshot

建立可序列化：

```python
xi_t = {
    "task": ...,
    "observations": ...,
    "candidate_pool": ...,
    "curated_set": ...,
    "evidence_links": ...,
    "verification_records": ...,
    "search_history": ...,
    "budget_state": ...,
    "tool_runtime_state": ...,
    "turn_id": ...
}
```

要求：

```text
snapshot 不包含未来 observation
snapshot 可 replay
snapshot hash 稳定
snapshot 不依赖 rendered prompt 文本
```

## 2.2 Dual Renderer

提供：

```text
render(snapshot, full_mask)
render(snapshot, minus_m_mask)
```

对同一 `snapshot_hash` 生成两个 view。

禁止：

```text
Full Harness 从头独立执行另一条 trajectory
```

## 2.3 Tool Span Parser

统一识别：

```text
tool/function name
argument keys
argument values
end_search
```

对 Harness-1 原生 tool interface 做单元测试。

---

# 3. Influence 定义的可执行版本

## 3.1 I_name

合法 tool name 是有限集合。

对每个 state，计算：

```text
P_full(tool_name | r_F(xi))
P_minus(tool_name | r_-m(xi))
```

通过 sequence logprob 对全部合法 tool name 归一化。

然后：

```text
I_name = JS(P_full || P_minus)
```

这是主 influence 指标。

## 3.2 I_args

arguments 是开放文本，不能简单枚举整个 action space。

首版采用：

```text
1. Full view greedy/teacher decode 一个 tool call a_T
2. 固定 a_T 的 argument token sequence
3. 分别在 full view / minus view 上 teacher-force
4. 对 argument-key / argument-value token 计算 token distribution divergence
```

输出：

```text
I_arg_key
I_arg_value
mean teacher-forced token KL/JS
```

同时报告行为统计：

```text
tool_name_disagreement_rate
exact_tool_call_disagreement_rate
argument_string_edit_distance
query_change_rate
doc_id_set_change_rate
stop_change_rate
```

---

# 4. Null Controls

必须有两类 null：

```text
N0: full render vs full render
N1: 只改变无语义字段顺序 / serialization ordering
```

任何 component 的 influence 必须与 null noise 对比。

如果：

```text
I_m ≈ null
```

不得标记成可迁移 policy effect。

---

# 5. 数据规模

每个 component：

```text
INF_CAL64
```

从 `-m` Harness 下让 Student 真正 rollout。

每题保存所有 decision states，但设置上限：

```text
最多 32 个 state / query
```

优先保留：

```text
search
read/review
curate
verify
end_search
```

若 64q 信号稳定，可扩到：

```text
INF_CONFIRM128
```

不要直接跑 830。

---

# 6. 8 卡并行队列

## GPU 0

```text
A0. adaptive_rerank_instruction / INF_CAL64
A1. content_dedup / INF_CAL64
```

每个 component 完成：

```text
reduced rollout
snapshot
dual render
I_name
I_args
null-normalized effect
```

---

## GPU 1

```text
B0. importance_tagging
B1. chunk_neighbors
```

---

## GPU 2

```text
C0. subtractive_curation
```

完成后：

```text
C1. 对 subtractive_curation 做 INF_CONFIRM128
```

仅当 CAL64 influence > null。

---

## GPU 3

```text
D0. auto_populate_first_search
```

完成后：

```text
D1. 首次搜索/首次 curate 附近 state 专项 influence
```

---

## GPU 4

```text
E0. evidence_graph
```

额外按 state 分层：

```text
graph empty
graph sparse
graph mature
```

看 influence 是否只在后期出现。

---

## GPU 5

```text
F0. sentence_compress
```

额外记录：

```text
full/minus rendered token count
context overlap
tool policy divergence vs token reduction
```

---

## GPU 6

```text
G0. verify_tool
```

分层：

```text
verify-eligible state
verify-ineligible state
```

若 verify event 极少：

不要人工伪造结论；
只输出 `LOW_EVENT_SUPPORT`。

可以额外做 targeted state mining，但必须单独标记，不与 natural influence 混写。

---

## GPU 7

```text
H0. token_budget_marker
```

按剩余 budget 分桶：

```text
early
middle
late
near-limit
```

检查 influence 是否只在 near-limit 才出现。

完成后负责：

```text
全局 null control
renderer parity
snapshot replay audit
```

---

# 7. Influence 输出

每个 component 输出：

```json
{
  "component": "...",
  "n_queries": 64,
  "n_states": 0,
  "event_support": 0,
  "I_name_mean": 0.0,
  "I_name_median": 0.0,
  "I_arg_key": 0.0,
  "I_arg_value": 0.0,
  "tool_name_disagreement": 0.0,
  "exact_call_disagreement": 0.0,
  "null_I_name": 0.0,
  "normalized_influence": 0.0
}
```

同时按 tool type 分层：

```text
search
read
review
curate
verify
end
```

---

# 8. 与 System Contribution 合并

如果 H100-1/2 报告已同步，生成：

```text
CONTRIBUTION_INFLUENCE_MAP.md
```

四象限：

```text
High Δ, High I
  强 migration candidate

High Δ, Low I
  runtime/state mechanism

Low Δ, High I
  behavior-changing but possibly unnecessary scaffolding

Low Δ, Low I
  direct-removal candidate
```

但**仍不运行 learnability**。

---

# 9. 特别注意

旧 SCOPE 的 Gate A “双侧事件不足”不能直接照搬。

SCAPE 中：

```text
tool call 本身就是统一 action
```

所以不需要为每个 capability 手工构造 KEEP/SKIP 双侧分类标签。

但自然事件支持仍要报告，因为：

```text
没有事件
=> influence estimate 无意义
```

这是 measurement issue，不是 capability-specific label gate。

---

# 10. 禁止事项

```text
不训练
不做 capability-specific classifier
不造 KEEP/SKIP
不造 CONTINUE/RECOVER
不使用 teacher future trajectory
不把不同 snapshot 的 full/minus prompt 做 divergence
不把 prompt token 差异本身当 policy influence
```
