# SCAPE H100-2 指令：Independent Replication + Coalition Interaction

## 任务定位

本机承担两个任务：

```text
1. 在独立 split / 独立官方 benchmark 上复现 component contribution
2. 测少量结构上合理的 component coalition interaction
```

**不训练。**

本机的价值不是再做一遍 H100-1，而是回答：

```text
component placement 是否 manifest-sensitive？
是否 domain-sensitive？
单模块 LOO 是否被 interaction 欺骗？
```

---

# 1. 工作目录

```text
/mnt/songzijun/Capability_Evolution/SCAPE
```

分支：

```text
exp/h1002-replication-coalition
```

输出：

```text
outputs/h100_2_replication_coalition/
```

最终必须有：

```text
RUN_MANIFEST.json
STATUS_LIVE.md
SECONDARY_BENCHMARK_SELECTION.md
LOO_REPLICATION.csv
COALITION_INTERACTION.csv
REPLICATION_REPORT.md
SHA256SUMS
```

---

# 2. Benchmark 选择规则

优先选择 **Harness-1 upstream 已官方支持、当前机器本地资源可完整运行** 的非 BrowseComp+ benchmark。

优先级：

```text
multi-hop QA
> patents
> finance
> 其他 upstream official retrieval benchmark
```

若机器只具备 BrowseComp+ backend：

不要自造新 benchmark，改用：

```text
BCP_REPL200
```

要求与 H100-1 CAL200 query-disjoint，stable hash seed=2202。

必须在 `SECONDARY_BENCHMARK_SELECTION.md` 记录：

```text
为什么选择该 benchmark
upstream runner
dataset version
retrieval backend
scorer
是否 source-domain / transfer-domain
```

---

# 3. 固定原则

条件间保持完全一致：

```text
model checkpoint
retrieval backend
index
query manifest
temperature
max turns
max context
tool budget
reranker
scorer
```

只改变 component mask。

主指标优先：

```text
curated recall
trajectory recall
final-answer recall
benchmark-native retrieval metric
runtime cost
```

---

# 4. 8 卡并行队列

## GPU 0

```text
A0. Full Harness / REPL200
A1. semantic-light Harness / REPL200
```

用于建立 replication baseline 与 broad removal gap。

---

## GPU 1

```text
B0. - adaptive_rerank_instruction / REPL200
B1. coalition:
    - adaptive_rerank_instruction
    - token_budget_marker
```

解释目标：

`rerank × budget-aware search` 是否存在 interaction。

---

## GPU 2

```text
C0. - importance_tagging / REPL200
C1. coalition:
    - importance_tagging
    - subtractive_curation
```

这是最重要的结构 coalition 之一：

```text
tag evidence
+
use tag to decide what to retain/remove
```

---

## GPU 3

```text
D0. - subtractive_curation / REPL200
D1. coalition:
    - subtractive_curation
    - evidence_graph
```

---

## GPU 4

```text
E0. - auto_populate_first_search / REPL200
E1. coalition:
    - auto_populate_first_search
    - adaptive_rerank_instruction
```

---

## GPU 5

```text
F0. - evidence_graph / REPL200
F1. coalition:
    - evidence_graph
    - sentence_compress
```

---

## GPU 6

```text
G0. - verify_tool / REPL200
G1. coalition:
    - verify_tool
    - evidence_graph
```

额外记录：

```text
verify trigger count
verify success
verification records
claim/evidence links
```

---

## GPU 7

顺序：

```text
H0. - token_budget_marker / REPL200
H1. - sentence_compress / REPL200
H2. - content_dedup / REPL200
H3. - chunk_neighbors / REPL200
```

如果单卡总任务过长：

优先级：

```text
token_budget_marker
sentence_compress
content_dedup
chunk_neighbors
```

剩余条件可由最先空闲 GPU work-steal，但必须保持输出目录名与 manifest 不变。

---

# 5. Coalition 统计

对于 pair `(m1,m2)`：

定义：

```text
Delta1 = J(full) - J(-m1)
Delta2 = J(full) - J(-m2)
Delta12 = J(full) - J(-m1,-m2)

interaction = Delta12 - Delta1 - Delta2
```

paired bootstrap CI。

解释：

```text
interaction ~ 0:
近似可加

interaction > 0:
synergy / joint dependency

interaction < 0:
redundancy / overlap
```

不要根据 200q 小样本写因果大结论；用于决定 H20 是否需要 coalition dropout。

---

# 6. Replication 判定

对每个 component 生成：

```text
H100-1 effect
H100-2 effect
same sign?
same rank region?
cross-domain sensitive?
```

若 H100-1 输出尚未同步：

先完成本机独立结果，不等待。

之后再运行：

```text
scripts/merge_crossnode_contribution.py
```

---

# 7. 重点输出：Placement Stability

最终生成：

```text
PLACEMENT_STABILITY.md
```

至少分四类：

```text
Stable-Positive
  两个 split/benchmark 都有正 contribution

Domain-Sensitive
  一个正一个弱/负

Interaction-Dependent
  单 LOO 弱，但 coalition 明显

Runtime-Like
  quality influence 小，但 cost/state behavior 明显
```

---

# 8. Stop Rule

本机不允许因为某个 component 很有希望而启动训练。

即使出现：

```text
large Δ
```

也只输出：

```text
CONTRIBUTION_REPLICATED=true
```

是否进入训练由：

```text
Contribution + Influence + Learnability
```

共同决定。

---

# 9. 禁止事项

```text
不跑 rollback
不跑旧 SCOPE retirement gate
不做 prompt tournament
不扩旧 finalization
不做 full 830/7405 大规模验证，除非 200q 结果用于最终 paper table 且 H20 已确认
不修改模型参数
```
