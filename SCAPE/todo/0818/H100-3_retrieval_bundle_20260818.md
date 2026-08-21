# H100-3：`content_dedup + auto_populate_first_search (+ adaptive_rerank_instruction)` 组合内化

## 0. 研究问题

这条线验证第二个组合假设：

> `auto_populate_first_search` 负责“第一次搜索后立即建立精选集”，`content_dedup` 负责“不要让近重复文档污染 evidence pool”；二者可能共同形成一种可内化的 retrieval hygiene 行为。`adaptive_rerank_instruction` 不一定单组件改变下一 route，但可能通过提高候选文档质量增强这个组合。

重点不是“多组件一定更强”，而是测试组件之间是否存在 **functional complementarity**。

内部实验名：

```text
RETRIEVAL_HYGIENE_BUNDLE
```

---

## 1. 强制规则

1. 8×H100。
2. actual LoRA / actual LLM only。
3. no-privilege Student real closed-loop。
4. 禁止 route-head 代替。
5. 禁止只看 route JS/agreement。
6. 主成功标准：Student_after > Student_before，再比较 baseline。
7. Full Harness 只作为参考 upper/reference，不要求 Student 超过。
8. 如果组合无增益：
   - 查机制；
   - redesign 一次；
   - 再失败就 discard 组合，不缩 claim。
9. 所有数据必须来自真实 runtime、真实 doc ids、真实 MinHash/dedup logic。

输出：

```text
outputs/0818_retrieval_hygiene_bundle/
```

---

## 2. 第一阶段：代码与 case 审核

先回答四个问题。

### 2.1 AUTO

复用 H100-1 audit 思路，确认第一次成功 search 后 AUTO 实际加入哪些 ids。

### 2.2 content_dedup

定位真实实现：

- MinHash threshold；
- 在 pool 哪一层丢弃；
- duplicate cluster / canonical representative 如何确定；
- Student 是否能看见被丢弃文档；
- duplicate side-effect 是否发生在模型下一决策前。

抽至少 50 个真实 dedup-trigger cases，输出：

```text
duplicate_id
canonical_id
similarity/hash evidence
Student history
后续是否重复 read/curate
```

### 2.3 adaptive_rerank_instruction

确认它到底改变：

- search query？
- retriever score？
- observation ordering？
- injected instruction text？
- top-K documents？

必须计算至少：

```text
same query 下 rerank ON/OFF 的 top-K overlap
qrel recall@K delta
AUTO 最终 curated ids delta
```

如果它几乎不改变 document candidates，只改变文本提示，则不要硬塞进组合。

### 2.4 旧 AUTO failure cases

复用/读取 H100-1 的 case artifact（若已产生），验证：
- 失败是否主要来自“Student 没有 search 后 curate”；
- duplicate 污染是否是第二大错误来源。

---

## 3. 第二阶段：定义可执行 projection

### 3.1 AUTO projection

同 H100-1：

```text
post-search harness-added curated ids
    ↓
curate(add_ids=..., remove_ids=[])
```

### 3.2 DEDUP projection

不要尝试让 Student“删除 environment pool 中已经被 harness 静默删除的对象”，除非 Student action space 真能做到。

应将 dedup side-effect 投影为 **Student 可实现的后续决策约束**：

当 duplicate cluster `{d1, d2, ...}` 中 canonical=`d*`：

- 若 Student 准备 `read_document(duplicate)`，target 变为 `read_document(canonical)`；
- 若 Student 准备 `curate(add_ids=[duplicate])`，target 改为 `curate(add_ids=[canonical])`；
- 若 canonical 已读/已精选，则 target 应避免再次 read/curate，并选择当前 teacher/native next action。

所有 projection 必须来自真实 cluster mapping，不能由 LLM 主观判 duplicate。

记录：

```text
DEDUP_PROJECTION_TYPE =
  READ_CANONICAL
  CURATE_CANONICAL
  SKIP_REDUNDANT
```

### 3.3 RERANK

`adaptive_rerank_instruction` 不直接投影文本 instruction。

它只能通过“改变了哪些 documents 成为高价值候选”进入 target。

即：

```text
rerank instruction
 -> different candidate ordering/content
 -> harness最终真正保留/读取的 document effect
 -> projected native student action
```

这样避免把贡献退化成 textual prompt distillation。

---

## 4. 第三阶段：组合 value / mechanism gate

在 actual LoRA 前先跑 real fork/replay。

至少比较：

```text
A. AUTO only
B. DEDUP only
C. AUTO + DEDUP
D. AUTO + DEDUP + RERANK
```

对相同 query/state 尽量 matched。

主要看：

```text
K4/K8 downstream reward
evidence recall
duplicate read rate
duplicate curate rate
unique relevant evidence count
tool cost
```

### 判断

- 若 `C <= max(A,B)`：没有组合增益，不能声称 complementarity。
- 若 `D <= C`：discard rerank from main bundle。
- 若 C 明显 > A/B：进入 LoRA。
- 若 dedup 本身几乎不触发，先重做 event-conditioned sampling，不要用大量 inactive states 稀释结果。

---

## 5. 第四阶段：8 GPU actual-LoRA 矩阵

根据 gate 结果运行。

默认矩阵：

| GPU | Variant | Seed |
|---|---|---:|
| 0 | `AUTO_PROJECTED` | 42 |
| 1 | `AUTO_PROJECTED` | 43 |
| 2 | `AUTO_DEDUP_PROJECTED` | 42 |
| 3 | `AUTO_DEDUP_PROJECTED` | 43 |
| 4 | `AUTO_DEDUP_RERANK_PROJECTED` | 42 |
| 5 | `AUTO_DEDUP_RERANK_PROJECTED` | 43 |
| 6 | `SHUFFLED_BUNDLE_PROJECTION` | 42 |
| 7 | `SHUFFLED_BUNDLE_PROJECTION` | 43 |

如果第三阶段已经证明 rerank 无效，则 GPU4/5 改为：

```text
DEDUP_PROJECTED seed42/43
```

### 训练 target

必须覆盖真实 executable tool-call args：

```text
curate.add_ids/remove_ids
read_document.doc_id
必要时 search query span
```

不要把主要 supervision 再退回 8-way route name。

---

## 6. 第五阶段：real closed-loop

从 initial state 跑：

```text
BASE_STUDENT
AUTO_PROJECTED
DEDUP_PROJECTED (若训练)
AUTO_DEDUP_PROJECTED
AUTO_DEDUP_RERANK_PROJECTED (若 gate 保留)
SHUFFLED_BUNDLE
FULL_HARNESS_REFERENCE
```

主指标除 reward/recall 外，必须增加：

```text
first_search_to_first_curate_turns
first_search_immediate_curate_rate
duplicate_read_rate
duplicate_curate_rate
unique_docs_read
unique_relevant_docs_read
curated_unique_relevant_docs
search_redundancy
qrel recall@curated
```

### GO

核心不是“bundle 看起来有点好”，而是：

```text
1. AUTO_DEDUP > BASE
2. AUTO_DEDUP > AUTO only
3. AUTO_DEDUP > shuffled
4. >=2 seeds same direction
5. pooled paired bootstrap CI > 0
6. mechanism metric 同时改善：
   - 更早 curate
   - 更少 duplicate
   - 更多 unique relevant evidence
```

Rerank 只有在：

```text
AUTO+DEDUP+RERANK > AUTO+DEDUP
```

且行为上能解释为候选质量提升时才留在主方法。

---

## 7. 如果失败，允许的一次 redesign

先做 60 个 case：

```text
AUTO succeeds / bundle fails
bundle succeeds / AUTO fails
duplicate trigger but Student still repeats
rerank changes top-K but hurts evidence diversity
early curate locks in bad top-K
dedup removes apparently duplicate but actually complementary chunks
```

允许的 substantive redesign：

- AUTO 从固定 top-K projection 改为 relevance-filtered projection；
- dedup 从 binary canonical 替换为 cluster-aware representative selection；
- 对“近重复但互补”建立 conservative threshold；
- rerank 仅在 query/domain gate 下启用；
- 对 projection 加 outcome/value weighting。

不允许只调 LR/seed 后宣布 redesign。

---

## 8. 与 `chunk_neighbors` 的关系

本轮不要一起蒸馏 `chunk_neighbors`。

理由：

- 它提供的是额外相邻内容，具有更强实时外部信息属性；
- Student 是否能主动补邻居需要额外 read/search 行为设计；
- 会让当前 retrieval hygiene bundle 的 causal attribution 变差。

把它保留为 runtime component 或下一篇 placement work。

---

## 9. Artifacts

```text
RUN_MANIFEST.json
STATUS_LIVE.md

CONTENT_DEDUP_CODE_AUDIT.md
DEDUP_TRIGGER_CASES.jsonl
RERANK_EFFECT_AUDIT.csv
RETRIEVAL_BUNDLE_SCHEMA.md

BUNDLE_K4_K8_RESULTS.csv
BUNDLE_VALUE_GATE.json

AUTO_PROJECTED_DATA.jsonl
DEDUP_PROJECTED_DATA.jsonl
AUTO_DEDUP_PROJECTED_DATA.jsonl
AUTO_DEDUP_RERANK_PROJECTED_DATA.jsonl

TRAINING_CELLS.csv
DEV_REAL_CLOSED_LOOP.csv
TEST_REAL_CLOSED_LOOP.csv
PAIRED_BOOTSTRAP.csv
RETRIEVAL_MECHANISM_METRICS.csv
RETRIEVAL_CASE_ANALYSIS.md

H1003_0818_HANDOFF.json
SHA256SUMS
```

最终结论限定为：

```text
GO_RETRIEVAL_BUNDLE
GO_AUTO_DEDUP_DISCARD_RERANK
REDESIGN_ONCE_RETRIEVAL_BUNDLE
DISCARD_RETRIEVAL_BUNDLE
```
