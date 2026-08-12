# SCAPE H100-1 指令：Harness-1 Reproduction + Component Contribution Map

## 任务定位

本机只负责：

```text
Phase 0: Harness-1 reproduction
Phase 1: BrowseComp+ calibration LOO contribution map
```

**不训练。**
**不改旧 SCOPE。**
**不继续 query-selection/finalization tournament。**

目标是得到 10 个 Harness-1 component 的第一张：

```text
System Contribution Δ_m
+
runtime cost ΔC_m
```

---

# 1. 工作目录与分支

优先：

```bash
cd /mnt/songzijun/Capability_Evolution
mkdir -p SCAPE
cd SCAPE
```

若 canonical SCAPE 已存在：

```bash
git status
git branch --show-current
git rev-parse HEAD
```

切分支：

```text
exp/h1001-contribution-map
```

若仓库不存在，不要从 SCOPE copy 整仓；只 clone / sync canonical SCAPE 或 upstream Harness-1，并记录 provenance。

输出根：

```text
outputs/h100_1_contribution/
```

必须生成：

```text
RUN_MANIFEST.json
STATUS_LIVE.md
BASELINE_REPRODUCTION.md
COMPONENT_CONTRIBUTION.csv
COMPONENT_CONTRIBUTION.json
COMPONENT_CONTRIBUTION.md
SHA256SUMS
```

---

# 2. Preflight：先检查，不要盲跑

检查：

```text
1. 8×H100 全部可见
2. 每卡空闲显存
3. Python >= 3.11
4. uv
5. vLLM GPT-OSS support
6. pat-jj/harness-1 checkpoint 或本地 cache
7. Harness-1 official code commit
8. BrowseComp+ manifest
9. compatible Chroma retrieval backend
10. scorer 可运行
```

做：

```text
1 query smoke
5 query smoke
20 query smoke
```

要求：

```text
errors = 0
tool calls 可解析
curated state 非空
trajectory 可持久化
component flag 确实改变 runtime
```

若缺大型 retrieval backend：

```text
不要替换成旧 SCOPE BM25
不要伪造 BrowseComp+ result
```

写：

```text
BLOCKED_RETRIEVAL_BACKEND.md
```

然后只完成 instrumentation/unit tests，结束 GPU rollout。

---

# 3. 数据冻结

从官方可运行 BrowseComp+ query 集构造：

```text
BCP_CAL200
BCP_HOLD200
BCP_SMOKE20
```

要求：

```text
stable hash split
query-disjoint
seed = 1101
manifest frozen
```

`BCP_HOLD200` 在本机 Phase 1 不用于调参，只用于 full baseline sanity / 最后 quick confirm。

主 decode：

```text
temperature = 0
do_sample = false
top_p = 1
max context / turns = upstream published setting
```

禁止为了让某个 ablation 好看而条件间改 budget。

---

# 4. 统一 config

创建：

```text
configs/harness/full.yaml
configs/harness/minus_<component>.yaml
```

10 个 component：

```text
adaptive_rerank_instruction
importance_tagging
subtractive_curation
auto_populate_first_search
evidence_graph
sentence_compress
content_dedup
chunk_neighbors
verify_tool
token_budget_marker
```

所有 `minus_m` 必须只改一个 flag。

运行前自动 diff：

```text
full config vs minus_m
```

如果 diff 超过目标 component，FAIL FAST。

---

# 5. 8 卡并行队列

## GPU 0

顺序执行：

```text
A0. Full Harness / BCP_CAL200
A1. Full Harness / BCP_HOLD200
A2. Full Harness / BCP_SMOKE20 replay parity check
```

目的：

```text
建立主 baseline
验证 scorer
验证 deterministic replay
提供 paired baseline
```

---

## GPU 1

```text
B0. - adaptive_rerank_instruction / CAL200
B1. - importance_tagging / CAL200
```

每个 run 后立即生成：

```text
delta curated recall
delta trajectory recall
delta final-answer recall
delta reward
delta turns
delta tool calls
delta rendered tokens
delta latency
paired W/L/T
```

---

## GPU 2

```text
C0. - subtractive_curation / CAL200
C1. - auto_populate_first_search / CAL200
```

额外统计：

```text
curated-set turnover
add/remove count
first-search -> first-curate latency
```

---

## GPU 3

```text
D0. - evidence_graph / CAL200
D1. - sentence_compress / CAL200
```

额外统计：

```text
evidence link count
rendered context tokens
pool/curated size
```

---

## GPU 4

```text
E0. - content_dedup / CAL200
E1. - chunk_neighbors / CAL200
```

额外统计：

```text
duplicate observation rate
repeated doc/read rate
retrieval expansion count
```

注意：

`content_dedup` 很可能是 runtime anchor；即使 removal 产生大 gap，也不要自动列入蒸馏 candidate。

---

## GPU 5

```text
F0. - verify_tool / CAL200
F1. - token_budget_marker / CAL200
```

额外统计：

```text
verify call count
verified record count
premature end
max-turn rate
budget utilization
```

---

## GPU 6

顺序执行两个系统级参照：

```text
G0. "semantic-light" Harness / CAL200
    一次关闭：
    adaptive_rerank_instruction
    importance_tagging
    subtractive_curation
    auto_populate_first_search

G1. "runtime-anchor-only" Harness / CAL200
    保留 executor/store/accounting/retrieval，
    关闭所有可以安全关闭的 cognitive/rendering enhancement
```

目的：

估计：

```text
single-component gap 是否可加
多组件一起移除时是否发生崩塌
SCAPE 最终 H_slim 的粗下界
```

这两组不进入单组件 Δ_m 排名。

---

## GPU 7

```text
H0. upstream/original Harness-1 ablation smoke reproduction
H1. Full Harness / CAL200 second independent process replay
H2. 运行 latency/token/state-operation profiling
```

若 upstream 论文或 repo 已提供 component ablation runner：

优先复用官方 runner 逻辑，与 SCAPE wrapper 输出做 parity。

---

# 6. 聚合

等待所有 GPU 当前队列结束后运行 CPU 聚合：

```text
scripts/aggregate_contribution.py
```

对每个 component 输出：

```json
{
  "component": "...",
  "n": 200,
  "delta_curated_recall": 0.0,
  "delta_trajectory_recall": 0.0,
  "delta_final_recall": 0.0,
  "delta_reward": 0.0,
  "paired_wlt": [0,0,0],
  "bootstrap_ci95": [0.0,0.0],
  "delta_tool_calls": 0.0,
  "delta_context_tokens": 0.0,
  "delta_latency": 0.0
}
```

生成 ranking：

```text
Contribution-High
Contribution-Neutral
Contribution-Negative
Runtime-Costly
```

不要在本机根据 Δ 直接宣布“可蒸馏”。

---

# 7. 自动 Gate

### Reproduction gate

只有 baseline：

```text
scorer 正常
retrieval 正常
trajectory 正常
component flag intervention 可验证
```

才继续 200q LOO。

### LOO candidate signal

标记 `CONTRIBUTION_SIGNAL=true` 需要：

```text
至少一个 Harness-1 原生核心 retrieval metric:
full > minus_m
且 paired effect 不是单个 outlier 驱动
```

若 canonical answer metric 为 0 floor：

```text
记录 floor
但不得因此把所有 module 判为 neutral
```

主排序使用：

```text
curated recall
trajectory recall
final recall
quality-cost utility
```

---

# 8. 禁止事项

```text
禁止训练
禁止修改模型权重
禁止复用 SCOPE duplicate_evidence classifier
禁止跑 rollback
禁止用旧 P_m
禁止把旧 BrowseComp BM25 当 Harness-1 backend
禁止条件间改变 turn/token/search budget
禁止只汇报最好的一项 metric
```

---

# 9. 最终报告必须回答

1. Harness-1 baseline 是否在本环境成功复现？
2. 哪些 component 删除后真正损伤 retrieval quality？
3. 哪些 component 主要改变 runtime cost，而非 quality？
4. 哪些 component 几乎可以直接删除？
5. 是否出现明显 component interaction 的迹象？
6. 给 H20 推荐的 top-4 **候选**是什么？这里只能是候选，不能越过 Influence/Learnability 直接宣布迁移。
