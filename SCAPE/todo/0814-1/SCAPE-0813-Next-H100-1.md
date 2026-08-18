# SCAPE-0813-Next-H100-1.md
## H100-1（8×H100）下一轮：Evidence-Graph Renderer Dose–Response 与 Runtime Pareto

> 当前 Evidence Graph 已不适合继续问“整个 graph 能不能删”。
>
> 已有结果说明：
>
> - graph state 本身有独立 utility；
> - `STATE + MINIMAL_RENDER` 已接近 `FULL_RENDER`；
> - `STATE_ONLY -> MINIMAL_RENDER` 的 policy influence 大于 `MINIMAL_RENDER -> FULL_RENDER`；
> - 因而下一步应寻找 **最小必要 renderer/controller 容量**，给 H20 定义真正的 hybrid migration target。

本机不训练。

---

# 1. Repo / 环境

```bash
cd /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE
git status
git rev-parse HEAD
```

GPU-heavy 环境：

```text
必须位于 /opt
禁止用 /mnt JuiceFS conda/venv 跑 torch/vLLM
```

输出：

```text
outputs/h100_1_graph_renderer_dose/
outputs/scape_prestage_v4/H1001_GRAPH_RENDER_HANDOFF.json
```

官方 Chroma 凭证若仍缺失：

```text
LOCAL_COMPAT_ONLY=true
official_chroma_parity=false
```

只检查一次，不轮询。

---

# 2. 核心问题

固定 external graph state，不改 graph store/update/execution。

只改变给 policy 的 graph renderer 容量：

```text
R0 = GRAPH_OFF
R1 = GRAPH_STATE_ONLY
R2 = STATE + 24-token minimal render
R3 = STATE + 48-token minimal render
R4 = STATE + 72-token minimal render
R5 = STATE + 108-token render
R6 = STATE + 144-token render
R7 = STATE + FULL_RENDER (~upstream current)
```

如果 upstream full render 不是约 180 token，不要硬裁成 180；以真实平均 token 量为准，并在 manifest 记录。

Renderer 的压缩必须是 **deterministic budgeted serialization**，禁止为每个 budget 重新调用 LLM summary。

---

# 3. 数据

冻结新的 query-disjoint：

```text
GRAPH_RENDER_DOSE400
seed = 1124
n = 400
```

必须与：

```text
BCP_GRAPH_DECOMP200
GRAPH_HYBRID_INF128
BCP_CONFIRM400
```

尽量 query-disjoint；若数据总量不足，至少与最后两个 influence/decomp split disjoint，并记录 overlap。

---

# 4. 8 卡并行

## GPU0
```text
R0 GRAPH_OFF
```

## GPU1
```text
R1 GRAPH_STATE_ONLY
```

## GPU2
```text
R2 STATE + render24
```

## GPU3
```text
R3 STATE + render48
```

## GPU4
```text
R4 STATE + render72
```

## GPU5
```text
R5 STATE + render108
```

## GPU6
```text
R6 STATE + render144
```

## GPU7
```text
R7 STATE + FULL_RENDER
```

每张卡先：

```text
smoke8
```

再：

```text
full 400
```

---

# 5. 每个条件必须统计

Quality：

```text
curated recall
trajectory recall
final recall
reward
benchmark-native metric
```

Behavior：

```text
turns
tool calls
curate count
verify count
end-search position
pool size
curated size
```

Runtime：

```text
renderer tokens / turn
total renderer tokens / episode
graph state ops
serialization latency
model input tokens
wall-clock latency
```

不要只比较 reward。

---

# 6. Pairwise dose-response

聚合时计算相邻差分：

```text
R1 -> R2
R2 -> R3
R3 -> R4
R4 -> R5
R5 -> R6
R6 -> R7
```

每个差分输出：

```text
Δquality
Δreward
Δrender_tokens
Δlatency
paired W/L/T
paired bootstrap 95% CI
```

定义：

```text
marginal_quality_per_100_tokens
```

寻找 renderer knee。

---

# 7. 第二阶段：Same-State Adjacent Influence

第一阶段结束后，不重新 rollout future teacher trajectory。

从每个相邻 pair 抽：

```text
128 queries
max 8 states/query
```

测：

```text
JS_name
teacher greedy tool disagreement
argument CE delta
```

8 卡顺序队列：

## GPU0
```text
R1 vs R2
```

## GPU1
```text
R2 vs R3
```

## GPU2
```text
R3 vs R4
```

## GPU3
```text
R4 vs R5
```

## GPU4
```text
R5 vs R6
```

## GPU5
```text
R6 vs R7
```

## GPU6
```text
R1 vs R7 端点对照
```

## GPU7
```text
identity + field-order null
```

---

# 8. 给 H20 的 target 选择规则

输出一个唯一推荐：

```text
student_runtime = R_k
teacher_view = R_{k+1} or R7
```

优先选择：

```text
1. R_k 的 runtime 明显更轻；
2. R_{k+1} 的 quality 已接近 R7；
3. R_k -> R_{k+1} 的 policy influence 明显高于 null；
4. 两者差异主要是 renderer/controller，而不是 graph state 是否存在。
```

优先期望：

```text
R1 STATE_ONLY
<- R4/R3 MINIMAL_RENDER teacher
```

但不要写死；由 dose curve 决定。

---

# 9. Runtime Pareto

生成：

```text
GRAPH_RENDER_PARETO.csv
GRAPH_RENDER_PARETO.md
GRAPH_RENDER_PARETO.png  # 可选
```

Pareto x-axis：

```text
renderer tokens / latency
```

y-axis：

```text
quality / retrieval reward
```

标出：

```text
state-only
knee
full
```

---

# 10. Stop Rule

如果：

```text
某个 minimal renderer 已在未训练情况下与 full 非劣
且成本显著更低
```

则标记：

```text
DIRECT_RUNTIME_COMPRESSION_CANDIDATE=true
```

这不是失败，而是说明这部分根本无需迁入 weights。

如果：

```text
state-only 明显掉点
minimal render 恢复
```

则标记：

```text
HYBRID_MIGRATION_TARGET_CONFIRMED=true
```

handoff 给 H20。

---

# 11. 最终产物

```text
RUN_MANIFEST.json
STATUS_LIVE.md
GRAPH_RENDER_DOSE_PER_QUERY.jsonl
GRAPH_RENDER_DOSE_SUMMARY.csv
GRAPH_RENDER_ADJACENT_INFLUENCE.csv
GRAPH_RENDER_PARETO.md
GRAPH_HYBRID_TARGET.md
outputs/scape_prestage_v4/H1001_GRAPH_RENDER_HANDOFF.json
SHA256SUMS
```
