# SCAPE-0813-Next-H100-3.md
## H100-3（8×H100）下一轮：Advantage-Conditioned Influence Probe

> 现有 Pre-stage 的失败点已经很清楚：
>
> `Contribution + Influence (+ short utility)` 没能预测 H20 learnability。
>
> 下一步不要再把 `I_m` 当“值得蒸馏”的充分条件。
>
> 本机要回答更细的问题：
>
> **在 influence 高的 same-state 上，Full-view teacher 与 Reduced-view student 的动作分歧，是否对应真实正 advantage？**
>
> 如果没有，那么“Influence 高”只能说明 Harness 改变了行为，不能说明这种行为值得迁移。

本机不训练。

---

# 1. Repo / Environment

```bash
cd /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE
```

使用：

```text
/opt/scape-hf-scorer
```

输出：

```text
outputs/h100_3_advantage_conditioned_influence/
outputs/scape_prestage_v4/H1003_VALUE_OF_INFLUENCE_HANDOFF.json
```

---

# 2. 输入

优先复用：

```text
outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl
```

7 components：

```text
evidence_graph
verify_tool
importance_tagging
subtractive_curation
content_dedup
chunk_neighbors
auto_populate_first_search
```

不要重新生成 influence，除非 snapshot 缺少 fork/replay 必需字段。

---

# 3. State Stratification

每个 component 选：

```text
128 states
```

按 `I_name_normalized` 四分位：

```text
Q1 32
Q2 32
Q3 32
Q4 32
```

若 component 如 chunk_neighbors 大量 I=0：

```text
保留真实 0-signal 分布
不要人为只挑 positive states
```

额外记录：

```text
turn quartile
tool type
argument class
teacher entropy
name disagreement
```

---

# 4. One-Step Corrective Fork

对同一 state：

```text
S branch = execute student selected tool call
T branch = execute full-view teacher selected tool call
```

然后：

```text
同一个 continuation policy
继续 K=4 步
```

主实验只用 K=4，以便覆盖 7 components。

指标：

```text
Δ reward
Δ curated recall
Δ trajectory recall
Δ evidence coverage
Δ state potential
Δ calls / turns
```

---

# 5. 8 卡并行

## GPU0
```text
evidence_graph
128 states × K4
```

## GPU1
```text
verify_tool
128 × K4
```

## GPU2
```text
importance_tagging
128 × K4
```

## GPU3
```text
subtractive_curation
128 × K4
```

## GPU4
```text
content_dedup
128 × K4
```

注意：它是 runtime control，实验目的是检测 policy influence 是否有 value，不是把 dedup 升级为 migration candidate。

## GPU5
```text
chunk_neighbors
128 × K4
```

## GPU6
```text
auto_populate_first_search
128 × K4
```

## GPU7
```text
sham action fork
+
same-action replay noise
+
全局 aggregation
```

---

# 6. 定义新的解释性统计

不要直接修改论文主方法，但先产生诊断量：

```text
A_m =
E[ U(T-branch) - U(S-branch) ]
```

以及：

```text
VAI_m = E[ I_name(state) * sign/normalized_advantage(state) ]
```

`VAI` 只作为 diagnostic 名称，不先写进 method。

更重要的是计算：

```text
Spearman(I_name, advantage)
```

和分桶：

```text
Q1/Q2/Q3/Q4 influence
vs
mean advantage
```

---

# 7. 关键问题

必须回答：

1. `evidence_graph` 的高 influence state 是否也有正 advantage？
2. `verify_tool` 的高 influence 是否只发生在低频/late-turn state？
3. `importance_tagging` 是否出现“改变很多，但没有 downstream gain”？
4. `subtractive_curation` 的 late-turn tool-name influence 是否有终局价值？
5. `content_dedup` 的高 I_name / 负 I_args 是否只是 renderer/runtime artifact？
6. `auto_populate` 的正 name influence + 大负 argument signal 是否对应错误/不稳定 action？
7. `chunk_neighbors` 的 zero influence 是否也意味着 zero policy-side advantage？

---

# 8. H20 handoff 分类

对每个 component 输出：

```text
HIGH_I_HIGH_A
HIGH_I_LOW_A
LOW_I_HIGH_A
LOW_I_LOW_A
RUNTIME_CONTROL
```

### 最重要的解释

```text
HIGH_I_LOW_A
```

意味着：

> Harness 的确改变了当前 policy，但这种改变未带来 downstream improvement，因而不值得被当作 OPD target。

这很可能解释为什么现有 C+I probe 无法预测 learnability / retirement。

---

# 9. 与 H100-2 的关系

H100-2：

```text
对 SC/IT/VT 做更深 K4/K8/terminal utility resolution
```

H100-3：

```text
对 7 components 做统一 K4 value-of-influence map
```

两者不要重复生成相同 split。

H100-3 负责 breadth；
H100-2 负责 B-side depth。

---

# 10. Stop Rule

本机不启动：

```text
training
8K
Stage S
candidate freeze
```

即使 `evidence_graph` 显著正，也只 handoff 给 H20 的 Hybrid line。

---

# 11. 最终产物

```text
RUN_MANIFEST.json
STATUS_LIVE.md
VALUE_OF_INFLUENCE_PER_STATE.jsonl
VALUE_OF_INFLUENCE_BY_COMPONENT.csv
VALUE_OF_INFLUENCE_BY_QUANTILE.csv
INFLUENCE_ADVANTAGE_CORRELATION.md
PRESTAGE_FAILURE_EXPLANATION.md
outputs/scape_prestage_v4/H1003_VALUE_OF_INFLUENCE_HANDOFF.json
SHA256SUMS
```
