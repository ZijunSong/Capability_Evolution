# SCAPE-0813-Next-H100-2.md
## H100-2（8×H100）下一轮：Candidate-B Long-Horizon Live Fork/Replay Advantage Resolution

> 当前 B-side 的 H100 结果互相冲突：
>
> - H100-2 true live K2/K4：`verify_tool` 最好；
> - H100-4 独立 K2/K4：`importance_tagging` 超过 `subtractive_curation`；
> - H20 对 SC/IT/VT 的 micro learnability 又全部 FAIL。
>
> 因此本机不要再做新的短 horizon ranking，也不要再选 Candidate B。
>
> 本轮只回答：
>
> **Full-view teacher 在发生 policy disagreement 时给出的动作，沿真实环境继续执行后，是否真的产生可重复的 downstream advantage？**

不训练模型权重。

---

# 1. Repo / Environment

```bash
cd /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE
```

Python：

```text
/opt/scape-hf-scorer/bin/python
```

若实际可用环境名称不同，使用 `/opt` 下可复现环境并写入 manifest。

输出：

```text
outputs/h100_2_long_horizon_advantage/
outputs/scape_prestage_v4/H1002_ADVANTAGE_HANDOFF.json
```

---

# 2. Components

只测：

```text
verify_tool
subtractive_curation
importance_tagging
```

不再扩新的 B 候选。

---

# 3. State Selection

冻结：

```text
ADV_LIVE256
seed = 2224
```

每个 component：

```text
256 natural candidate-bearing states
```

state 必须来自：

```text
H_-m occupancy
```

优先分层抽样：

```text
high I_name
medium I_name
low-but-positive I_name
```

不要只选 top influence，避免 selection bias。

每 state 保存：

```text
xi_t
student action a_S
full-view teacher action a_T
I_name
teacher entropy
turn
tool type
```

---

# 4. Fork/Replay Contract

从同一 `xi_t` fork：

```text
Branch S:
execute student action a_S
then hand back to the same continuation policy

Branch T:
execute teacher action a_T
then hand back to the same continuation policy
```

比较：

```text
K=4
K=8
```

额外在高价值候选上做：

```text
to terminal / max remaining budget
```

禁止：

```text
teacher 接管后续完整 trajectory
```

后续 continuation policy 必须相同。

---

# 5. Utility

优先使用真实 downstream utility：

```text
Δ curated recall
Δ trajectory recall
Δ final recall
Δ reward
Δ supported evidence
Δ runtime cost
```

如果在 K 步内 final metric 不可定义，使用：

```text
state potential proxy
```

但必须单独标记，不能覆盖真实终局结果。

HF continuation logprob 只能作为辅助解释，不是主 utility。

---

# 6. 8 卡并行

## GPU0
```text
verify_tool K=4 / 256
```

完成后：
```text
verify_tool terminal-confirm / top128
```

## GPU1
```text
verify_tool K=8 / 256
```

完成后：
```text
verify_tool independent replay-noise
```

## GPU2
```text
subtractive_curation K=4 / 256
```

完成后：
```text
subtractive terminal-confirm / top128
```

## GPU3
```text
subtractive_curation K=8 / 256
```

## GPU4
```text
importance_tagging K=4 / 256
```

完成后：
```text
importance terminal-confirm / top128
```

## GPU5
```text
importance_tagging K=8 / 256
```

## GPU6
```text
sham fork/replay K=4
same action executed on both branches
```

用于测 replay noise。

## GPU7
```text
sham fork/replay K=8
+
runtime profiler
```

---

# 7. 统计

每个 component × horizon：

```text
mean(T-S)
median(T-S)
positive_fraction
paired W/L/T
bootstrap 95% CI
```

分层：

```text
I_name quantile
teacher entropy quantile
turn quartile
tool type
```

计算：

```text
corr(I_name, downstream_advantage)
```

---

# 8. 判定

### `ADVANTAGE_POSITIVE`

要求：

```text
K4/K8 至少总体同方向
mean T-S > replay noise
paired bootstrap 不完全跨越明显负区间
terminal top128 不反转
```

### `SHORT_HORIZON_ONLY`

若：

```text
K4 正
K8/terminal 消失或变负
```

说明该 component 的 teacher action 只改变局部行为，不足以作为 migration target。

### `DOMAIN_SENSITIVE`

若：

```text
本轮与 H100-4 独立 confirm 排名仍反转
```

不要继续争 Candidate-B #1。

### `NO_VALUE_OF_INFLUENCE`

若：

```text
I 高但 downstream advantage ≈0
```

则这是非常重要的 negative result：
Influence 不是 distillation value。

---

# 9. 不允许做

```text
不训练
不再跑 UTILITY_STATE256 旧 short-horizon proxy
不使用旧 Behavior-only recommendation 冻结 B
不因为 verify influence 高就自动给 H20
```

---

# 10. 最终产物

```text
RUN_MANIFEST.json
STATUS_LIVE.md
ADVANTAGE_PER_STATE.jsonl
ADVANTAGE_BY_COMPONENT_HORIZON.csv
ADVANTAGE_BY_INFLUENCE_QUANTILE.csv
REPLAY_NOISE.md
CANDIDATE_B_VALUE_RESOLUTION.md
outputs/scape_prestage_v4/H1002_ADVANTAGE_HANDOFF.json
SHA256SUMS
```

Handoff 只能给：

```text
MIGRATION_VALUE_POSITIVE
CONDITIONAL_RUNTIME
SHORT_HORIZON_ONLY
DOMAIN_SENSITIVE
NO_VALUE_OF_INFLUENCE
```

不要直接写“Candidate B frozen”。
