# SCAPE-0813-Next-五机协调.md
## 基于 2026-08-13 最新结果的下一轮总调度

> 机器：
>
> - H20：8×H20，长期主训练
> - H100-1：8×H100
> - H100-2：8×H100
> - H100-3：8×H100
> - H100-4：4×H100
>
> 这一轮的核心判断是：
>
> **不要再重复 Candidate 扫描。当前最大不确定性已经从“选哪个组件”变成了“Learnability metric 是否有效、迁移目标是否定义得太宽、Influence 是否真的代表值得蒸馏的动作”。**

---

# 1. 当前证据状态

已经完成：

```text
10-component contribution
independent replication
real-model same-state influence
verify follow-up confirm
B-side utility confirm
evidence_graph full-component Stage L
SC/IT/VT micro tournament
```

当前结果：

```text
evidence_graph:
  Contribution +
  Influence +
  full-component Learnability FAIL

SC / IT / VT:
  micro Learnability all FAIL

B-side H100 utility:
  split/horizon dependent
  ranking unstable

evidence_graph decomposition:
  external graph state 应保留
  minimal renderer/controller 是更窄的迁移边界
```

因此原：

```text
Contribution -> Influence -> Learnability -> full component removal
```

必须暂时调整为：

```text
Metric Validity
    ↓
Value of Influence
    ↓
Hybrid Placement Target
    ↓
Hybrid Learnability
    ↓
Closed-loop Stage S
```

---

# 2. 五机职责

|机器|GPU|本轮唯一主任务|
|---|---:|---|
|H20|8×H20|Gate-L metric audit → 现有 ckpt 重评 → Graph-Hybrid micro → 条件触发 Stage S / clean setting|
|H100-1|8×H100|Evidence-Graph renderer dose-response + runtime Pareto，确定最小 hybrid target|
|H100-2|8×H100|SC/IT/VT long-horizon live fork/replay advantage，解决 B-side utility 冲突|
|H100-3|8×H100|7-component Advantage-Conditioned Influence，解释“高 Influence 是否有 value”|
|H100-4|4×H100|独立重算 Learnability metric，cross-node 验证 H20 Gate-L|

---

# 3. 跨机依赖

并行启动：

```text
H20 Phase A metric audit
H100-1 renderer dose-response
H100-2 long-horizon advantage
H100-3 value-of-influence
H100-4 independent metric audit
```

不需要互相等待。

第一道 barrier：

```text
H20 Metric V2
+
H100-4 Independent Audit
```

只有二者一致后，才允许任何新的 H20 训练。

第二道 barrier：

```text
H100-1 Graph Renderer Handoff
```

决定 H20 Graph-Hybrid 的：

```text
student runtime
teacher view
renderer budget
```

H100-2/H100-3 只改变 Pre-stage 解释与后续 candidate priority，**不能覆盖 H20 learnability gate**。

---

# 4. 最重要的 Go/No-Go

## Go 1：旧 Gate 有 bug

如果 H20/H100-4 发现：

```text
旧 d_* 不是 KL/JS
或 improvement 方向写反
```

则：

```text
先重评已有 checkpoint
不要重训
```

若某已有 setting 真 PASS：

```text
直接 Stage S four-grid
```

---

## Go 2：旧 Gate 真 FAIL，但 Hybrid target 更窄

则：

```text
Graph State Only student
<- Minimal Renderer teacher
```

先 512/2K。

---

## Go 3：Published Harness-1 checkpoint 仍不可学

才进入：

```text
openai/gpt-oss-20b
+ Harness-1 public SFT
```

clean mechanism setting。

---

## No-Go

以下全部停止：

```text
Evidence Graph full-removal 第三次 rescue
SC/IT/VT uniform 8K 盲扩
Candidate-B 继续靠 influence 排名
multi-component annealing
旧 SCOPE capability lines
prompt/finalizer tournament
```

---

# 5. 本轮希望得到的论文级信息

最重要的不是多一个 benchmark 数字，而是确定以下三件事：

### Q1. Learnability 到底测对了吗？

如果没测对：

```text
修正 Pre-stage 的第三轴。
```

### Q2. Harness component 应该以什么粒度迁移？

Evidence Graph 很可能给出：

```text
state stays external
decision policy moves to weights
renderer becomes slimmer
```

这是 SCAPE 最核心的 capability placement case。

### Q3. Harness-induced behavior change 是否值得蒸馏？

如果：

```text
Influence 高
但 downstream advantage 低
```

那么 Pre-stage 应从“Influence=distillability proxy”修正为：

```text
Influence describes policy effect
Advantage/value decides supervision usefulness
Learnability describes parameter absorption
```

这会比继续堆候选更有论文价值。

---

# 6. 五份机器指令

```text
SCAPE-0813-Next-H20.md
SCAPE-0813-Next-H100-1.md
SCAPE-0813-Next-H100-2.md
SCAPE-0813-Next-H100-3.md
SCAPE-0813-Next-H100-4.md
```

Cursor Agent 在对应机器只执行对应文件。

---

# 7. 统一记录规则

所有机器：

```text
RUN_MANIFEST.json
STATUS_LIVE.md
SHA256SUMS
```

每个实验必须记录：

```text
repo commit
model revision
split hash
seed
component/view
scorer
retrieval backend
GPU env
LOCAL_COMPAT_ONLY
official_chroma_parity
```

GPU-heavy Python：

```text
H100: /opt
```

不要从 `/mnt` JuiceFS 环境运行 torch/vLLM。

---

# 8. 官方 Chroma

当前记录仍显示：

```text
OPENAI_API_KEY missing
CHROMA_API_KEY missing
CHROMA_DATABASE missing
```

本轮只在 preflight 检查一次。

若仍缺：

```text
OFFICIAL_CHROMA_BLOCKED=true
```

继续 mechanism experiments。

不要：

```text
轮询 credential
用 local BM25/HF 冒充 official parity
```

---

# 9. 本轮结束条件

本轮结束时至少应得到以下之一：

```text
A. 原 Gate-L measurement bug 被确认并纠正；
B. 原 FAIL 被独立确认；
C. Graph-Hybrid micro 出现可重复 PASS；
D. Graph-Hybrid 也 FAIL，合理触发 clean setting；
E. Influence 与 downstream advantage 的关系被量化，解释旧 Pre-stage 失效。
```

只要拿到 A/B + E，本轮就已经有方法诊断价值；不要为了“必须训出正结果”继续无限 rescue。
