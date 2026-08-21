# H100-4：Actual-Model Baselines + Novelty Guard + Main-Table Contract

> 本机现在按 **8×H100** 使用。

## 0. 任务定位

H100-4 不再做 route-head baseline factory。

1. 把最接近本工作的公开方法做成 **faithful actual-model / no-privilege / real closed-loop baselines**；
2. 持续做 novelty collision audit，确保论文贡献不是已经公开的：
   - harness internalization；
   - non-text privileged information；
   - selective OPD；
   - state matching；
   - outcome verification；
   - evidence-conditioned self-distillation。

如果 H100-1/2/3 产生 positive redesigned Student，本机负责将它放进同一 main-table contract。

输出：

```text
outputs/0818_actual_baselines_novelty/
```

---

# Part A. Novelty Guard：先做，不需要 GPU

## A1. 必须逐篇读方法，不只读 abstract

截至 2026-08-18，至少核对以下公开工作：

```text
OPHSD
  Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning
  arXiv:2605.08741

Privileged Information Distillation / OPSD
  arXiv:2602.04942

SEED
  Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning
  arXiv:2607.14777

OPID
  On-Policy Skill Distillation for Agentic Reinforcement Learning
  arXiv:2606.26790

SERL / Selective Hindsight Distillation
  arXiv:2605.19447

SMRC-SD
  When Privileged Guidance Misaligns: State-Matched Routing and Contextualized Self-Distillation
  arXiv:2608.05219

OVCSD
  From Scoring to Acting: Outcome-Verified Comparative Self-Distillation
  arXiv:2607.27937

EviSD
  Evidence-Conditioned Self-Distillation for Search-Augmented Agents
  arXiv:2608.01359

PS-OPSD
  Is More Privileged Information Better? From Solution Traces to Problem-Solving Structure
  arXiv:2608.01589

TurnOPD
  arXiv:2607.05804

ReOPD
  Multi-Turn On-Policy Distillation with Prefix Replay
  arXiv:2607.04763
```

此外搜索 2026-08-08 至 2026-08-18 新增 arXiv，关键词：

```text
harness distillation
agent harness internalization
privileged on-policy distillation
search agent self-distillation
tool-use distillation
runtime intervention distillation
state-aligned distillation
```

输出：

```text
NOVELTY_MATRIX_20260818.md
NOVELTY_RED_LINES_20260818.md
NEW_PAPERS_0808_0818.md
```

---

## A2. 当前明确不能声称的 novelty

必须写进 red lines：

```text
NOT: first to internalize a harness
```

OPHSD 已明确提出 harness self-distillation，并且 Appendix C 已讨论 fully/partially/non-distillable harness regimes。

```text
NOT: first non-text privileged information distillation
NOT: first action-only privileged distillation
```

PI-Distill/OPSD 已覆盖。

```text
NOT: first selective/state-conditioned OPD
```

SERL、SAGE-OPD、SMRC-SD 等已覆盖。

```text
NOT: first state-aligned successful-teacher correction
```

OVCSD / SMRC-SD 已非常接近。

```text
NOT: first evidence-conditioned self-distillation for search agents
```

EviSD 已覆盖。

```text
NOT: first to study privileged-information representation
```

PS-OPSD 已覆盖 structured/problem-space guidance。

---

## A3. 本轮要重点验证是否仍未被占据的贡献

不要直接写成论文 claim，先作为 collision hypotheses：

### Candidate C1：automatic harness side-effect → executable student action

区别于 OPHSD 的 terminal-context/trajectory distillation：

```text
Harness 自动改变 runtime state
    ↓
该改变不是 teacher 显式发出的 action
    ↓
从 pre/post state delta 编译出 Student 原生 action
    ↓
在 Student 自己 occupancy 上训练
```

重点搜索是否已有工作明确做：

```text
deterministic middleware intervention
post-harness state delta
compile/project to native tool action
distill automatic side effect into model-emitted action
```

### Candidate C2：action-space realizability，而不是泛化的“distillability taxonomy”

OPHSD 已有三档 distillability taxonomy，因此不要抢这个点。

我们的更细问题必须是：

```text
某个 harness component 的 effect 是否能在 Student action space 中被实现？
如果可以，如何将非模型动作的 side-effect 映射为 executable action sequence？
```

例如：

- `verify_tool`：改变 action space，Student 无接口 → 不可直接内化；
- AUTO：自动改 curated state，但 Student 有 `curate` → effect-realizable；
- subtractive + importance：自动 add/remove，但 Student 有 add/remove → effect-realizable；
- chunk_neighbors：提供新外部内容 → 很可能仍需 runtime。

检查是否已有论文把“action-space effect realizability”作为训练机制而非只做讨论。

### Candidate C3：compositional internalization of interacting harness components

重点查：

```text
是否有人不是整体 distill harness，
而是识别两个组件的 functional dependency，
然后联合 internalize 其 composed effect。
```

特别是：

```text
importance + subtractive
auto-populate + dedup (+ rerank)
```

如果已有明确同类，立刻记录 collision，不要继续宣称。

---

# Part B. Faithful actual-model baselines

## B1. 统一 evaluator contract

所有 baseline 必须共享：

```text
same base checkpoint
same train/valid/test query manifests
same retriever
same max_steps
same terminal reward
same qrel/gold
same student inference privilege=false
same LoRA rank / optimizer budget where method permits
same real closed-loop evaluator
```

禁止：

- route-head 代替 actual LLM；
- 64-row argmax bridge 当 paper-grade baseline；
- parsable_rate=0 的 bridge 当 baseline；
- 用不同 Base label；
- 把 Full Harness 当 Base Student。

先重建并冻结：

```text
ACTUAL_BASELINE_PROTOCOL.md
BASE_STUDENT_REAL_CLOSED_LOOP.csv
FULL_HARNESS_REFERENCE.csv
```

Full Harness 仅作为 reference，不要求 Ours 超越。

---

## B2. 优先 baseline

本轮优先做四类，每类 2 seeds，正好 8 卡。

| GPU | Baseline | Seed |
|---|---|---:|
| 0 | `OPSD_ACTION_PI` | 42 |
| 1 | `OPSD_ACTION_PI` | 43 |
| 2 | `OPHSD_FAITHFUL` | 42 |
| 3 | `OPHSD_FAITHFUL` | 43 |
| 4 | `MATCHED_TEXT_PRIVILEGE` | 42 |
| 5 | `MATCHED_TEXT_PRIVILEGE` | 43 |
| 6 | `SEED_OR_OPID_FAITHFUL` | 42 |
| 7 | `SEED_OR_OPID_FAITHFUL` | 43 |

### B2.1 OPSD

按 PI-Distill/OPSD 原定义实现，teacher 看到训练期 PI，Student 不看。

必须实际产生 teacher/student token distributions 或论文规定的 RL+reverse-KL 目标。

不要用“route distribution argmax → tool text”伪装。

### B2.2 OPHSD

直接参考官方 OPHSD 代码/公式，做 Search 场景 adaptation。

关键是：

- teacher supervision 来自 harness-orchestrated context/trajectory；
- Student 无 harness；
- reverse KL；
- actual LLM weights。

由于我们这里是 tool-interactive Search，必须明确记录哪些外部动态内容 teacher 看到了，避免把 Student 根本无法得到的信息当“可内化能力”。

### B2.3 Matched Text

与 redesigned Ours 使用同一训练 states、同一 semantic fields、同一 update budget。

如果 Ours 使用 harness state delta：

- textual baseline 只能把同一 delta/事件信息 deterministic textualize；
- 不允许给它额外 gold、未来 reward 或 reasoning；
- 做 round-trip information audit。

### B2.4 SEED / OPID

优先使用官方公开代码和真实 skill extraction/analyzer。

如果在本 Search harness 上无法 faithful adaptation：

```text
status = BLOCKED_FAITHFUL_ADAPTATION
```

不要用简化 prompt baseline 冒充 SEED/OPID。

此时 GPU6/7 改跑：

```text
SMRC-SD or OVCSD faithful adaptation
```

二选一，以代码/环境可移植性更高者为准。

同样禁止 mock。

---

# Part C. 接收 H100-1/2/3 positive Student

持续轮询共享 outputs（不要后台承诺；Agent 在本次执行脚本中按阶段检查）。

优先级：

```text
1. H100-1 PROJECTED_ACTION_AUTO
2. H100-2 PROJECTED_CURATION_BUNDLE
3. H100-3 RETRIEVAL_HYGIENE_BUNDLE
```

只有 handoff 明确 `GO_*` 的 checkpoint 才进入主比较。

若只有一个 GO：
- 用它做 Ours main row。

若多个 GO：
- 分别作为 component rows；
- 再根据组件兼容性构造一版 `COMPOSED_OURS`；
- 不允许直接把三个 LoRA adapter stack 在一起就称组合方法；
- 必须重新用多组件 teacher/on-policy data 训练一次。

---

# Part D. Main table 的判定逻辑

论文主线首先要求：

```text
Ours Student_after > same-init Student_before
```

第二层：

```text
Ours > Matched Text
Ours > OPSD
Ours > OPHSD
Ours > SEED/OPID or closest faithful multi-turn baseline
```

不要求：

```text
Ours > Full Harness Teacher
```

Full Harness 是 deployment upper/reference，不是成功门槛。

必须报告：

```text
reward
trajectory_recall
curated_evidence_recall
final_answer_recall
invalid_tool_rate
turns/tool cost
paired bootstrap
training tokens / GPU-hours
```

如果 Ours 只赢 Base、没有赢强 baseline：
- 方法尚未形成足够贡献；
- redesign 方法或 framing 的“技术对象”；
- 不允许把 claim 缩成“证明能学到一点 signal”。

如果所有 redesigned Ours 都不超过 Base：
- 论文当前方法线应停止；
- 转向“为什么 automatic harness side-effects 不能通过普通 OPD 内化”的新方法问题，而不是写弱结论论文。

---

# Part E. Case-level baseline analysis

至少抽：

```text
Ours wins / OPSD loses
Ours wins / OPHSD loses
Matched Text wins / Ours loses
Ours learns projected action but downstream fails
Baseline succeeds despite no explicit projected behavior
```

每类 20 cases。

重点寻找：

1. Ours 是否真的把 automatic side-effect 转成自主 tool behavior；
2. OPHSD 是否只能学到一般 procedural pattern；
3. Matched Text 是否通过自然语言“绕过” structured/action projection；
4. Ours 的增益是否来自多调用工具而非更好证据；
5. 是否存在 state/teacher mismatch（与 SMRC-SD/OVCSD 的已知问题区分）。

---

# Part F. 必须输出

```text
RUN_MANIFEST.json
STATUS_LIVE.md

NOVELTY_MATRIX_20260818.md
NOVELTY_RED_LINES_20260818.md
NEW_PAPERS_0808_0818.md
CLAIM_COLLISION_DECISION.md

ACTUAL_BASELINE_PROTOCOL.md
BASE_STUDENT_REAL_CLOSED_LOOP.csv
FULL_HARNESS_REFERENCE.csv

OPSD_TRAINING_CELLS.csv
OPSD_REAL_CLOSED_LOOP.csv
OPHSD_TRAINING_CELLS.csv
OPHSD_REAL_CLOSED_LOOP.csv
MATCHED_TEXT_TRAINING_CELLS.csv
MATCHED_TEXT_REAL_CLOSED_LOOP.csv
SEED_OPID_STATUS.md
CLOSEST_RECENT_BASELINE_STATUS.md

MAIN_TABLE.csv
MAIN_TABLE.md
PAIRED_BOOTSTRAP.csv
COMPUTE_COST.csv
BASELINE_CASE_ANALYSIS.md

H1004_0818_HANDOFF.json
SHA256SUMS
```

最终 handoff 必须同时回答两个问题：

```text
scientific_result:
  DOES_OURS_BEAT_BASE_AND_STRONG_BASELINES?

novelty_result:
  IS_AUTOMATIC_HARNESS_EFFECT_TO_EXECUTABLE_ACTION_PROJECTION
  STILL_DISTINCT_FROM_PUBLIC_WORK_AS_OF_2026-08-18?
```

若 novelty collision 被发现，立即记录具体论文、章节、相同机制，不允许通过换名字规避。
