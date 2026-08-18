# SCAPE 五机下一阶段总调度（2026-08-12）

## 0. 当前真实状态

以最新 `SCAPE result-record` 为准：

- H100-1：10-component contribution LOO 已在 `local_bm25_compat` 上完成；官方 Chroma parity 仍阻塞。
- H100-2：旧结果 consolidation 已完成，但**没有完成原计划的独立 10-component REPL200**。
- H100-3：10-component influence 已完成，但只是 `deterministic_offline_stub`，**不是 released Harness-1 真实 logits / logprob influence**。
- H20：Qwen2.5-7B + BM25 provisional 的 A=`auto_populate_first_search`、B=`verify_tool` Stage S 均失败；该线归档，不再扩 seed / Stage M。
- H20 尚有一个不依赖 H100 科学结果、但属于 SCAPE 主方法的工程缺口：**真实 same-environment-state + tool-token OPD 路径尚未完成**。
- 新增 H100-4：4×H100；H100 四机代码与数据共通。

因此下一阶段不是“继续旧任务”，而是补齐三个证据缺口：

```text
(1) contribution 是否可在独立 fresh split 上复现？
(2) influence 是否在 released Harness-1 真实模型分布上成立？
(3) 真正 SCAPE tool-token OPD 能否对经 (1)(2) 筛出的 component 完成 closed-loop migration？
```

---

# 1. 全局优先级

```text
P0 Git/code canonicalization
P1 H100-1 contribution fresh confirmation
P1 H100-2 independent 10-component replication
P1 H100-3 real-model same-state influence
P1 H100-4 independent influence confirmation
P2 H20 true SCAPE tool-token OPD implementation
P3 Candidate selection
P4 H20 Learnability -> Single-component migration
P5 Multi-component only after Gate S
```

官方 Chroma credential 缺失不能阻塞 P1–P4；但是所有 local BM25 结果必须明确标记 `LOCAL_COMPAT_ONLY`。

---

# 2. Git 同步策略：必须先做

## 2.1 为什么

目前有两棵可能不同步的代码树：

```text
H100 shared:
/mnt/songzijun/Capability_Evolution/SCAPE

H20:
/data/ppnm/Capability_Evolution/SCAPE
```

禁止直接让 H100-1/2/3/4 在同一个 Git working tree 上并发 checkout / merge。

## 2.2 H100-3 负责 H100 shared snapshot

在任何其他 H100 Cursor 执行 Git 写操作之前：

```text
H100-3:
  inspect shared dirty tree
  commit code-only snapshot
  branch = sync/h100-20260812
  push branch
```

只提交：

```text
scape/
scripts/
configs/
tests/
docs/
pyproject/requirements/lock files
small manifests needed to reproduce code behavior
```

禁止提交：

```text
outputs/
models/
checkpoints/
large datasets/indexes
logs/
pid files/
API secrets
.env
```

如果 origin / gh auth 不可用：

```text
生成 git bundle:
  artifacts/git/scape-h100-20260812.bundle
并写 GITHUB_SYNC_BLOCKED.md
```

## 2.3 H20 负责 H20 snapshot

同理：

```text
branch = sync/h20-20260812
```

push 或生成 bundle。

## 2.4 H20 做 integration

H20 fetch H100 branch 后建立：

```text
integration/scape-20260812
```

原则：

```text
H20 core SCAPE framework 为主
H100 experiment/influence instrumentation 合并进来
旧 SCOPE compatibility code 只能进入 legacy adapter，不能成为 canonical training path
```

跑：

```text
pytest -q
python scripts/preflight_scape.py
```

通过后：

```text
push integration branch
创建 draft PR 或 fast-forward main（仅在仓库工作流允许时）
```

推荐最终 canonical commit 记录到：

```text
docs/CANONICAL_COMMIT.md
```

## 2.5 H100 独立 worktree

canonical main 同步后，由 H100-3 创建：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4
```

分支：

```text
exp/h1001-contribution-confirm
exp/h1002-independent-repl
exp/h1003-real-influence
exp/h1004-influence-confirm
```

之后四个 Cursor **禁止切换 shared root 的 branch**。

---

# 3. 科学候选原则

当前 local/offline map 只能作为 prior：

```text
High Δ, High I:
  evidence_graph
  chunk_neighbors

Low Δ, High I:
  subtractive_curation
  importance_tagging
  verify_tool

High Δ, Low I:
  auto_populate_first_search
  content_dedup
  adaptive_rerank_instruction
```

但：

- `chunk_neighbors` 本质上偏 retrieval/environment execution，默认作为 runtime/hybrid control，不优先做“完全内化”。
- `content_dedup` 偏 deterministic runtime，也默认作为 runtime control。
- 真正 migration candidate 优先从：
  - `evidence_graph`
  - `subtractive_curation`
  - `importance_tagging`
  - `auto_populate_first_search`
  - `verify_tool`
  中由新结果自动选。

禁止因为旧 quadrant 已经写出来就手工固定 Candidate A/B。

---

# 4. Candidate Selection Gate

H100 三类证据齐后生成：

```text
outputs/scape_prestage_v2/
  CONTRIBUTION_CONFIRM.json
  INDEPENDENT_REPLICATION.json
  REAL_INFLUENCE.json
  REAL_INFLUENCE_CONFIRM.json
  CANDIDATE_SELECTION_V2.json
  CANDIDATE_SELECTION_V2.md
```

进入 H20 Stage L 的 component 必须满足：

### Gate C — Contribution
在 H100-1 fresh confirm 或 H100-2 independent replication 中：

```text
至少一个核心 retrieval metric:
full > minus_m
并且 paired effect 不是纯 outlier
```

### Gate R — Replication
至少两个独立 split 的贡献方向一致，或一边显著、一边不冲突。

### Gate I — Real Influence
released Harness-1 真实模型下：

```text
I_name / I_args 明显高于 null controls
```

### Gate P — Placement sanity
不是纯：

```text
exact executor
retrieval backend
hard accounting
persistent store
cheap deterministic checker
```

首轮只选 top 2。

---

# 5. H20 后续的唯一 canonical 训练线

归档：

```text
Qwen2.5-7B + BM25 provisional
A=auto_populate
B=verify
SCOPE-OPD proxy Gate L/S
```

这些只作为历史 diagnostics。

新的 canonical SCAPE 训练必须使用：

```text
same-environment-state snapshot xi_t
full/reduced dual view
released Harness-1 compatible model
tool-call token mask
true teacher/student token distribution
no capability-specific KEEP/SKIP label
```

若 retrieval 仍只能 local BM25：

```text
结果标记 LOCAL_COMPAT_ONLY
```

但机制实验仍可继续。

---

# 6. 五台机器文件

- `SCAPE_NEXT_H100_1.md`
- `SCAPE_NEXT_H100_2.md`
- `SCAPE_NEXT_H100_3.md`
- `SCAPE_NEXT_H100_4.md`
- `SCAPE_NEXT_H20.md`

每台 Cursor 只执行对应文件；H100 Git 写操作按 H100-3 的同步 barrier 执行。
