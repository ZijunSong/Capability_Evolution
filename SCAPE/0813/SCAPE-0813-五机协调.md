# SCAPE 下一阶段五机总调度（基于 2026-08-12 13:14 最新 result-record）

## 0. 已经成立的当前状态

本轮不要再重复已经完成的 Pre-stage 大扫表：

- H100-1：`BCP_CONFIRM400` fresh contribution confirm 已完成，10/10 errors=0。
- H100-2：`BCP_REPL200_V2` 独立 full + 10 LOO + 4 coalition 已完成，16/16 errors=0。
- H100-3：released Harness-1 checkpoint 的 HF continuation-logprob real influence 已完成，7 components × 1024 states。
- H100-4：`evidence_graph` / `importance_tagging` / `subtractive_curation` 的 CONFIRM128 real influence 已完成。
- H20：旧 Qwen2.5-7B + BM25 provisional 的 `auto_populate` / `verify` 线 Gate S 均失败，应归档。
- H20：真正的 SCAPE `same-environment-state + dual-view + tool-token OPD` 尚未完成，是当前主线。

当前证据：

```text
Candidate A = evidence_graph

Candidate B (provisional) = importance_tagging

High-priority challenger = verify_tool

Runtime controls =
    chunk_neighbors
    content_dedup
```

原因：

- `evidence_graph`：
  - H100-1 fresh contribution：positive trajectory/reward
  - H100-2 independent replication：positive trajectory/reward
  - H100-3 real influence：rank #1
  - H100-4 real influence：confirmed positive
- `importance_tagging`：
  - contribution 有正信号但 split-sensitive
  - H100-3/H100-4 real influence 都 positive
- `verify_tool`：
  - local contribution 为 neutral
  - 但 H100-3 real influence rank #2
  - 尚缺 H100-4 confirm，因此继续补证据
- `chunk_neighbors`：
  - contribution positive，但 real influence = 0 above null
  - 更像 runtime / retrieval mechanism
- `content_dedup`：
  - local contribution 可正
  - 但 deterministic/runtime 属性强，不作为首轮 full internalization target

---

# 1. 五台机器分工

```text
H20
  └─ 主线：真正 SCAPE tool-token OPD
     evidence_graph 立即开 Stage L
     Candidate B 等 H100-2/H100-4 补证据后再确定

H100-1
  └─ Evidence Graph Placement Decomposition
     回答：graph 的收益来自 external state、renderer，还是 semantic decision？

H100-2
  └─ Candidate-B Utility Resolution
     importance_tagging vs verify_tool vs subtractive_curation
     不再做全 10-component LOO

H100-3
  └─ Real Influence Attribution + Training-Target Audit
     把 A/B influence 分解到具体 tool / turn / argument token
     同时负责 H100 code branch 的 GitHub 同步

H100-4
  └─ verify_tool CONFIRM128 + targeted verify-event confirmation
     直接补当前最明确的证据缺口
```

---

# 2. 全局禁止

不要继续：

```text
旧 SCOPE rollback
旧 P_m
KEEP/SKIP capability classifier
旧 Information-Safe Routing
Qwen2.5 provisional A/B 扩 seed
旧 Stage M
全 10-component Influence 再跑一遍
全 10-component contribution 再跑一遍
```

官方 Chroma 仍缺 credential：

```text
OPENAI_API_KEY
CHROMA_API_KEY
CHROMA_DATABASE
```

每台机器启动时只检查一次。

没有就：

```text
OFFICIAL_CHROMA_BLOCKED=true
```

继续本地机制实验，不等待、不轮询。

---

# 3. Git 规则

H100 已存在四个 worktree：

```text
SCAPE-wt-h100-1
SCAPE-wt-h100-2
SCAPE-wt-h100-3
SCAPE-wt-h100-4
```

各机器只能在自己的 worktree 改代码。

每台机器开始先：

```bash
git status -sb
git rev-parse HEAD
git remote -v
```

代码变化：

```text
只 commit scripts / scape / configs / tests / docs
不 commit outputs / checkpoints / indexes / secrets
```

H100-3 负责确认 snapshot `0f0934bd...` 及必要公共 scorer/snapshot 代码已 push 到 GitHub。

H20 负责最终 integration / canonical main。

---

# 4. 下一阶段 Go/No-Go

## Gate A — Evidence Graph Learnability

H20 true-SCAPE Stage L 要回答：

```text
evidence_graph 的 real tool-policy influence
是否能通过 same-state tool-token KL 进入 weights？
```

如果：

```text
512 -> 2k -> 8k held-out divergence 不下降
```

则停止 Evidence Graph migration，不进入 Stage S。

## Gate B — Candidate B

Candidate B 只有在：

```text
importance_tagging
vs
verify_tool
```

的“有用 policy influence”证据比较完成后才最终冻结。

## Gate S — Retirement

必须严格四格：

```text
theta0 + H_full
theta0 + H_-m
theta' + H_-m
theta' + H_full
```

只有：

```text
theta' + H_-m
≈ or >
theta0 + H_full

且 runtime cost 下降
```

才能进入 module retirement / multi-component。

---

# 5. 本轮最重要的 paper-level question

这轮不是继续找“哪个模块数值最大”，而是要开始回答：

> `Contribution–Influence–Learnability` 三轴是否真的能预测最终 capability placement？

对于 `evidence_graph`，现在已经有：

```text
Contribution = positive
Influence = positive + replicated
Learnability = 待 H20 true-SCAPE
Retirement = 待 Stage S
```

因此 Evidence Graph 是当前最完整、最适合用来验证新 Probe 是否有效的第一个 component。
