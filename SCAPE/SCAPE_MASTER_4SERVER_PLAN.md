# SCAPE 四机实验总调度（2026-08-11）

> 目标：从旧 SCOPE 实验线切换到独立 SCAPE 仓库，以 Harness-1 为公开 Search Harness 主 testbed。
>
> 四台机器分工：
>
> - **H100-1：System Contribution / LOO 主筛选**
> - **H100-2：跨 split / 跨 benchmark 复现 + coalition interaction**
> - **H100-3：Same-State Policy Influence**
> - **H20：Canonical SCAPE repo + Learnability + Single/Multi-Component Migration**
>
> 原则：**H100 只做短期、可并行、无训练或极轻计算的测量；H20 承担代码改造、数据持久化、训练和长闭环。**

---

## 0. 为什么不继续在 SCOPE 仓库上改

SCAPE 已经改变了研究对象：

```text
SCOPE:
hand-designed capability
→ capability-specific typed label
→ local classification / retirement gate

SCAPE:
public real search harness
→ component intervention
→ same-environment-state dual view
→ unified tool-call OPD
→ post-training runtime recomposition
```

因此：

1. **新建独立 `SCAPE` 仓库。**
2. 旧 `/SCOPE` 仓库只读，不再继续开发。
3. 只允许从 SCOPE 复制：
   - 通用的 experiment manifest / seed / hash / paired bootstrap 工具；
   - GPU launcher；
   - result-record 汇总模板；
   - 与方法无关的日志、断点续跑、校验工具。
4. **禁止复制**：
   - `duplicate_evidence` 的 KEEP/SKIP 标签体系；
   - rollback Stage1/Stage2；
   - `P_m` OFF/PROC/FULL；
   - Information-Safe Gate；
   - O7 discriminative classifier；
   - `ModuleRetirementGate` 的旧 A/B/C gate 逻辑。
5. 旧 SCOPE 结果仅作为：
   - `duplicate_evidence` = 已知“简单局部行为可内化”的历史正例；
   - rollback = 已知 hard negative；
   - “canonical answer floor / metric mismatch” = 新实验的指标设计警告。

---

## 1. Canonical Repo 设计

H20 上建立主仓库：

```text
/data/ppnm/Capability_Evolution/SCAPE/
```

H100 上使用：

```text
/mnt/songzijun/Capability_Evolution/SCAPE/
```

若路径无写权限，改为当前用户可写的等价路径，并在 `RUN_MANIFEST.json` 中写入真实绝对路径。

推荐结构：

```text
SCAPE/
├── external/
│   └── harness-1/                 # pinned upstream commit/submodule
├── scape/
│   ├── adapters/
│   ├── state/
│   │   └── snapshot.py
│   ├── rendering/
│   │   └── dual_view.py
│   ├── probes/
│   │   ├── contribution.py
│   │   ├── influence.py
│   │   └── learnability.py
│   ├── training/
│   │   ├── tool_mask.py
│   │   ├── tool_opd.py
│   │   ├── harness_dropout.py
│   │   └── teacher.py
│   ├── eval/
│   │   ├── paired_bootstrap.py
│   │   ├── retirement.py
│   │   └── pareto.py
│   └── metrics/
├── configs/
│   ├── harness/
│   ├── probes/
│   └── training/
├── manifests/
├── scripts/
├── tests/
├── outputs/
├── docs/
│   ├── METHOD_SNAPSHOT.md
│   ├── LEGACY_SCOPE_NOTES.md
│   └── EXPERIMENT_REGISTRY.md
└── result-record.md
```

Harness-1 作为 pinned upstream dependency；**不要直接把 SCAPE 修改堆在 upstream 源码里**。必须优先通过 wrapper / adapter / config override 实现 component mask、snapshot 和 dual-view rendering。只有 upstream 无 hook 时才做最小 patch，并把 patch 放进 `patches/`。

---

## 2. Harness-1 Preflight

Cursor/Claude Agent 首先检查：

```text
Python >= 3.11
uv 可用
CUDA / driver 正常
vLLM 支持 GPT-OSS
pat-jj/harness-1 checkpoint 可访问或已缓存
官方 Harness-1 代码可运行
兼容 retrieval backend 可用
```

特别注意：

- Harness-1 官方 BrowseComp+ 本地完整评测依赖兼容的 Chroma retrieval backend；
- 大型 retrieval index **不随代码仓库一起提供**；
- H100 不允许为了“先跑起来”偷偷换成旧 SCOPE BM25 backend；
- 如果 H100 缺 retrieval backend，只完成 model/harness smoke、代码 instrumentation 和静态测试，写出 `BLOCKED_RETRIEVAL_BACKEND.md`，由 H20 建好统一 backend 后再同步。

---

## 3. 固定 component taxonomy

从 Harness-1 当前公开组件中至少覆盖：

```text
V8D_SUBTRACTIVE_CURATION
V8D_IMPORTANCE_TAGGING
V8D_AUTO_POPULATE_FIRST_SEARCH
V8D_EVIDENCE_GRAPH
V8D_SENTENCE_COMPRESS
V8D_CHUNK_NEIGHBORS
V8D_CONTENT_DEDUP
V8D_VERIFY_TOOL
V8D_TOKEN_BUDGET_MARKER
V8D_ADAPTIVE_RERANK_INSTRUCTION
```

统一 canonical id：

```text
subtractive_curation
importance_tagging
auto_populate_first_search
evidence_graph
sentence_compress
chunk_neighbors
content_dedup
verify_tool
token_budget_marker
adaptive_rerank_instruction
```

必须有：

```json
{
  "component_id": "...",
  "upstream_flag": "...",
  "enabled": true,
  "semantic_or_runtime": "...",
  "changes_context": true,
  "changes_state": true,
  "changes_execution": false
}
```

---

## 4. 四机依赖关系

```text
H100-1 ── Contribution map ───────┐
                                  │
H100-2 ── Replication/coalition ──┼──> H20 candidate selector
                                  │
H100-3 ── Same-state Influence ───┘
                                           │
                                           v
H20:
  Micro Learnability
      ↓
  Single-component migration
      ↓
  objective / same-state ablations
      ↓
  post-training retirement
      ↓
  multi-component annealing
      ↓
  final quality-cost Pareto
```

H20 不需要“等待”其他服务器空转：

- 若 H100 summary 已同步，直接读取；
- 若未同步，H20 自己先跑 CAL64 的最小 LOO + Influence bootstrap；
- 一旦 H100 的正式结果到达，再更新 candidate ranking；
- 已开始的训练若 candidate 排名发生变化，只完成当前 checkpoint，不自动扩更多 seed。

---

## 5. Pre-stage 统一统计口径

### 5.1 System Contribution

对每个 component：

```text
Full Harness
vs
Full Harness - component m
```

至少报告：

```text
curated recall
trajectory recall
final-answer recall（若官方 scorer 支持）
Harness-1 reward / task metric
tool calls
turns
context/rendered tokens
latency
state/memory operations
```

不要只使用 canonical final answer accuracy。

每个 task 做 paired comparison：

```text
delta = full - minus_m
paired W/L/T
paired bootstrap 95% CI
```

### 5.2 Same-State Policy Influence

必须由 **Reduced Harness Student rollout** 决定 state occupancy：

```text
xi_t ~ d^(pi, H_-m)
```

对同一个 `xi_t`：

```text
student view = r_-m(xi_t)
full view    = r_F(xi_t)
```

禁止 Full Harness 独立 rollout。

主统计：

```text
I_name:
  对有限合法 tool name 集合计算归一化概率分布的 JS divergence

I_args:
  在同一 teacher-decoded tool call 上，
  对 argument-key / argument-value token 做 teacher-forced token divergence

额外：
  tool-name disagreement
  exact tool-call disagreement
  argument edit distance
  teacher entropy
  student entropy
```

加入 null control：

```text
same render vs same render
field-order-only perturbation
```

避免 renderer 噪声被误判成 capability influence。

### 5.3 Local Learnability

只在 H20：

```text
N = 512 -> 2k -> 8k same-state samples
```

测：

```text
D_pre
D_post
L_m = 1 - D_post/(D_pre+eps)
```

不以训练集 loss 作为 learnability。

---

## 6. Candidate Selector

不要手工指定“必须蒸馏 rerank / curation”。

自动生成：

```text
outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.json
outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.md
```

排序原则：

### Priority A：进入 H20 micro-distillation
满足：

1. contribution 在至少一个核心 retrieval metric 上为正，或 quality-cost utility 明确为正；
2. same-state influence 高于 null noise；
3. component 的收益至少部分来自 semantic decision，而非纯 deterministic execution。

### Priority B：direct removal / optional
低 contribution，但 runtime cost 高，且 removal 无显著退化。

### Runtime anchor
例如：

```text
exact tool execution
exact token/call accounting
deterministic hard truncation
cheap deterministic dedup
retrieval backend
persistent external store
```

即使重要，也不因为“想删 Harness”而强行蒸馏。

最终只选 **top 2** 做第一轮 learnability。

---

## 7. H20 训练推进 Gate

### Gate L：Learnability
candidate 进入 full migration 前需满足：

- held-out tool divergence 相比 pre-training 明确下降；
- 512→2k→8k 至少总体呈改善趋势；
- 至少两个 seed 不出现方向相反；
- tool syntax / invalid call rate 不恶化。

### Gate S：Single-component migration
必须评估四格：

```text
theta0 + H_full
theta0 + H_-m
theta' + H_-m
theta' + H_full
```

优先目标：

```text
J(theta', H_-m) >= J(theta0, H_full)
```

最低可接受目标：

```text
J(theta', H_-m) non-inferior to J(theta0, H_full)
AND
runtime cost lower
```

同时报告：

```text
CCR_m
N_m_post
HRR
```

### Gate M：Multi-component
只有至少一个 single-component candidate 通过 Gate S 后才允许启动。

---

## 8. 明确停止的旧方向

不再投入 GPU：

```text
SCOPE rollback rescue
rollback checkpoint selector
KEEP/SKIP duplicate_evidence 扩展到更多手工 capability
旧 P_m OFF/PROC/FULL
旧 information-safe routing
旧 query-conditioned finalization tournament
旧 structured readout tournament
```

这些可以写进 legacy/negative results，但不占 SCAPE 主实验预算。

---

## 9. 四份执行指令

- `SCAPE_H100_1_CONTRIBUTION.md`
- `SCAPE_H100_2_REPLICATION_COALITION.md`
- `SCAPE_H100_3_INFLUENCE.md`
- `SCAPE_H20_TRAINING_MIGRATION.md`

每台机器只执行自己的文件，所有产物必须写 `RUN_MANIFEST.json`、`STATUS_LIVE.md`、最终报告和 SHA256。
