# Harness-1 → Bidirectional OPD–Harness Co-Evolution
## Search Agent 场景代码实现 TODO List

> 目标：基于 Harness-1，将原本散落在环境变量、WorkingMemory、工具和上下文渲染函数中的 Harness 机制，重构为**预定义、可开关、可记录、可消融的能力模块**；随后实现 Harness-to-Model 的 OPD 内化，以及 Model-to-Harness 的模块保留、条件化或退休决策。

---

## 0. 项目范围与核心假设

### 0.1 第一版只回答一个问题

- [ ] 验证：Search Harness 中某个能力模块带来的行为增益，能否通过 OPD 部分迁移到模型参数中。
- [ ] 验证：模型完成 OPD 后，该模块的边际价值是否下降。
- [ ] 根据模块训练前后的边际价值变化，执行：
  - `retain`
  - `conditional`
  - `retire`

### 0.2 第一版不做的内容

- [ ] 不做自动节点发现。
- [ ] 不做自动节点聚类。
- [ ] 不做节点级穷举消融。
- [ ] 不做任意 Python Harness 代码自修改。
- [ ] 不做全模块同时 OPD。
- [ ] 不把实时检索内容本身蒸馏进模型。
- [ ] 不重新设计完整 RL 算法。
- [ ] 不同时引入多智能体 Harness。

### 0.3 第一版核心假设

设当前模型为 \(\theta_t\)，Harness 模块为 \(M\)。

模块边际价值：

\[
\Delta R_M(\theta_t)
=
R(\theta_t,H_{\mathrm{full}})
-
R(\theta_t,H_{\mathrm{full}}\setminus M)
\]

OPD 更新后：

\[
\theta_{t+1}
=
U_{\mathrm{OPD}}(\theta_t,D_M)
\]

模块可内化程度：

\[
D_M
=
1-
\frac{
\Delta R_M(\theta_{t+1})
}{
\Delta R_M(\theta_t)
}
\]

实现时必须增加稳定性保护：

```python
if delta_before < min_effect:
    distillability = None
else:
    distillability = 1.0 - delta_after / max(delta_before, eps)
```

模块生命周期决策：

```text
ΔR_after 显著为正             → retain
ΔR_after 下降但仍显著为正     → conditional
ΔR_after 接近 0               → retire
ΔR_after 显著为负             → remove
```

---

# 1. 先复现原始 Harness-1

## 1.1 建立开发分支

- [ ] Fork 或复制 `pat-jj/harness-1`。
- [ ] 创建开发分支：

```bash
git checkout -b feat/module-level-opd-loop
```

- [ ] 不直接修改原始评测入口，先保留以下文件作为 reference：
  - `harness/ultra_core.py`
  - `harness/agent.py`
  - `harness/tools.py`
  - `harness/trajectory.py`
  - `training/train_rl.py`
  - `inference/evaluate_harness1.py`
  - `inference/queue_browsecomp_ablation.py`

## 1.2 跑通最小环境

- [ ] 安装基础依赖：

```bash
uv sync
```

- [ ] 跑 smoke tests：

```bash
uv run python tests/smoke_imports.py
uv run python tests/smoke_cli.py
```

- [ ] 跑通一个模型加载测试。
- [ ] 跑通 5–10 个 BrowseComp+ 样例。
- [ ] 保存以下基线输出：
  - `recall`
  - `trajectory_recall`
  - `final_answer_recall`
  - `precision`
  - `turns`
  - `tool_diversity`
  - `error`
  - 完整 trajectory
  - 环境变量快照

### 验收标准

- [ ] 固定 seed 时，同一配置能重复运行。
- [ ] 能得到 Harness-1 原始轨迹 JSON。
- [ ] 能从输出中还原每轮 action、tool call、observation 和 WorkingMemory。
- [ ] 10 个样例中不存在因重构前准备工作引入的额外错误。

---

# 2. 预定义节点与模块

## 2.1 总体设计原则

节点定义：

> 具有明确输入输出、能够记录 telemetry，并可以被关闭或替换为合法 fallback 的 Harness 操作单元。

模块定义：

> 一组共同解决同一种能力问题、共享消融语义，并具有统一 fallback 的节点。

第一版不根据 embedding 聚类节点。节点和模块全部人工定义，并在配置文件中显式注册。

---

## 2.2 模块总图

```text
                         ┌──────────────────────┐
                         │ Search Policy θ      │
                         └──────────┬───────────┘
                                    │ semantic action
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│ M0 Retrieval Interface                                      │
│ 固定环境接口：第一版不做 retire，不作为主要 OPD 目标         │
└──────────────────────────┬──────────────────────────────────┘
                           │ documents / chunks
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ M1 Evidence State                                           │
│ pool、去重、证据图、curation、importance、review memory       │
└──────────────────────────┬──────────────────────────────────┘
                           │ structured evidence state
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ M2 Verification                                             │
│ claim verification、verification records、验证感知 curation  │
└──────────────────────────┬──────────────────────────────────┘
                           │ verified evidence state
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ M3 Context & Budget                                         │
│ compression、context rendering、window、budget、stop hint    │
└──────────────────────────┬──────────────────────────────────┘
                           │ rendered observation
                           └──────────────────────► Policy θ

Phase 2:
┌─────────────────────────────────────────────────────────────┐
│ M4 Recovery Control                                         │
│ stagnation detection、replan hint、checkpoint、rollback      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2.3 M0：Retrieval Interface

### 定位

- 环境外部能力。
- 第一版固定存在。
- 不尝试将“实时检索内容”内化。
- 只允许后续蒸馏检索策略，例如何时检索、使用哪类查询。

### 节点

| Node ID | 节点 | Harness-1 对应位置 | 输入 | 输出 |
|---|---|---|---|---|
| `R1` | `SearchCorpusNode` | `SearchCorpusTool` | query | ranked chunks |
| `R2` | `GrepCorpusNode` | `GrepCorpusTool` | regex/pattern | exact-match chunks |
| `R3` | `FanOutSearchNode` | `FAN_OUT_SEARCH_SCHEMA` | query list | merged results |
| `R4` | `ReadDocumentNode` | `ReadDocumentTool` | document ID | full document |
| `R5` | `RerankNode` | `harness/rerank.py` | query + candidates | reranked candidates |
| `R6` | `NeighborExpansionNode` | `V8D_CHUNK_NEIGHBORS` | chunk ID | adjacent chunks |

### 第一版处理

- [ ] 将 M0 标记为 `required=True`。
- [ ] 配置层禁止 `enabled=False`。
- [ ] 允许关闭 `R5/R6` 做工程调试，但不作为主论文模块消融。
- [ ] 给每次检索记录 query、返回文档、延迟和调用成本。

---

## 2.4 M1：Evidence State

### 研究问题

> 结构化证据状态是否比原始增长式 transcript 更有效？其中多少状态管理行为可以被模型内化？

### 节点

| Node ID | 节点 | Harness-1 对应位置 | 功能 |
|---|---|---|---|
| `E1` | `MinimalSelectionStateNode` | `WorkingMemory.curated_ids` | 保留任务必需的最终文档选择接口 |
| `E2` | `CandidatePoolNode` | `WorkingMemory.add_to_pool` | 保存历史候选文档 |
| `E3` | `ContentDedupNode` | `ContentDedupTracker` | 去除近重复内容 |
| `E4` | `EvidenceGraphNode` | `EvidenceGraph` | 构造 entity–document 关联 |
| `E5` | `ImportanceTagNode` | `curated_importance` | 维护证据重要性 |
| `E6` | `SubtractiveCurationNode` | `V8D_SUBTRACTIVE_CURATION` | 满容量时替换低价值文档 |
| `E7` | `AutoSeedNode` | `auto_populate_from_first_search` | 首次检索后初始化候选证据 |
| `E8` | `ReviewMemoryNode` | `WorkingMemory.review_docs` | 免费重读历史文档 |
| `E9` | `EvidenceStateRendererNode` | `WorkingMemory.render/snapshot` | 将证据状态写入模型上下文 |

### 关键边界

`E1 MinimalSelectionStateNode` 是评测所需的最小输出接口，不能彻底删除。

因此，M1 的“关闭”语义不是删除全部 WorkingMemory，而是：

```text
Full Evidence State
→ Minimal Evidence State
```

### M1 Full

```yaml
evidence_state:
  enabled: true
  candidate_pool: true
  content_dedup: true
  evidence_graph: true
  importance_tagging: true
  subtractive_curation: true
  auto_seed: true
  review_memory: true
  render_structured_state: true
```

### M1 Fallback：Minimal Evidence State

```yaml
evidence_state:
  enabled: false
  candidate_pool: false
  content_dedup: false
  evidence_graph: false
  importance_tagging: false
  subtractive_curation: false
  auto_seed: false
  review_memory: false
  render_structured_state: false
  preserve_minimal_selection: true
```

Fallback 行为：

- [ ] 保留 `curated_ids` 作为最终输出槽位。
- [ ] 不向模型展示完整候选池。
- [ ] 不展示 evidence graph。
- [ ] 不提供 importance tag。
- [ ] 不提供 `review_docs`。
- [ ] 检索结果按普通 observation 进入 recent transcript。
- [ ] 超出上下文时使用固定截断规则。
- [ ] 不允许模型因模块关闭而遇到未知 tool。

---

## 2.5 M2：Verification

### 研究问题

> 显式证据验证带来的收益，能否转化为模型自主判断“何时验证、验证哪个 claim、发现证据不足后如何行动”的能力？

### 节点

| Node ID | 节点 | Harness-1 对应位置 | 功能 |
|---|---|---|---|
| `V1` | `VerifyToolNode` | `VERIFY_SCHEMA` + `exec_verify_claim` | 检查文档是否支持 claim |
| `V2` | `VerificationRecordNode` | 新增 | 保存 claim、doc、结论与 rationale |
| `V3` | `VerificationStateRendererNode` | 新增 | 将验证记录注入 WorkingMemory |
| `V4` | `VerificationAwareCurationNode` | importance/curate adapter | 将验证结果用于证据升级或降级 |
| `V5` | `VerificationTelemetryNode` | 新增 | 记录验证是否改变后续 action |

### M2 Full

```yaml
verification:
  enabled: true
  expose_verify_tool: true
  store_records: true
  render_records: true
  verification_aware_curation: true
```

### M2 Fallback

```yaml
verification:
  enabled: false
  expose_verify_tool: false
  store_records: false
  render_records: false
  verification_aware_curation: false
```

Fallback 行为：

- [ ] 从 tool schema 中移除 `verify`。
- [ ] System prompt 中同步移除 verify 说明。
- [ ] 模型直接根据已有文档判断相关性。
- [ ] 保留正常 curation，避免改变任务输出格式。
- [ ] 若旧 checkpoint 仍调用 `verify`，返回显式的 `tool_unavailable` 观察，而不是程序崩溃。
- [ ] 统计 unavailable-tool call rate，判断模型是否依赖该模块。

---

## 2.6 M3：Context & Budget

### 研究问题

> 上下文压缩、结构化渲染和预算提示是否可以部分被模型吸收，从而在弱化外部控制后仍保持有效搜索？

### 节点

| Node ID | 节点 | Harness-1 对应位置 | 功能 |
|---|---|---|---|
| `C1` | `SentenceCompressionNode` | `compress_chunk` / `compress_search_observation` | 压缩检索结果 |
| `C2` | `ContextAssemblerNode` | `render_context_within_budget` | 组装 system、WM、recent turns |
| `C3` | `RecentWindowNode` | `RECENT_K` / result summaries | 保留最近交互 |
| `C4` | `BudgetMarkerNode` | `append_token_marker` | 显示上下文占用 |
| `C5` | `DeterministicTruncationNode` | 新增 fallback | 保证关闭模块后仍不超长 |
| `C6` | `StopBudgetControllerNode` | `MAX_TURNS` / `end_search` adapter | 提供预算和停止信号 |

### M3 Full

```yaml
context_budget:
  enabled: true
  sentence_compression: true
  structured_context_rendering: true
  recent_window: true
  token_budget_marker: true
  stop_budget_hint: true
```

### M3 Fallback

```yaml
context_budget:
  enabled: false
  sentence_compression: false
  structured_context_rendering: false
  recent_window: false
  token_budget_marker: false
  stop_budget_hint: false
  deterministic_truncation: true
```

Fallback 行为：

- [ ] 使用原始 observation。
- [ ] 只采用统一、无语义的 oldest-first truncation。
- [ ] 保留硬性 context limit，避免 OOM。
- [ ] 保留硬性 max turns，避免无限循环。
- [ ] 不向模型暴露 token budget marker。
- [ ] 不注入“应当结束搜索”的软提示。

---

## 2.7 M4：Recovery Control（Phase 2）

### 研究问题

> 显式失败检测与恢复控制是否能被模型部分内化？

### 新增节点

| Node ID | 节点 | 功能 |
|---|---|---|
| `X1` | `StagnationDetectorNode` | 检测重复查询、零新增文档、重复 review |
| `X2` | `FailureClassifierNode` | 分类 stale angle、wrong entity、missed facet |
| `X3` | `ReplanHintNode` | 注入最小恢复提示 |
| `X4` | `CheckpointNode` | 保存 WorkingMemorySnapshot |
| `X5` | `RollbackNode` | 回退到稳定快照 |
| `X6` | `RecoveryBudgetNode` | 限制恢复次数 |

### 第一版状态

- [ ] 先创建接口和空实现。
- [ ] 默认 `enabled=false`。
- [ ] M1–M3 与 OPD 跑通前不加入主实验。
- [ ] 不把 prompt 中的 backtracking 文本直接当成完整 Recovery 模块。

---

# 3. 新目录结构

在尽量不破坏上游代码的前提下新增：

```text
harness/
├── graph/
│   ├── __init__.py
│   ├── node.py
│   ├── module.py
│   ├── registry.py
│   ├── execution_context.py
│   └── events.py
├── modules/
│   ├── __init__.py
│   ├── retrieval.py
│   ├── evidence_state.py
│   ├── verification.py
│   ├── context_budget.py
│   └── recovery.py
├── views/
│   ├── student_view.py
│   ├── teacher_view.py
│   └── privileged_artifacts.py
├── telemetry/
│   ├── recorder.py
│   ├── schema.py
│   └── jsonl_writer.py
├── lifecycle/
│   ├── contribution.py
│   ├── distillability.py
│   ├── decision.py
│   └── state.py
├── configs/
│   ├── modules_full.yaml
│   ├── modules_minimal.yaml
│   ├── ablate_evidence_state.yaml
│   ├── ablate_verification.yaml
│   ├── ablate_context_budget.yaml
│   └── opd_verification.yaml
└── existing upstream files...

training/
├── existing upstream files...
├── opd/
│   ├── rollout_worker.py
│   ├── shadow_harness.py
│   ├── teacher_scorer.py
│   ├── student_scorer.py
│   ├── token_alignment.py
│   ├── loss.py
│   ├── replay_buffer.py
│   └── trainer.py
├── train_opd.py
└── train_coevolution_round.py

inference/
├── existing upstream files...
├── evaluate_modules.py
├── queue_module_ablation.py
├── evaluate_pre_post_opd.py
└── summarize_lifecycle.py

tests/
├── existing upstream tests...
├── test_module_config.py
├── test_module_fallbacks.py
├── test_module_trajectory_equivalence.py
├── test_teacher_student_views.py
├── test_opd_alignment.py
└── test_lifecycle_decision.py
```

---

# 4. 定义统一 Node 接口

## 4.1 `harness/graph/node.py`

- [ ] 定义节点状态：

```python
from enum import Enum

class NodeStatus(str, Enum):
    ENABLED = "enabled"
    FALLBACK = "fallback"
    DISABLED = "disabled"
```

- [ ] 定义节点接口：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class NodeResult:
    output: Any
    metadata: dict[str, Any]
    changed_state: bool
    cost: dict[str, float]

class HarnessNode(ABC):
    node_id: str
    module_id: str

    @abstractmethod
    def run(
        self,
        payload: Any,
        context: "ExecutionContext",
    ) -> NodeResult:
        ...

    @abstractmethod
    def fallback(
        self,
        payload: Any,
        context: "ExecutionContext",
    ) -> NodeResult:
        ...
```

- [ ] 每个节点必须返回：
  - `node_id`
  - `module_id`
  - `enabled`
  - `input_digest`
  - `output_digest`
  - `latency_ms`
  - `token_delta`
  - `changed_state`
  - `fallback_used`
  - `error`

### 验收标准

- [ ] 所有节点可以单元测试。
- [ ] 节点关闭时不会破坏下游输入类型。
- [ ] 节点异常时可以按配置 `raise` 或进入 fallback。
- [ ] 不允许静默吞掉异常。

---

# 5. 定义统一 Module 接口

## 5.1 `harness/graph/module.py`

```python
@dataclass
class ModuleConfig:
    module_id: str
    enabled: bool
    lifecycle_managed: bool
    required: bool
    node_overrides: dict[str, bool]
    fallback_mode: str

@dataclass
class HarnessModule:
    module_id: str
    nodes: list[HarnessNode]
    config: ModuleConfig
```

- [ ] 支持：
  - `module.enabled`
  - `node override`
  - `fallback_mode`
  - `required`
  - `lifecycle_managed`
- [ ] M0 设置：
  - `required=true`
  - `lifecycle_managed=false`
- [ ] M1–M3 设置：
  - `required=false`
  - `lifecycle_managed=true`
- [ ] M4 第一版：
  - `enabled=false`
  - `lifecycle_managed=false`

---

# 6. 配置系统改造

## 6.1 不再直接依赖散落环境变量

原有 `V8D_*` 变量继续兼容，但统一转换到 `HarnessConfig`。

- [ ] 新增：

```python
@dataclass
class HarnessConfig:
    retrieval: ModuleConfig
    evidence_state: ModuleConfig
    verification: ModuleConfig
    context_budget: ModuleConfig
    recovery: ModuleConfig
```

- [ ] 配置加载优先级：

```text
CLI override
> YAML config
> legacy environment variables
> default values
```

- [ ] 编写 legacy adapter：

```python
def from_legacy_env() -> HarnessConfig:
    ...
```

- [ ] 在日志开头保存 resolved config。

## 6.2 示例 CLI

```bash
PYTHONPATH=. uv run python inference/evaluate_modules.py \
  --module-config harness/configs/modules_full.yaml \
  --dataset browsecompplus \
  --limit 100 \
  --seed 42
```

### 验收标准

- [ ] 原始 feature flags 能映射到新配置。
- [ ] `modules_full.yaml` 与原 Harness-1 full operating point 结果近似一致。
- [ ] 所有实验目录自动保存 `resolved_config.yaml`。
- [ ] 配置 hash 写入结果文件，防止混淆实验。

---

# 7. 将 Harness-1 机制封装为模块

## 7.1 Evidence State Adapter

- [ ] 新建 `EvidenceStateModule`。
- [ ] 内部复用而不是复制：
  - `WorkingMemory`
  - `ContentDedupTracker`
  - `EvidenceGraph`
  - `auto_populate_from_first_search`
- [ ] 给 `WorkingMemory` 增加统一访问器：
  - `get_minimal_state()`
  - `get_structured_state()`
  - `get_privileged_state()`
- [ ] 增加 `MinimalEvidenceStateAdapter`。
- [ ] 避免在多个文件中直接读取 `V8D_*`。

## 7.2 Verification Adapter

- [ ] 新建：

```python
@dataclass
class VerificationRecord:
    turn_id: int
    claim: str
    doc_ids: list[str]
    judgments: dict[str, bool]
    rationales: dict[str, str]
```

- [ ] `WorkingMemory` 增加：

```python
verification_records: list[VerificationRecord]
```

- [ ] `VerifyToolNode` 调用原有 `exec_verify_claim`。
- [ ] `VerificationStateRendererNode` 只渲染 compact records。
- [ ] Teacher view 可以访问完整 rationale。
- [ ] Student view 默认不能访问 privileged rationale。

## 7.3 Context & Budget Adapter

- [ ] 新建 `ContextBudgetModule`。
- [ ] 封装：
  - `compress_search_observation`
  - `render_context_within_budget`
  - `append_token_marker`
  - recent-window 选择
- [ ] 增加 `RawTruncationRenderer` 作为 fallback。
- [ ] 保证 full 和 fallback 均不会超过模型 context limit。

### 验收标准

- [ ] Full 模式与原 Harness-1 的 action space 一致。
- [ ] 关闭任一模块后，episode 仍能合法结束。
- [ ] 模块关闭不会导致输入 schema 缺失。
- [ ] 模块开关不需要修改核心 episode loop。

---

# 8. 统一轨迹与 Telemetry

## 8.1 扩展 trajectory schema

每个 episode 保存：

```json
{
  "episode_id": "...",
  "query_id": "...",
  "model_id": "...",
  "module_config_hash": "...",
  "task_metrics": {},
  "cost_metrics": {},
  "turns": [
    {
      "turn_id": 0,
      "student_observation": "...",
      "action_tokens": [],
      "tool_calls": [],
      "node_events": [],
      "module_artifacts": {},
      "working_memory_snapshot": {}
    }
  ]
}
```

## 8.2 节点事件

```json
{
  "node_id": "E4",
  "module_id": "evidence_state",
  "status": "enabled",
  "invoked": true,
  "fallback_used": false,
  "changed_state": true,
  "latency_ms": 3.4,
  "input_size": 421,
  "output_size": 932
}
```

## 8.3 模块级指标

- [ ] `module_invocation_count`
- [ ] `module_latency_ms`
- [ ] `module_output_tokens`
- [ ] `module_state_changes`
- [ ] `module_fallback_count`
- [ ] `unavailable_tool_calls`
- [ ] `module_artifact_usage`
- [ ] `module_cost_per_success`

## 8.4 行为指标

- [ ] 查询重复率。
- [ ] 连续 search 次数。
- [ ] 新增文档率。
- [ ] curation precision/recall。
- [ ] verify 后 action change rate。
- [ ] review 后 document retention rate。
- [ ] budget marker 后停止概率。
- [ ] 最后一轮是否因 max-turn 强制结束。

---

# 9. 模块级消融评测

## 9.1 替换原逐机制消融脚本

新增：

```text
inference/queue_module_ablation.py
```

条件只保留：

```python
MODULE_ABLATIONS = {
    "full": "modules_full.yaml",
    "minimal": "modules_minimal.yaml",
    "minus_evidence_state": "ablate_evidence_state.yaml",
    "minus_verification": "ablate_verification.yaml",
    "minus_context_budget": "ablate_context_budget.yaml",
}
```

## 9.2 强制 paired evaluation

- [ ] 所有条件使用相同 query IDs。
- [ ] 相同 seed。
- [ ] 相同 checkpoint。
- [ ] 相同温度和最大 turns。
- [ ] 相同 retrieval backend snapshot。
- [ ] 将 query IDs 写入实验目录。

## 9.3 统计

对每个模块计算：

```python
delta = metric_full - metric_minus_module
```

同时输出：

- [ ] mean delta
- [ ] paired bootstrap 95% CI
- [ ] win / tie / loss
- [ ] rescue count
- [ ] harm count
- [ ] token delta
- [ ] turn delta
- [ ] latency delta

### 输出表

| Condition | Recall | Final Recall | Precision | Turns | Tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Full | | | | | | |
| Minimal | | | | | | |
| − Evidence State | | | | | | |
| − Verification | | | | | | |
| − Context/Budget | | | | | | |

### 验收标准

- [ ] 能稳定识别至少一个正贡献模块。
- [ ] 模块消融不是因程序错误导致性能下降。
- [ ] fallback condition 的错误率与 full 相近。
- [ ] 每个结果可追溯到具体 config hash 和 query ID。

---

# 10. 先确定第一个 OPD 目标模块

## 推荐顺序

1. `M2 Verification`
2. `M3 Context & Budget`
3. `M1 Evidence State`

### 首先选择 Verification 的原因

- 行为边界清晰。
- tool 可关闭。
- 结果容易形成 privileged context。
- 可以区分：
  - 外部 verification compute
  - 模型的 verification policy
- 蒸馏目标明确：
  - 何时验证
  - 验证什么 claim
  - 选择哪些文档
  - 验证失败后继续搜还是结束

### 进入 OPD 的门槛

只有满足以下条件才继续：

- [ ] `minus_verification` 在 paired evaluation 中显著下降。
- [ ] verify tool 确实被调用。
- [ ] verify 调用后存在可观测的 action change。
- [ ] 模块收益不是纯粹来自额外 token。
- [ ] 至少存在一批：
  - full 成功
  - minus-verification 失败
  的 rescue tasks。

---

# 11. 构造 Student View 与 Teacher View

## 11.1 Student View

Student 只能看到部署时允许的信息：

```text
query
+ ordinary recent trajectory
+ 当前启用的非 privileged module state
```

对于 Verification OPD：

```text
Student 不可见：
- verifier rationale
- verifier yes/no labels
- hindsight-selected critical claims
- 最终成功轨迹的未来信息
```

## 11.2 Teacher View

Teacher 在同一个 student prefix 上额外看到：

```text
query
+ student prefix
+ verification records
+ verified / unsupported claims
+ evidence gaps
+ remaining budget
```

## 11.3 数据类

```python
@dataclass
class PrivilegedArtifacts:
    module_id: str
    turn_id: int
    compact_text: str
    structured_payload: dict
    provenance: list[str]
    future_leakage: bool
```

- [ ] 默认拒绝 `future_leakage=True` 的 artifact。
- [ ] 任何 artifact 必须标明来自当前 turn 之前还是之后。
- [ ] 真正 OPD 训练只允许使用当前决策时可计算的 privileged 信息。
- [ ] hindsight 版本作为单独实验，不与 causal 版本混淆。

---

# 12. Shadow Harness：避免跑两条完全独立轨迹

## 12.1 目标

Student 按目标部署配置进行 on-policy rollout，同时 shadow module 从同一环境状态计算 privileged artifact，但不改变 Student 的实际 observation 和 action。

```text
Student rollout
      │
      ├── Student View → policy sample action
      │
      └── Shadow Verification → privileged artifact
                                 ↓
                         Teacher View scoring
```

## 12.2 `training/opd/shadow_harness.py`

- [ ] 输入：
  - query
  - current environment state
  - current retrieved docs
  - student action prefix
- [ ] 输出：
  - privileged module artifact
- [ ] 不修改：
  - student WorkingMemory
  - student tool availability
  - student action history
  - student reward
- [ ] 记录 shadow compute cost。

## 12.3 第一版简化

若实时 shadow verification 成本过高：

- [ ] 先在 rollout 后对缓存文档离线生成 verification records。
- [ ] 将这一步明确标记为 `offline privileged annotation`。
- [ ] 先验证 teacher signal 是否有效。
- [ ] 再升级为真正在线 shadow Harness。

---

# 13. OPD 数据结构

## 13.1 Transition

```python
@dataclass
class OPDTransition:
    episode_id: str
    query_id: str
    turn_id: int

    student_input_ids: list[int]
    action_ids: list[int]
    action_mask: list[bool]

    teacher_input_ids: list[int]
    privileged_module_id: str

    reward: float
    success: bool
    metadata: dict
```

## 13.2 只在关键 action token 上训练

优先训练：

- tool name token
- tool argument token
- `end_search`
- query generation
- curate add/remove IDs
- verify target selection
- 失败后下一步动作

默认不训练：

- 无关格式 token
- 固定 system prompt token
- observation token
- padding token

---

# 14. Teacher Scoring 与 OPD Loss

## 14.1 后端接口

```python
class PolicyBackend:
    def sample(self, input_ids, sampling_config):
        ...

    def score_tokens(self, input_ids, target_ids):
        ...

    def train_step(self, batch, loss_config):
        ...
```

- [ ] 支持当前 Harness-1 模型后端。
- [ ] 不在 Harness 逻辑中硬编码 Tinker。
- [ ] Teacher 使用冻结权重或 EMA 权重。
- [ ] Student 使用正在更新的权重。

## 14.2 Loss

第一版实现 token-level reverse KL 或 sampled-action distillation：

\[
\mathcal L_{\mathrm{OPD}}
=
\sum_t w_t
D_{\mathrm{KL}}
\left[
\pi_\theta(\cdot\mid s_t)
\parallel
\pi_{\theta^-}(\cdot\mid s_t,z_M)
\right]
\]

若无法取得完整 vocab logits，先实现 action-token NLL：

\[
\mathcal L_{\mathrm{sampled}}
=
-\sum_t w_t
\log \pi_\theta(a_t\mid s_t)
\]

其中 \(a_t\) 来自 privileged teacher。

## 14.3 Token 权重

```python
teacher_advantage = teacher_logp - student_logp
weight = clamp(relu(teacher_advantage), 0.0, max_weight)
```

- [ ] 支持：
  - uniform
  - teacher-advantage
  - outcome-gated
  - module-critical-token
- [ ] 首轮默认：
  - successful episode only
  - teacher-advantage weighting
  - action tokens only

## 14.4 与 RL 的关系

第一版训练顺序：

```text
Base/Harness-1 checkpoint
→ Verification OPD
→ Evaluation
```

暂不联合 RL。

第二版：

\[
\mathcal L
=
\mathcal L_{\mathrm{RL}}
+
\lambda_{\mathrm{OPD}}
\mathcal L_{\mathrm{OPD}}
\]

---

# 15. OPD 训练脚本

## 15.1 `training/train_opd.py`

CLI：

```bash
PYTHONPATH=. uv run python training/train_opd.py \
  --base-model <checkpoint> \
  --target-module verification \
  --student-config harness/configs/ablate_verification.yaml \
  --teacher-config harness/configs/modules_full.yaml \
  --dataset browsecompplus \
  --split train \
  --output-dir outputs/opd_verification
```

## 15.2 最小训练阶段

### Stage A：10–50 样例 Pipeline Smoke Test

- [ ] rollout。
- [ ] 生成 privileged artifact。
- [ ] teacher scoring。
- [ ] token alignment。
- [ ] loss 非 NaN。
- [ ] backward 正常。
- [ ] 保存 checkpoint。

### Stage B：小规模有效性测试

- [ ] 200–500 个任务。
- [ ] 只训练 Verification。
- [ ] 评估：
  - Full Harness
  - Minus Verification
  - Minimal Harness

### Stage C：正式规模

- [ ] 扩大训练集。
- [ ] 多 seed。
- [ ] 固定 held-out test。
- [ ] 保存所有训练配置与 artifact schema 版本。

---

# 16. 模型更新后的模块重新评估

对 \(\theta_t\) 和 \(\theta_{t+1}\) 使用相同评测集：

```text
Before OPD:
  full
  minus verification

After OPD:
  full
  minus verification
```

计算：

```python
delta_before = full_before - minus_before
delta_after = full_after - minus_after
```

同时检查：

```python
bare_gain = minus_after - minus_before
full_gain = full_after - full_before
```

预期核心现象：

```text
minus_after > minus_before
且
delta_after < delta_before
```

说明：

- 无 Verification 模块时模型变强；
- Verification 模块的额外边际价值下降；
- 模块能力发生了部分内化。

### 必须排除的替代解释

- [ ] 只是模型整体在 benchmark 上过拟合。
- [ ] 只是输出格式更稳定。
- [ ] 只是搜索 turns 增加。
- [ ] 只是 token 使用增加。
- [ ] 只是对固定 verifier prompt 的模仿。
- [ ] 只在训练问题上下降，OOD 上不成立。

---

# 17. Lifecycle Decision Engine

## 17.1 状态

```python
class LifecycleState(str, Enum):
    ACTIVE = "active"
    DISTILLING = "distilling"
    CONDITIONAL = "conditional"
    RETIRED = "retired"
    REACTIVATED = "reactivated"
```

## 17.2 决策输入

```python
@dataclass
class ModuleAudit:
    module_id: str
    delta_before: float
    delta_after: float
    ci_before: tuple[float, float]
    ci_after: tuple[float, float]
    cost_before: float
    cost_after: float
    harm_rate_after: float
    ood_delta_after: float
```

## 17.3 第一版规则

```python
def decide(audit: ModuleAudit) -> LifecycleState:
    if audit.ci_after[1] < 0:
        return LifecycleState.RETIRED

    if audit.ci_after[0] <= 0 <= audit.ci_after[1]:
        return LifecycleState.RETIRED

    relative_drop = (
        audit.delta_before - audit.delta_after
    ) / max(abs(audit.delta_before), 1e-6)

    if relative_drop >= 0.7:
        return LifecycleState.CONDITIONAL

    return LifecycleState.ACTIVE
```

### 注意

- [ ] 阈值只能在 validation set 上确定。
- [ ] test set 只做最终报告。
- [ ] 若模块具有不可替代的外部能力，不允许完全 retire。
- [ ] M0 Retrieval 永远不进入自动 retire。

---

# 18. Conditional Module Router

若模块未完全退休，实现轻量条件触发。

## 18.1 第一版 Router

输入特征：

- query length
- constraint count
- multi-hop likelihood
- current evidence count
- conflicting evidence count
- remaining budget
- model confidence
- repeated search count

输出：

```text
activate verification?
activate structured evidence state?
activate budget control?
```

## 18.2 先使用规则，不使用学习 Router

Verification 示例：

```python
activate = (
    multi_constraint_query
    or conflicting_sources
    or low_model_confidence
)
```

- [ ] 先验证 conditional 是否节省成本。
- [ ] 后续再训练 Router。
- [ ] Router 训练不与第一轮 OPD 同时进行。

---

# 19. Co-Evolution Round Orchestrator

## 19.1 `training/train_coevolution_round.py`

```text
Round t
│
├── 1. Evaluate θ_t with full/minus-module configs
├── 2. Select target module
├── 3. Collect on-policy student rollouts
├── 4. Compute shadow privileged artifacts
├── 5. Run OPD → θ_{t+1}
├── 6. Re-evaluate full/minus-module configs
├── 7. Audit module value
├── 8. Update lifecycle state
└── 9. Export H_{t+1} config
```

## 19.2 Round Manifest

每轮输出：

```yaml
round: 0
input_model: ...
output_model: ...
input_harness_config: ...
target_module: verification
training_data_hash: ...
evaluation_query_ids_hash: ...
delta_before: ...
delta_after: ...
decision: conditional
output_harness_config: ...
```

---

# 20. 测试 TODO

## 20.1 Unit Tests

- [ ] Node enabled path。
- [ ] Node fallback path。
- [ ] Module config override。
- [ ] Legacy env mapping。
- [ ] WorkingMemory minimal/full view。
- [ ] VerificationRecord serialization。
- [ ] Context fallback 不超长。
- [ ] Lifecycle decision boundary。

## 20.2 Integration Tests

- [ ] Full Harness episode。
- [ ] Minus Evidence State episode。
- [ ] Minus Verification episode。
- [ ] Minus Context/Budget episode。
- [ ] 同一 query 的 paired output。
- [ ] Shadow module 不改变 student state。
- [ ] Teacher view 比 student view 多且仅多 privileged artifact。
- [ ] OPD token alignment 正确。

## 20.3 Regression Tests

- [ ] `modules_full.yaml` 与原 full flags 指标近似。
- [ ] 原 inference CLI 仍可用。
- [ ] 原 smoke tests 仍通过。
- [ ] 配置关闭不会造成 tool schema 与 prompt 不一致。
- [ ] 没有 secret 写入 trajectory。

---

# 21. 实验顺序

## Experiment 0：原始系统复现

目标：

- 确认环境和指标可用。

规模：

- 10 个 smoke cases。
- 100 个 paired evaluation cases。

---

## Experiment 1：模块级消融

设置：

```text
Full
Minimal
− Evidence State
− Verification
− Context/Budget
```

目标：

- 找到第一个高贡献且可内化的模块。

首选判断：

```text
若 Verification 有稳定正贡献 → 进入 Experiment 2
否则选择 Context/Budget
```

---

## Experiment 2：Verification OPD

比较：

```text
θ_before + Full
θ_before − Verification
θ_after + Full
θ_after − Verification
```

核心指标：

- `minus_after - minus_before`
- `delta_before - delta_after`
- token/cost 变化
- verify dependence rate
- OOD generalization

---

## Experiment 3：Verification Lifecycle

设置：

```text
Always On
Always Off
Conditional
```

目标：

- 在性能接近 Full 的情况下减少 verify 调用和成本。

---

## Experiment 4：第二个模块

仅在 Verification 闭环成立后选择：

```text
Context/Budget
或
Evidence State
```

不同时对两个模块做 OPD。

---

# 22. 推荐数据与模型顺序

## 22.1 数据

### 开发阶段

- BrowseComp+ 小规模固定子集。
- 先使用 50–100 个 query 做模块消融。
- 使用独立 query 子集调试 OPD。

### 正式阶段

至少包含：

- in-domain retrieval。
- multi-hop QA。
- web transfer。
- 证据冲突或多约束任务。
- 长程搜索任务。

### 数据划分

```text
train:
  OPD trajectory collection

validation:
  module selection
  lifecycle threshold

test:
  final paired evaluation

OOD:
  transfer evaluation
```

---

## 22.2 模型

### 最小调试

- 使用较小、可本地训练且支持 logits 的模型。
- 重点验证 pipeline，而不是追求最终结果。

### 正式训练

- 先使用与 Harness-1 action format 兼容的 checkpoint。
- 确认：
  - tool-call tokenization
  - teacher scoring
  - student scoring
  - action alignment
  均稳定后再扩大模型。

---

# 23. 关键日志与可视化

## 23.1 每轮必须保存

- module config
- model checkpoint
- query IDs
- trajectory
- privileged artifacts
- teacher/student logprobs
- token weights
- reward
- task metrics
- cost metrics
- lifecycle decision

## 23.2 建议图表

- [ ] 模块消融前后性能柱状图。
- [ ] \(\Delta R_M\) before vs after OPD。
- [ ] 模块成本 vs 边际收益。
- [ ] teacher advantage 分布。
- [ ] 不同 turn 的 OPD token weight。
- [ ] verify 调用次数 before/after。
- [ ] Always On / Off / Conditional Pareto curve。
- [ ] ID 与 OOD 的 distillability 对比。

---

# 24. 里程碑与停止条件

## Milestone A：模块化 Harness

完成条件：

- [ ] M1–M3 均可由 YAML 开关。
- [ ] Full 模式复现原 Harness-1。
- [ ] 所有 fallback 合法。
- [ ] 模块级 paired ablation 可运行。

## Milestone B：找到有效模块

完成条件：

- [ ] 至少一个模块具有显著正边际价值。
- [ ] 模块价值不是由 crash 或 schema mismatch 导致。
- [ ] 找到稳定 rescue task 子集。

## Milestone C：OPD 内化成立

完成条件：

- [ ] `minus_after > minus_before`。
- [ ] `delta_after < delta_before`。
- [ ] held-out test 和至少一个 OOD 数据集成立。
- [ ] 不是单纯增加 token 或 turns。

## Milestone D：Lifecycle 有效

完成条件：

- [ ] Conditional/Retired 配置降低成本。
- [ ] 性能损失处于预设容忍范围。
- [ ] 生命周期决策在多个 seed 上稳定。

## 止损条件

若出现以下情况，应暂停扩大训练：

- [ ] 所有模块消融增益都很小。
- [ ] OPD 只提高 Full Harness，不提高 minus-module。
- [ ] 模块边际价值训练后不下降。
- [ ] OPD 只记忆固定 prompt/tool schema。
- [ ] OOD 上出现明显负迁移。
- [ ] Teacher privileged artifact 存在严重 future leakage。
- [ ] 训练成本远大于部署时保留模块的成本。

---

# 25. 推荐的最小提交顺序

## PR 1：Module Config Foundation

- [ ] 新增 Node/Module/Registry。
- [ ] 新增 YAML config。
- [ ] 新增 legacy env adapter。
- [ ] 不改变原始行为。

## PR 2：Module Wrappers

- [ ] M1 Evidence State。
- [ ] M2 Verification。
- [ ] M3 Context/Budget。
- [ ] 合法 fallback。

## PR 3：Module-Level Ablation

- [ ] paired query runner。
- [ ] summary。
- [ ] bootstrap CI。
- [ ] telemetry。

## PR 4：Privileged Views

- [ ] student view。
- [ ] teacher view。
- [ ] verification records。
- [ ] shadow Harness。

## PR 5：OPD Trainer

- [ ] teacher scoring。
- [ ] token alignment。
- [ ] OPD loss。
- [ ] checkpoint save/load。

## PR 6：Lifecycle Audit

- [ ] before/after module evaluation。
- [ ] distillability。
- [ ] retain/conditional/retire。
- [ ] round manifest。

## PR 7：Recovery Module（可选）

- [ ] stagnation detector。
- [ ] replan hint。
- [ ] checkpoint/rollback。
- [ ] Recovery OPD。

---

# 26. 第一周最值得完成的任务

按顺序：

- [ ] 跑通原 Harness-1 的 10-query evaluation。
- [ ] 将现有 `V8D_*` flags 映射到一个 YAML 配置。
- [ ] 实现 `HarnessConfig` 和 `ModuleRegistry`。
- [ ] 定义 M0–M3 和所有 Node ID。
- [ ] 实现 `modules_full.yaml`。
- [ ] 实现 `ablate_verification.yaml`。
- [ ] 实现 `ablate_evidence_state.yaml`。
- [ ] 实现 `ablate_context_budget.yaml`。
- [ ] 将原 `queue_browsecomp_ablation.py` 改成模块级 runner。
- [ ] 在固定 50 个 query 上完成第一张模块消融表。
- [ ] 根据结果决定首个 OPD 模块。
- [ ] 若 Verification 有效，开始加入 `VerificationRecord` 与 teacher/student view。

---

# 27. 最终最小系统定义

第一篇工作的最小完整闭环应当只有：

```text
Harness-1 Full
    ↓
模块级消融识别 Verification 的价值
    ↓
Student 在 Verification-Off 配置下 on-policy rollout
    ↓
Shadow Verification 生成 privileged artifact
    ↓
Teacher 在同一 student prefix 上评分
    ↓
OPD 更新模型
    ↓
重新评估 Verification 的边际价值
    ↓
Retain / Conditional / Retire
```

只要这个闭环在 held-out 和 OOD 上成立，就已经验证了核心研究命题：

> Search Agent 的能力可以在模型参数与外部 Harness 模块之间动态迁移；Harness 不再是固定不变的运行时，而是会随着模型能力变化而调整生命周期。
