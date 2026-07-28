# SCOPE 代码改造 TODO List

> 面向当前 BiSHOP 代码库，将第一篇工作从“双向 Harness–Model 共演化”收缩为：
>
> **固定 Search Harness → 同状态局部指导 → Dual-mode OPD → 选择性能力内化**
>
> 本文档只规划第一篇工作的代码改造。  
> Model → Harness 生命周期更新、模块 retire/conditional、自动 co-evolution、Recovery 均不进入主实现路径。

---

# 0. 改造总原则

当前代码库的执行主干已经可用：

```text
SlidingWindowSearchEnv
    + ultra_core.WorkingMemory
    + tools
    + V8D_* flags
```

不建议为了论文叙事重新实现一套 Harness runtime。应保留现有 Ultra 环境作为唯一真实执行路径，只增加以下能力：

1. 从学生真实 rollout 中导出标准化决策状态；
2. 在同一状态上调用 typed Harness module；
3. 生成结构化 privileged artifact；
4. 将学生动作划分为 endorse / correct / ignore；
5. 生成可验证的 OPD transition；
6. 将 dual-mode OPD 与现有 RL loss 联合训练；
7. 评测模块能力是否在 minimal runtime 下被保留。

改造后的主路径：

```text
SlidingWindowSearchEnv
  → Student on-policy rollout
  → DecisionState snapshot
  → CriticalStateSelector
  → Typed Shadow Module
  → PrivilegedArtifact
  → Endorse / Correct / Ignore
  → OPDTransitionV2
  → RL + Selective OPD
```

核心原则：

> **真实环境只执行学生动作；Harness shadow branch 不修改学生 WorkingMemory，不推进环境，不产生第二条环境轨迹。**

---

# 1. 代码库拆分策略

## 1.1 建议新建独立分支

```bash
git checkout -b scope-selective-internalization
```

暂时保留原有 BiSHOP 代码，但将其分为三个逻辑区域：

```text
legacy_bishop/
    lifecycle/
    coevolution/
    module_retirement/

scope/
    state/
    modules/
    artifacts/
    routing/
    opd/
    stats/
    evaluation/
```

若不希望大规模移动文件，可以先采用兼容式目录：

```text
training/opd_v2/
harness/shadow/
harness/artifacts/
harness/capability/
inference/scope/
configs/scope/
```

推荐后者，降低重构风险。

---

## 1.2 第一篇明确冻结的旧功能

以下代码保留但不进入主训练入口：

- `harness/lifecycle/`
- `train_coevolution_round.py`
- `compute_distillability` 驱动的 retire / conditional 决策
- M4 Recovery
- 自动 Harness 重组
- 模块删除和生成
- 基于下一轮配置的 co-evolution manifest

处理方式：

```text
[ ] 在 README 中标记为 legacy / future work
[ ] 主配置默认关闭
[ ] 主训练脚本不 import lifecycle
[ ] 主实验不输出 ACTIVE / CONDITIONAL / RETIRED
```

不要删除这些代码，后续 BiSHOP 第二篇仍可复用。

---

# 2. 目标目录结构

建议逐步整理为：

```text
harness/
├── ultra_core.py
├── tools.py
├── harness_config.py
├── views/
│   ├── student_view.py
│   └── teacher_view.py
├── shadow/
│   ├── base.py
│   ├── registry.py
│   ├── evidence_shadow.py
│   ├── verification_shadow.py
│   └── budget_shadow.py
├── artifacts/
│   ├── schema.py
│   ├── visibility.py
│   ├── validators.py
│   └── reason_codes.py
├── capability/
│   ├── state.py
│   ├── selectors.py
│   ├── action_space.py
│   └── adapters.py
└── telemetry/
    ├── events.py
    └── writer.py

training/
├── train_rl.py
├── train_scope.py
└── opd_v2/
    ├── transitions.py
    ├── dataset.py
    ├── endorse.py
    ├── correct.py
    ├── weighting.py
    ├── trainer.py
    └── collator.py

inference/
└── scope/
    ├── evaluate_runtime.py
    ├── evaluate_transfer.py
    ├── evaluate_error_slices.py
    └── queue_scope_ablation.py

configs/
└── scope/
    ├── base.yaml
    ├── evidence_only.yaml
    ├── verification_only.yaml
    ├── dual_mode.yaml
    ├── endorse_only.yaml
    ├── fixed_weight.yaml
    ├── adaptive_weight.yaml
    └── minimal_runtime.yaml
```

---

# 3. Phase 0：建立回归基线

在改动 OPD 前，先固定当前代码行为。

## 3.1 固定 baseline 配置

```text
[ ] 导出当前 full Harness YAML
[ ] 导出 verification ablation YAML
[ ] 导出 minimal runtime YAML
[ ] 固定 BrowseComp+ 数据版本和 split
[ ] 固定 BM25 / Chroma 索引版本
[ ] 固定模型 checkpoint
[ ] 固定随机种子
[ ] 固定 max_turns / token budget / search budget
```

建议保存：

```text
artifacts/baselines/
├── full_harness_metrics.json
├── minus_verification_metrics.json
├── minimal_runtime_metrics.json
├── sample_trajectories.jsonl
└── environment_manifest.json
```

## 3.2 回归测试

```text
[ ] 原 `train_rl.py` 单 episode 可运行
[ ] `WorkingMemory` snapshot/restore 一致
[ ] full Harness 指标与当前结果一致
[ ] verification ablation 指标与当前结果一致
[ ] YAML → V8D_* 映射无变化
```

完成标准：

> 新代码尚未启用时，所有旧配置输出与重构前一致。

---

# 4. Phase 1：统一学生真实状态表示

当前 `SlidingWindowSearchEnv` 中的状态分散在：

- query；
- turn history；
- WorkingMemory；
- pool / curated；
- doc_store；
- verify records；
- budget；
- 当前 observation；
- 最近 action。

需要引入只读的标准状态对象。

## 4.1 新建 `DecisionState`

文件：

```text
harness/capability/state.py
```

建议定义：

```python
@dataclass(frozen=True)
class DecisionState:
    episode_id: str
    task_id: str
    turn_id: int

    query: str
    rendered_context: str

    action_history: tuple["ActionRecord", ...]
    observation_ids: tuple[str, ...]
    visible_document_ids: tuple[str, ...]

    pool_document_ids: tuple[str, ...]
    curated_document_ids: tuple[str, ...]
    evidence_claims: tuple["ClaimState", ...]
    verification_records: tuple["VerificationRecord", ...]

    remaining_turns: int
    remaining_search_calls: int | None
    token_budget_used: int
    token_budget_total: int

    last_action_type: str | None
    repeated_query_score: float | None

    wm_snapshot_hash: str
```

要求：

```text
[ ] immutable
[ ] JSON serializable
[ ] 不包含不可见 future observation
[ ] 不直接暴露完整 doc_store
[ ] 能从 env 当前状态确定性构造
[ ] 带 schema_version
```

---

## 4.2 在 `SlidingWindowSearchEnv` 增加导出接口

修改：

```text
training/train_rl.py::SlidingWindowSearchEnv
```

新增：

```python
def export_decision_state(self) -> DecisionState:
    ...
```

以及：

```python
def export_visible_state(self) -> dict:
    ...
```

不要允许 shadow module 直接持有 env 引用。shadow module 只接收 `DecisionState` 和只读 artifact store。

---

## 4.3 observation lineage

当前 future leakage 防护不能只依赖 prompt 约束，需要显式记录 observation 来源。

新增：

```python
@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    source_type: Literal["search", "grep", "read", "review", "verify"]
    source_document_ids: tuple[str, ...]
    created_turn: int
    visible_to_student: bool
    text_hash: str
```

TODO：

```text
[ ] 每次 tool execution 生成 observation_id
[ ] WM 中引用 observation_id，而不是只存文本
[ ] curated item 记录来源 observation_id
[ ] verification record 记录依据 observation_id
[ ] shadow artifact 必须声明 evidence_ids
```

---

# 5. Phase 2：定义可蒸馏动作空间

当前 Harmony tool call 的 token 序列较自由。Dual-mode correction 如果直接生成任意文本，验证难度太高。

应先定义有限、结构化的决策动作。

## 5.1 新建 `CapabilityAction`

文件：

```text
harness/capability/action_space.py
```

建议：

```python
class CapabilityActionType(str, Enum):
    SEARCH = "search"
    REWRITE_QUERY = "rewrite_query"
    OPEN_DOCUMENT = "open_document"
    CURATE_DOCUMENT = "curate_document"
    UPDATE_EVIDENCE = "update_evidence"
    VERIFY_CLAIM = "verify_claim"
    CONTINUE_SEARCH = "continue_search"
    STOP_AND_ANSWER = "stop_and_answer"
    ABSTAIN = "abstain"
```

结构：

```python
@dataclass(frozen=True)
class CapabilityAction:
    action_type: CapabilityActionType
    arguments: dict[str, Any]
    target_claim_id: str | None
    source_observation_ids: tuple[str, ...]
```

---

## 5.2 编写 Action Adapter

文件：

```text
harness/capability/adapters.py
```

负责：

```text
Harmony token/tool call
    ↔ CapabilityAction
```

接口：

```python
def parse_policy_action(raw_action: str) -> CapabilityAction | None:
    ...

def render_capability_action(action: CapabilityAction) -> str:
    ...
```

TODO：

```text
[ ] 支持现有 fan_out_search
[ ] 支持 grep/read/review_docs
[ ] 支持 curate
[ ] 支持 verify
[ ] 支持 end_search
[ ] 无法解析时返回 None，不中断 rollout
[ ] 添加 round-trip 测试
```

---

# 6. Phase 3：关键状态选择器

不能每一步都运行 Shadow Harness。

## 6.1 新建 `CriticalStateSelector`

文件：

```text
harness/capability/selectors.py
```

接口：

```python
class CriticalStateSelector(Protocol):
    def select(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> list[str]:
        # 返回需要调用的 module ids
        ...
```

第一版规则：

```text
verification:
    - 学生准备 end_search
    - 学生更新 evidence 为 supported
    - 学生引用文档
    - 发现 conflict
    - curated 中存在未验证 claim

evidence_state:
    - curate 后
    - review_docs 后
    - 新文档加入 pool 后
    - claim 状态发生变化时

budget_control:
    - 连续两次相似查询
    - 剩余 turn 低
    - 准备停止
    - evidence coverage 长时间不变
```

Toy experiment 首先只实现：

```text
[ ] verification selector
[ ] evidence-state selector
```

Budget selector 后置。

---

## 6.2 selector telemetry

每次选择记录：

```json
{
  "event": "shadow_trigger",
  "episode_id": "...",
  "turn_id": 4,
  "module_id": "verification",
  "trigger": "before_stop",
  "student_action": "stop_and_answer"
}
```

后续需分析：

- shadow 调用率；
- 有效 endorse/correct 比例；
- 不同 trigger 的收益；
- 每个模块的调用成本。

---

# 7. Phase 4：Typed Shadow Harness

现有 `ShadowHarness` 主要服务 Verification，需要升级为模块注册机制。

## 7.1 基类

文件：

```text
harness/shadow/base.py
```

```python
class ShadowModule(ABC):
    module_id: str

    @abstractmethod
    def analyze(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> "PrivilegedArtifact":
        ...

    @abstractmethod
    def validate_candidate(
        self,
        state: DecisionState,
        candidate: CapabilityAction,
        artifact: "PrivilegedArtifact",
    ) -> "ValidationResult":
        ...
```

约束：

```text
[ ] 不允许修改 WorkingMemory
[ ] 不允许执行会改变环境的工具
[ ] 默认不允许额外检索
[ ] 不推进 turn
[ ] 不生成新 observation
[ ] 所有输出带 module_id 和 schema_version
```

---

## 7.2 Shadow Registry

文件：

```text
harness/shadow/registry.py
```

```python
class ShadowRegistry:
    def register(self, module: ShadowModule) -> None:
        ...

    def get(self, module_id: str) -> ShadowModule:
        ...
```

配置：

```yaml
scope:
  shadow_modules:
    - evidence_state
    - verification
```

---

## 7.3 Verification Shadow

文件：

```text
harness/shadow/verification_shadow.py
```

复用现有：

- `exec_verify_claim`
- verify records
- curated/pool/qrels 可用逻辑
- `StudentView`
- `TeacherView`
- `PrivilegedArtifacts`

但输出改为结构化 artifact，不直接返回完整 teacher prefix。

第一版目标：

```text
学生准备回答
    → 检查每个关键 claim 是否有直接证据
    → 检查 cited doc 是否已访问
    → 检查是否存在 unresolved conflict
    → 输出 endorse/correct 建议
```

---

## 7.4 Evidence State Shadow

文件：

```text
harness/shadow/evidence_shadow.py
```

第一版不需要复杂 LLM judge，可先使用 deterministic + heuristic：

- curated 文档是否实际覆盖目标实体；
- claim 是否至少绑定一个 visible document；
- 是否重复 curate 近重复文档；
- claim 状态与 verification record 是否一致；
- evidence coverage 是否提高。

后续再加入 teacher model。

---

# 8. Phase 5：Privileged Artifact Schema

## 8.1 核心数据结构

文件：

```text
harness/artifacts/schema.py
```

```python
class GuidanceMode(str, Enum):
    ENDORSE = "endorse"
    CORRECT = "correct"
    IGNORE = "ignore"

@dataclass(frozen=True)
class PrivilegedArtifact:
    artifact_id: str
    schema_version: str

    episode_id: str
    turn_id: int
    module_id: str

    mode: GuidanceMode
    target_claim_id: str | None
    reason_code: str

    student_action: CapabilityAction
    recommended_action: CapabilityAction | None

    evidence_ids: tuple[str, ...]
    document_ids: tuple[str, ...]

    confidence: float
    metadata: dict[str, Any]
```

---

## 8.2 reason codes

文件：

```text
harness/artifacts/reason_codes.py
```

Verification：

```text
VERIFICATION_SUPPORTED
MISSING_DIRECT_EVIDENCE
UNRESOLVED_CONFLICT
INVALID_CITATION
SOURCE_NOT_VISIBLE
PREMATURE_STOP
```

Evidence：

```text
EVIDENCE_UPDATE_VALID
CLAIM_WITHOUT_SUPPORT
DUPLICATE_EVIDENCE
WEAK_SOURCE_ONLY
MISSING_CLAIM_LINK
INVALID_STATUS_TRANSITION
```

Budget：

```text
REPEATED_QUERY
LOW_INFORMATION_GAIN
COVERAGE_SUFFICIENT
BUDGET_EXHAUSTION_RISK
```

不要让 reason code 完全由自由文本生成。

---

# 9. Phase 6：Visibility Guard

这是与普通 Skill-OPD 拉开差异的关键代码。

## 9.1 新建 visibility validator

文件：

```text
harness/artifacts/visibility.py
```

```python
@dataclass(frozen=True)
class VisibilityCheck:
    valid: bool
    violations: tuple[str, ...]

def check_artifact_visibility(
    state: DecisionState,
    artifact: PrivilegedArtifact,
) -> VisibilityCheck:
    ...
```

必须检查：

```text
[ ] artifact.evidence_ids ⊆ state.observation_ids
[ ] artifact.document_ids ⊆ state.visible_document_ids
[ ] recommended action 不引用不可见文档
[ ] recommended query 可以是新文本，但不能包含未观察事实
[ ] artifact 不包含 future turn 信息
[ ] artifact task_id / episode_id / turn_id 一致
```

---

## 9.2 Prompt-level 防泄漏不作为唯一手段

保留 TeacherView 中的 prompt 约束，但必须增加程序检查：

```python
visibility = check_artifact_visibility(state, artifact)
if not visibility.valid:
    artifact = replace(artifact, mode=GuidanceMode.IGNORE)
```

Telemetry 必须记录 violation 类型。

---

# 10. Phase 7：Endorse / Correct / Ignore 路由

## 10.1 Guidance Router

新建：

```text
training/opd_v2/router.py
```

```python
@dataclass(frozen=True)
class GuidanceDecision:
    mode: GuidanceMode
    artifact: PrivilegedArtifact
    validation: ValidationResult
```

逻辑：

```text
artifact 无效
    → IGNORE

Harness 认可学生动作 + 学生动作合法
    → ENDORSE

Harness 不认可学生动作
    + 有 recommended_action
    + recommended_action 通过局部 verifier
    → CORRECT

其他
    → IGNORE
```

---

## 10.2 Endorse 不应简单等于成功 episode

Endorse 的条件是局部能力模块认可当前动作，而不是：

```text
final reward > 0
```

例如失败轨迹中某次正确的 verify 行为仍可 endorse。

---

## 10.3 Correct 不执行第二条真实环境轨迹

第一版 candidate validation 仅做局部验证：

- action schema 合法；
- target claim 存在；
- referenced document 可见；
- recommended action 与 reason code 一致；
- 参数满足工具约束；
- 无 future leakage。

后续可选增加短 continuation，但不要作为 toy experiment 前置依赖。

---

# 11. Phase 8：OPDTransitionV2

现有 `OPDTransition` 包含 student/teacher 两套 prefix + action，需要升级。

文件：

```text
training/opd_v2/transitions.py
```

建议：

```python
@dataclass(frozen=True)
class OPDTransitionV2:
    transition_id: str

    episode_id: str
    task_id: str
    turn_id: int
    module_id: str
    mode: GuidanceMode
    reason_code: str

    student_state_text: str
    student_action_text: str

    teacher_state_text: str | None
    recommended_action_text: str | None

    student_action_token_ids: tuple[int, ...]
    recommended_action_token_ids: tuple[int, ...] | None

    artifact: PrivilegedArtifact

    validity_mask: int
    teacher_confidence: float

    final_reward: float
    module_weight: float

    policy_version: str
    tokenizer_version: str
    schema_version: str
```

---

## 11.1 为什么仍保留 teacher_state_text

Endorse mode 仍可复用 teacher-conditioned re-scoring：

```text
student prefix
vs.
student prefix + privileged artifact
```

Correct mode 则需要对推荐动作计算 student policy 概率。

---

## 11.2 数据版本与可重放性

```text
[ ] 每条 transition 保存 wm_snapshot_hash
[ ] 保存 state hash
[ ] 保存 artifact hash
[ ] 保存 policy checkpoint id
[ ] 保存 tokenizer id
[ ] 保存配置 hash
[ ] 支持 JSONL 落盘
[ ] 支持离线 replay/debug
```

---

# 12. Phase 9：Dual-mode Loss

## 12.1 Endorse Loss

文件：

```text
training/opd_v2/endorse.py
```

首版实现：

```python
L_endorse = -gate.detach() * logp_student_action
```

其中：

```python
gap = (
    logp_teacher_on_student_action
    - logp_student_on_student_action
).detach()

gate = sigmoid(beta * gap)
```

要求：

```text
[ ] teacher branch stop-gradient
[ ] gate stop-gradient
[ ] validity_mask 乘在 loss 上
[ ] 只对 action token 计算
[ ] observation/tool output token 不参与
[ ] 支持按 module 分组统计
```

---

## 12.2 Correct Loss

文件：

```text
training/opd_v2/correct.py
```

首版采用 sequence-level pairwise loss：

```python
margin = (
    logp_student_recommended_action
    - logp_student_original_action
)

L_correct = -logsigmoid(margin)
```

建议进行长度归一化：

```python
score(action) = mean(token_log_probs)
```

避免短动作天然占优。

TODO：

```text
[ ] original/recommended action 使用同一 student state
[ ] 两个 action 均只计算 policy response token
[ ] 支持 margin 超参数
[ ] 支持 label smoothing 可选项
[ ] correct 样本不使用 teacher 自由生成 CoT
```

---

## 12.3 联合 Loss

文件：

```text
training/opd_v2/trainer.py
```

```python
loss = (
    loss_rl
    + lambda_endorse * loss_endorse
    + lambda_correct * loss_correct
)
```

推荐初始值：

```yaml
opd:
  lambda_base: 0.01
  beta: 5.0
  correct_scale: 1.0
```

不要在第一版修改 GRPO advantage 语义。

---

# 13. Phase 10：与现有 RL 训练循环集成

当前 OPD 路径偏独立 HF 更新。最终应进入统一训练循环。

## 13.1 新建 `train_scope.py`

不要立即重写 `train_rl.py`，先新建：

```text
training/train_scope.py
```

结构：

```text
rollout collection
    → trajectory reward
    → state/action extraction
    → shadow guidance
    → OPDTransitionV2 buffer
    → GRPO batch
    → endorse batch
    → correct batch
    → joint update
```

---

## 13.2 推荐分两步实现

### Stage A：离线 transition 验证

```text
[ ] 用冻结 student rollout 生成 transition JSONL
[ ] 单独训练 endorse/correct loss
[ ] 验证 loss、mask、数据正确性
```

### Stage B：在线联合训练

```text
[ ] 每轮用当前 policy rollout
[ ] 同轮生成 shadow artifacts
[ ] 同轮生成 OPD transitions
[ ] 立即联合更新
[ ] 下一轮重新采样
```

不要一开始同时改 rollout、trainer、distributed backend。

---

# 14. Phase 11：候选替代动作生成

Correct mode 的难点是如何生成 \(a_t^+\)。

## 14.1 第一版：规则模板

Verification：

```text
PREMATURE_STOP
    → VERIFY_CLAIM / CONTINUE_SEARCH

MISSING_DIRECT_EVIDENCE
    → REWRITE_QUERY(target_claim)

UNRESOLVED_CONFLICT
    → SEARCH(independent_source)

INVALID_CITATION
    → OPEN_DOCUMENT(valid_visible_doc)
```

Evidence：

```text
CLAIM_WITHOUT_SUPPORT
    → UPDATE_EVIDENCE(status="unsupported")

DUPLICATE_EVIDENCE
    → CURATE_DOCUMENT(non_duplicate_candidate)

MISSING_CLAIM_LINK
    → UPDATE_EVIDENCE(bind_doc_to_claim)
```

优势：

- 易验证；
- 不依赖额外强模型；
- action space 稳定；
- 适合 toy experiment。

---

## 14.2 第二版：Teacher Model 生成

接口：

```python
class CandidateGenerator(Protocol):
    def generate(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> list[CapabilityAction]:
        ...
```

流程：

```text
Teacher 生成 K 个结构化 action
    → schema validation
    → visibility validation
    → module local verifier
    → 选择最高置信度 action
```

不要直接使用自由文本 response 作为 correct target。

---

# 15. Phase 12：局部 Verifier

## 15.1 通用接口

文件：

```text
harness/artifacts/validators.py
```

```python
@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    score: float
    reasons: tuple[str, ...]

class LocalVerifier(Protocol):
    def validate(
        self,
        state: DecisionState,
        action: CapabilityAction,
        artifact: PrivilegedArtifact,
    ) -> ValidationResult:
        ...
```

---

## 15.2 Verification Verifier

检查：

```text
[ ] claim_id 存在
[ ] action 与缺失证据类型一致
[ ] cited doc 已可见
[ ] stop 时不存在 hard unresolved item
[ ] verify 工具参数合法
```

## 15.3 Evidence Verifier

检查：

```text
[ ] claim–doc binding 合法
[ ] status transition 合法
[ ] 不将未验证 claim 标成 verified
[ ] 不加入近重复文档
[ ] 文档来自 visible pool
```

---

# 16. Phase 13：Capability-aware Weighting

动态权重不是 toy experiment 的第一优先级。

## 16.1 新建统计存储

文件：

```text
training/opd_v2/weighting.py
```

```python
@dataclass
class ModuleStats:
    module_id: str

    shadow_calls: int
    endorse_count: int
    correct_count: int
    valid_count: int
    invalid_count: int

    agreement_count: int
    evaluation_count: int

    ema_contribution: float
    ema_reliability: float
    ema_internalization: float
```

---

## 16.2 统计量近似

### Contribution

```python
G_m = valid_guidance_count / shadow_calls
```

### Reliability

```python
U_m = valid_candidate_count / generated_candidate_count
```

### Internalization

在 held-out shadow states 上：

```python
rho_m = policy_action_agreement / evaluation_count
```

---

## 16.3 权重

```python
lambda_m = lambda_0 * clip(
    G_m * U_m * (1 - rho_m),
    min_scale,
    max_scale,
)
```

配置：

```yaml
opd:
  weighting:
    enabled: false
    ema_decay: 0.95
    min_scale: 0.1
    max_scale: 1.0
    update_every: 20
```

实施顺序：

```text
[ ] 先 fixed weight
[ ] 再仅 reliability weight
[ ] 最后完整 adaptive weight
```

---

# 17. Phase 14：奖励设计改动

当前 v3/v4 reward 不要大改。

## 17.1 保持终局 reward 主导

保留：

- recall；
- precision；
- trajectory_recall；
- final_answer_recall；
- 现有 qrels 指标。

新增可选 citation validity：

```python
reward = existing_reward + alpha_citation * citation_score
```

---

## 17.2 不新增复杂 process reward

首版不增加：

- 每个 Evidence 更新奖励；
- 每次 verify 奖励；
- 每步 information gain reward；
- 大型 typed reward graph。

这些局部信息通过 OPD artifact 和 verifier 提供，不进入 RL reward。

---

## 17.3 固定预算优先

```text
[ ] 固定 max_turns
[ ] 固定 search calls
[ ] 固定 token budget
[ ] 同预算比较成功率
```

只有过度搜索严重时，增加：

```python
reward -= eta * tool_call_count
```

---

# 18. Phase 15：Telemetry 重构

现有 telemetry 未深度挂到每步 RL，需要增加统一事件。

## 18.1 事件类型

文件：

```text
harness/telemetry/events.py
```

事件：

```text
episode_start
decision_state_exported
student_action_parsed
shadow_trigger
artifact_generated
visibility_check
guidance_routed
candidate_generated
candidate_validated
opd_transition_created
loss_computed
episode_end
```

---

## 18.2 必须记录的统计

```text
[ ] 每个 module shadow 调用次数
[ ] endorse / correct / ignore 比例
[ ] visibility violation 数量
[ ] candidate verifier 通过率
[ ] 不同 reason_code 分布
[ ] 各模块平均 loss
[ ] 各模块 gradient norm
[ ] correct action 与学生动作类型转移矩阵
[ ] 每种错误类型的修复率
```

---

# 19. Phase 16：配置重构

新建：

```text
configs/scope/
```

## 19.1 `base.yaml`

```yaml
scope:
  enabled: true

  modules:
    evidence_state: true
    verification: true
    budget_control: false

  selector:
    before_stop: true
    after_curate: true
    after_verify: true
    repeated_query: false

  guidance:
    endorse: true
    correct: true
    ignore_invalid: true

  visibility:
    strict: true

  candidate_generation:
    backend: rule_based
    num_candidates: 1

  opd:
    lambda_base: 0.01
    beta: 5.0
    correct_scale: 1.0
    adaptive_weighting: false
```

---

## 19.2 不再依赖“教师 full YAML / 学生 ablation YAML”表达方法核心

旧路径：

```text
student = ablate_verification.yaml
teacher = modules_full.yaml
```

新路径应改为：

```text
student runtime = minimal train runtime
shadow modules = explicitly registered typed modules
```

教师能力由 shadow registry 决定，而不是通过切换整套 YAML 创建另一条完整运行环境。

旧配置仍可用于 baseline。

---

# 20. Phase 17：评测脚本

## 20.1 Runtime Evaluation

文件：

```text
inference/scope/evaluate_runtime.py
```

评测四种配置：

```text
bare_model
minimal_executor
minimal_executor_plus_hard_verifier
full_harness
```

输出：

```json
{
  "answer_accuracy": 0.0,
  "final_answer_recall": 0.0,
  "citation_precision": 0.0,
  "unsupported_answer_rate": 0.0,
  "search_calls": 0.0,
  "trajectory_length": 0.0
}
```

---

## 20.2 Transfer Evaluation

文件：

```text
inference/scope/evaluate_transfer.py
```

支持：

```text
[ ] Evidence schema 替换
[ ] 字段顺序变化
[ ] reason code 重命名
[ ] BM25 → Chroma
[ ] 新 corpus
[ ] fresh documents
```

---

## 20.3 Error Slice Evaluation

文件：

```text
inference/scope/evaluate_error_slices.py
```

错误切片：

```text
premature_stop
missing_direct_evidence
invalid_citation
unresolved_conflict
duplicate_search
claim_without_support
```

分别统计 pre/post training 修复率。

---

# 21. Phase 18：单元测试

建议新建：

```text
tests/scope/
```

## 21.1 State tests

```text
[ ] DecisionState immutable
[ ] serialization round-trip
[ ] snapshot hash stable
[ ] 不包含不可见 doc
[ ] observation lineage 完整
```

## 21.2 Visibility tests

```text
[ ] future observation 被拒绝
[ ] 不可见文档被拒绝
[ ] 当前可见 evidence 被接受
[ ] 推荐 query 不携带 hidden answer
```

## 21.3 Action tests

```text
[ ] Harmony → CapabilityAction
[ ] CapabilityAction → Harmony
[ ] 非法参数被拒绝
[ ] action token mask 正确
```

## 21.4 Loss tests

```text
[ ] endorse positive gap 权重更大
[ ] gate detach
[ ] teacher branch 无梯度
[ ] correct action 分数上升时 loss 下降
[ ] 长度归一化正确
[ ] invalid sample loss 为 0
```

## 21.5 End-to-end tests

```text
[ ] 一个 episode 可生成 endorse transition
[ ] 一个 episode 可生成 correct transition
[ ] shadow 不改变 student WM
[ ] shadow 不推进 turn
[ ] OPD disabled 时与旧 RL 完全一致
```

---

# 22. Phase 19：Toy Experiment 最小闭环

## 22.1 最小范围

```text
Module:
    Verification only

Triggers:
    before end_search
    after verify

Candidate generation:
    rule-based

Loss:
    endorse + correct

Weight:
    fixed λ = 0.01

Reward:
    existing terminal reward

Runtime:
    SlidingWindowSearchEnv
```

---

## 22.2 需要完成的最小代码

```text
[ ] DecisionState
[ ] CapabilityAction
[ ] VerificationShadow
[ ] PrivilegedArtifact
[ ] VisibilityGuard
[ ] GuidanceRouter
[ ] OPDTransitionV2
[ ] endorse loss
[ ] correct loss
[ ] train_scope.py
[ ] evaluate_error_slices.py
```

暂不实现：

```text
[ ] Evidence Shadow
[ ] Budget Shadow
[ ] adaptive weighting
[ ] LLM candidate generator
[ ] short continuation verifier
[ ] co-evolution
```

---

## 22.3 Toy baselines

```text
B0: GRPO
B1: GRPO + 旧 full-trace OPD
B2: GRPO + same-state endorse-only
B3: GRPO + same-state dual-mode
```

关键指标：

```text
answer accuracy
final_answer_recall
unsupported answer rate
premature stop rate
search calls
```

核心验证：

> B3 是否在 premature-stop 和 missing-evidence 切片上明显优于 B1/B2。

---

# 23. Phase 20：完整实验扩展

Toy experiment 成立后按以下顺序扩展。

## Stage 1：Evidence State

```text
[ ] EvidenceShadow
[ ] claim–doc binding artifact
[ ] evidence status transition verifier
[ ] evidence-specific error slices
```

## Stage 2：跨 schema

```text
[ ] 训练 schema A
[ ] 测试 schema B
[ ] 取消模板 token 匹配依赖
```

## Stage 3：跨 retriever / corpus

```text
[ ] BM25 训练
[ ] Chroma 测试
[ ] fresh corpus
```

## Stage 4：Capability weighting

```text
[ ] module stats
[ ] fixed vs reliability vs full weight
```

## Stage 5：Budget Control

```text
[ ] repeated query detector
[ ] low information gain detector
[ ] fixed-budget evaluation
```

---

# 24. 建议删除或降级的旧接口

以下接口不应继续作为第一篇主入口：

```text
train_coevolution_round.py
lifecycle.decide()
ACTIVE / CONDITIONAL / RETIRED
manifest-driven next-round Harness config
full minus module → auto retirement
```

处理：

```text
[ ] 移至 legacy 说明
[ ] 从主 README 移除
[ ] 不在 `train_scope.py` import
[ ] 不在论文主实验中运行
```

以下接口可以复用但需改名或升级：

| 旧接口 | 新用途 |
|---|---|
| `ShadowHarness` | `VerificationShadow` |
| `StudentView` | 保留 |
| `TeacherView` | artifact-conditioned teacher view |
| `PrivilegedArtifacts` | 替换为 typed `PrivilegedArtifact` |
| `OPDTransition` | 升级为 `OPDTransitionV2` |
| teacher-advantage NLL | `EndorseLoss` |
| reverse-KL proxy | endorse baseline |
| distillability | 仅作为评测指标，不驱动生命周期 |

---

# 25. 新旧代码路径映射

| 当前代码 | 调整后 |
|---|---|
| `SlidingWindowSearchEnv` | 保持执行主干，增加 state export 和 hooks |
| `ultra_core.WorkingMemory` | 保持状态中枢，增加 lineage 和只读 snapshot |
| `V8D_*` | 继续控制 runtime baseline，不再代表 shadow teacher |
| YAML module layer | 用于消融与 runtime 配置 |
| `ShadowHarness` | typed shadow module registry |
| student/teacher two prefixes | endorse mode 保留；correct mode增加替代动作 |
| teacher-advantage NLL | endorse loss |
| reverse-KL proxy | endorse baseline |
| lifecycle delta | 离线 retention/internalization 指标 |
| coevolution manifest | 第二篇保留 |
| M4 Recovery | 完全移出本项目 |

---

# 26. 推荐提交顺序

建议每一阶段独立 commit。

```text
commit 1:
    add immutable DecisionState and state export

commit 2:
    add observation lineage and visibility guard

commit 3:
    add CapabilityAction and Harmony adapters

commit 4:
    add typed PrivilegedArtifact schemas

commit 5:
    refactor ShadowHarness into VerificationShadow

commit 6:
    add CriticalStateSelector and guidance router

commit 7:
    add OPDTransitionV2 and JSONL dataset

commit 8:
    add endorse loss

commit 9:
    add rule-based correct candidates and verifier

commit 10:
    add pairwise correct loss

commit 11:
    add train_scope offline transition mode

commit 12:
    add online RL + dual-mode OPD

commit 13:
    add runtime and error-slice evaluation

commit 14:
    add EvidenceShadow

commit 15:
    add capability-aware weighting
```

---

# 27. 验收标准

## 27.1 工程验收

```text
[ ] OPD disabled 时完全复现原 RL
[ ] shadow 执行不修改学生 WM
[ ] shadow 执行不增加环境 turn
[ ] transition 可完整重放
[ ] visibility violation 为 0 或被正确 mask
[ ] endorse/correct loss 均有单测
[ ] 所有模块可通过 YAML 独立关闭
```

## 27.2 方法验收

```text
[ ] same-state endorse 优于旧 full-trace OPD，或至少更稳定
[ ] dual-mode 优于 endorse-only
[ ] 收益集中在对应错误切片
[ ] minimal runtime 下收益仍保留
[ ] 更换 Evidence schema 后收益不完全消失
[ ] fresh corpus 下不出现 privileged-content memorization
```

## 27.3 止损标准

出现以下情况时，不继续扩展模块：

```text
1. dual-mode 对错误切片无提升；
2. correct candidate 通过率长期低于可接受阈值；
3. 跨 schema 性能完全崩溃；
4. full-trace OPD 与 same-state 无稳定差异；
5. 收益仅来自更长 teacher prompt，而非局部能力监督。
```

此时应优先重新检查：

- 状态表示是否真的 same-state；
- correct target 是否过于粗糙；
- action space 是否过宽；
- verifier 是否提供了真实区分信号；
- 学生 rollout 中关键失败状态是否足够多。

---

# 28. 最优先实现清单

按优先级排序：

```text
P0
[ ] DecisionState
[ ] observation lineage
[ ] CapabilityAction
[ ] VerificationShadow
[ ] PrivilegedArtifact
[ ] VisibilityGuard
[ ] GuidanceRouter
[ ] OPDTransitionV2

P1
[ ] EndorseLoss
[ ] Rule-based Correct Candidate
[ ] CorrectVerifier
[ ] CorrectPairwiseLoss
[ ] Offline transition training
[ ] Error-slice evaluation

P2
[ ] Online RL + dual-mode OPD
[ ] EvidenceShadow
[ ] Minimal runtime evaluation
[ ] Cross-schema evaluation

P3
[ ] Capability statistics
[ ] Adaptive module weighting
[ ] Retriever/corpus transfer
[ ] BudgetControlShadow
```

---

# 29. 最终代码叙事

改造后，代码库不再讲：

> 模型与 Harness 在同一篇工作中不断双向共演化。

而是明确讲：

```text
1. Ultra env 产生学生真实 on-policy 搜索轨迹；
2. 系统在关键学生状态上调用 typed shadow module；
3. shadow module 只读当前可见信息，不改变真实环境；
4. 对学生正确动作进行 endorse；
5. 对学生错误动作生成并验证替代 action；
6. RL 保持 outcome objective；
7. selective OPD 提供局部能力监督；
8. 训练后仅保留 minimal executor 和 hard verifier；
9. 通过模块保留、跨 schema 和 fresh corpus 验证能力内化。
```

这套改造最大限度复用当前 `SlidingWindowSearchEnv`、`WorkingMemory`、工具链和验证代码，同时将方法核心从“完整 Harness 轨迹蒸馏 + 生命周期决策”切换为：

> **same-state、typed、verified、dual-mode 的 Search Harness 能力内化。**
