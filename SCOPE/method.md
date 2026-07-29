SCOPE: Selective Capability On-Policy Encapsulation from Search Harnesses  
0. 研究背景
传统 RL：
$$\tau\sim\pi_\theta \rightarrow R(\tau) \rightarrow \text{update}$$
问题是 Agent 一条轨迹可能几十步，但最后只有一个 reward，不知道中间哪一步做对、哪一步做错。
OPD 改成：
$$\tau\sim\pi_\theta \rightarrow \pi_T(a_t\mid h_t) \rightarrow A_t^D$$
$$A_t^D$$表示 Teacher 相比当前 Policy，对当前 token/action 的 dense correction。
也就是 Student 走自己的轨迹，强 Teacher 在 Student 真正访问到的状态上逐 token 指导。所以它同时解决两个问题：（1）on-policy 解决训练—推理分布错配；（2）distillation 把 trajectory-level sparse reward 变成 token-level dense supervision。 
然而，OPD 的问题是必须维护一个更强的外部 Teacher。OPSD 的关键变化是：
$$\pi_T \quad\Longrightarrow\quad \pi_\theta(\cdot\mid h_t,{c})$$
即 Teacher 和 Student 是同一个模型，只是 Teacher 多看 privileged information $$c$$，如 reasoning 中的 verified solution。于是从“大模型教小模型”，变成“有额外信息的自己，教没有额外信息的自己”。研究问题也变成了“训练时给模型什么额外 context，能够产生值得 internalize 的行为？”
近期热门的 SDAR / OPID / SEED：都在回答 Agent 的$$c$$从哪里来。
- SDAR 解决“怎么稳定用”：Agent 多步交互后，privileged teacher 也可能给错信号
$$A_t=A^{RL}+\lambda\underbrace{g_t}_{\text{gate}}A_t^D$$
因此 RL 保持主干，通过 gate 非对称控制 positive/negative distillation signal。
- OPID 解决“蒸馏什么”：把完整 on-policy trajectory 的 hindsight 提炼成：
$$c= \begin{cases} s_{\text{step}}, & \text{critical step}\\ s_{\text{episode}}, & \text{otherwise} \end{cases}$$
也就是把事后经验组织成层次化 skill，再蒸馏回来。
- SEED 解决“skill 怎么持续进化”：进一步让当前 Policy 自己：
$$\tau \rightarrow s_\theta(\tau) \rightarrow A_t^D \rightarrow \theta' \rightarrow s_{\theta'}(\tau')$$
即 Policy 在进化，产生 hindsight skill 的 Analyzer 也一起进化，形成 self-evolving supervision。
所以这里有一个很自然的问题：为什么 privileged capability 必须只是 solution 或 skill？Agent 的 planning、memory、tool routing、state tracking、failure recovery 等大量能力其实存在于 Harness。
SCOPE 因此想研究：能不能把 Harness 看成 privileged teacher，让带 Harness 的 Policy 教不带 Harness 的 Policy；进一步识别哪些 Harness 能力可以被 internalize，哪些只能部分内化，哪些必须长期保留在 runtime。
1. 研究动机
1.1 Search Harness 不是一个可整体蒸馏的单一教师
近期 OPHSD 已经向前走了一步：它不再仅将某个静态 solution 作为 privileged context，而是让当前模型运行在一个外部 Harness 中，并将 Harness 运行后产生的结构化 terminal context 提供给 teacher，再通过 on-policy self-distillation 训练没有 Harness 的 Student。换言之，OPHSD 首次直接验证了外部 Harness 所诱导出的部分行为模式可以被写回模型参数。但这也留下了一个尚未解决的问题：被蒸馏进去的究竟是哪一类 Harness capability？
OPHSD 的两个实验实例分别是 draft–verify 和 plan–solve，其主要目标都是向模型提供一种更好的推理过程结构，而不是处理完整 Agent runtime 中异质能力的内化问题。
在数学推理中，plan–solve Harness 使用 reference solution 先生成一个 strategic sketch，再由 solver 根据该 plan 完成推导。Student 最终学习到的主要是 problem decomposition、先规划后求解以及在计算前识别关键约束和潜在陷阱等 structural reasoning prior。
例如原文的 OlympiadBench case 中，Harness 先在 plan 阶段明确指出 (2,8) 与 (8,2) 应作为两个 ordered pairs 计数；训练后的无 Harness Student 也会主动检查这一点，而普通 OPSD Student 则直接将其合并并得到错误答案。
draft–verify 先从 online memory bank 检索相似样例生成 draft，再按照 draft 标签检索 confirmers 和 challengers，最后重新判断。Student 并没有把 memory bank 本身“蒸馏进去”，而是学习了其程序性结构：即类似“回忆相似案例 → 比较候选 → 决策”的 case-based reasoning pattern。训练后即使移除外部检索，模型仍大量自发产生类似的 precedent-based reasoning。
因此，OPHSD 实际证明的是 Harness-induced procedural reasoning can be internalized。
Zhao Z, Ma L, Zhang W. Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning[J]. arXiv preprint arXiv:2605.08741, 2026. 具体内容参看 OPHSD。
SCOPE 期望从这个边界继续向前一步，重点关注 Search Agent 的场景。Search Agent 的 Harness 并不是一个单一的 reasoning workflow，而是由 planning、evidence management、verification、tool routing、context management、budget control、persistent state 和 failure recovery 等异质能力组成。其中有些是可学习的决策策略，有些同时包含策略与外部状态，有些则本质上依赖 runtime execution。
因此，SCOPE 更关注“一个真实 Search Harness 中，究竟哪些 capability 可以 internalize，哪些只能部分 internalize，哪些必须长期保留在 runtime？”
预设的 Search Agent 场景下的 Harness 组件/节点与模块：
模块
节点
具体作用
内化到Harness中的潜力
M1 Evidence State
E1 MinimalSelectionState
保存当前最小必要搜索状态，如已覆盖子问题、核心证据、未解决 claim 和下一步目标，避免重复携带完整历史。
中。模型可学习维护紧凑状态的原则，但跨轮持久保存和一致更新最好由 Harness 保证。

E2 CandidatePool
维护尚未正式纳入证据集合的候选文档、片段和 claim，记录来源、相关性和待处理状态。
低—中。候选筛选策略可以内化；动态集合、索引和跨步骤持久化属于外部状态管理能力。

E3 ContentDedup
检测重复网页、近重复段落、同源转载和语义高度重叠的证据，避免上下文被冗余内容占满。
中。模型可学习语义去重判断，但哈希、URL 归一化和确定性重复检测更适合保留在 Harness。

E4 EvidenceGraph
将问题、claim、证据、来源和支持／反驳关系组织为结构化图，显式表示哪些证据支撑哪些结论。
中。关系识别和图构建策略可以部分内化；图的持久存储、更新和一致性检查应由 Harness 承担。

E5 ImportanceTag
为证据添加必要性、可信度、独立性、覆盖范围和潜在冲突等标签，区分核心证据与辅助信息。
高。证据重要性判断属于模型可学习的认知策略，适合通过轨迹监督和 OPD 内化。

E6 SubtractiveCuration
主动删除过时、低价值、重复或被更强证据替代的内容，而不是只向上下文中不断追加信息。
高。判断“应删除什么、保留什么”的策略具有较强可学习性，是很适合 BiSHOP 蒸馏的节点。

E7 AutoSeed
在任务开始或证据不足时，根据问题自动建立初始实体、claim、子问题和候选搜索方向。
中—高。初始问题分解和 seed 生成可以内化，但 seed 对应的信息获取仍依赖检索接口。

E8 ReviewMemory
记录已审阅内容、审阅结论、拒绝原因和待复核项，防止重复阅读或再次采纳已判定无效的证据。
低—中。写入和读取记忆的策略可内化，但可靠的跨步骤、跨轮次记忆必须由外部存储实现。

E9 EvidenceStateRenderer
将内部证据状态转换成模型可读的文本、表格或结构化提示，控制字段顺序、粒度和展示范围。
中。模型可学习适合自己的证据表达形式，但确定性序列化、格式约束和长度控制仍适合由 Harness 完成。
期望模型从 Evidence State 模块中学到以下能力：创建、更新和合并 claim；将证据绑定到具体 claim；区分直接证据和间接相关信息；标记 supported / unsupported / conflicting；决定哪些证据进入当前工作上下文。
模块
节点
具体作用
内化到Harness中的潜力
M2 Verification

V1 VerifyTool
对 claim 执行外部核验，如重新搜索、跨来源比对、数据库查询、规则检查或专用 verifier 调用。
低。验证策略和工具选择可以内化；获得实时事实、执行程序和独立外部核验的能力不可内化。

V2 VerificationRecord
保存每个 claim 的验证状态、所用证据、验证方法、置信度、冲突信息和失败原因。
低。模型可生成记录内容，但可靠的结构化存储、版本控制和跨轮一致性应保留在 Harness。

V3 VerificationStateRenderer
将 claim 的已验证、待验证、冲突、证据不足等状态呈现给模型，帮助其决定下一步搜索行为。
中。验证状态的理解可以内化；状态读取、格式渲染和确定性暴露仍属于 Harness 功能。

V4 VerificationAwareCuration
根据验证结果调整证据集合：保留独立支持证据，降低未验证内容权重，移除被反驳或来源不可靠的材料。
高。这是典型的认知控制策略，可通过“验证状态—证据操作”轨迹进行内化。

V5 VerificationTelemetry
统计验证调用次数、成功率、冲突率、未验证 claim 数量、验证成本及其对最终答案的贡献。
低。对 telemetry 的理解可学习，但指标采集、计数和实验观测必须由 Harness 提供。
期望模型从 Verification 模块中学到以下能力：何时需要额外验证；应验证哪个 claim；如何处理冲突证据；何时打开原始来源；何时拒绝基于不足证据作答。
模块
节点
具体作用
内化到Harness中的潜力
M3 Context & Budget

C1 SentenceCompression
将长文档或证据片段压缩为保留关键实体、关系、数值、限定条件和出处的紧凑表达。
高。信息压缩和摘要属于模型擅长学习的能力，具有很高的参数内化潜力。

C2 ContextAssembler
根据当前子目标，将任务描述、证据状态、验证状态、历史动作和工具反馈组合成下一步模型上下文。
高。上下文选择和组织策略可以显著内化；严格格式和安全边界仍可由 Harness 保底。

C3 RecentWindow
保留最近若干轮交互或最近发生的关键事件，裁剪较早且已被状态摘要覆盖的历史内容。
中。模型可学习关注近期信息，但精确窗口维护和 token 级裁剪最好作为确定性 Harness 操作。

C4 BudgetMarker
显式告知剩余 token、搜索次数、工具调用数、时间或成本预算，使模型能够进行资源感知决策。
低—中。预算意识可内化，但真实剩余资源必须由外部系统计量，不能依赖模型自行估计。

C5 DeterministicTruncation
当上下文超过上限时，按照固定优先级和确定性规则删除内容，保证运行稳定且实验可复现。
低。这是典型运行时机制；模型可以学习减少冗余，但不能替代严格的 token 长度约束。

C6 StopBudgetController
根据剩余资源、证据充分性和预期收益决定继续搜索、停止搜索或直接输出答案，同时执行硬预算上限。
中—高。停止决策和边际收益判断可以内化；硬性次数、时间和费用限制必须由 Harness 强制执行。
期望模型从 Context & Budget 模块中学到以下能力：继续搜索、改写查询、打开文档还是停止；如何避免重复搜索；如何在固定 search-call 和 token budget 下分配资源；何时信息增益已经不足。
更完整的 M0-M5 Harness 模块参看 双向闭环 BiSHOP 的 Step 0。
1.2 SCOPE 的核心训练原则：在真实学生状态上分离可迁移能力与特权信息
DOPD 的 privilege illusion 与 DAgger 的 covariate shift 指向同一个问题：教师更强，并不意味着其完整行为都适合作为学生监督。
[1] Yu X, Li G, Si Q, et al. DOPD: Dual On-policy Distillation[J]. arXiv preprint arXiv:2606.30626, 2026. 具体参看 DOPD。
[2] Li C, Qiang R, Huang J, et al. Revisiting DAgger in the Era of LLM-Agents[J]. arXiv preprint arXiv:2605.12913, 2026. 具体参看 DAgger。
一方面，Full Harness 的优势可能同时来自可学习的决策能力与额外信息：
$$\Delta_{\text{harness}}=\Delta_{\text{capability}}
+
\Delta_{\text{information}}.$$
前者包括 query 选择、证据组织、停止决策和冲突处理等策略；后者则来自额外网页、verifier、外部 memory 或真实预算状态。SCOPE 只希望内化前者，因此只有当 Harness 不引入学生尚未观察到的事实信息，却仍能改善当前决策时，才将其视为 transferable capability signal。
另一方面，这类监督必须发生在学生真实访问的状态上。仅学习 teacher trajectory 会产生状态分布偏移，而纯 student rollout 在早期又可能迅速进入失败区域，缺少有价值的恢复状态。因此，SCOPE 采用：
- 基于学生真实状态的监督（student-state supervision）：监督始终建立在学生实际 rollout 所访问到的状态上，而不是直接让学生模仿 Full Harness 独立生成的完整轨迹。具体来说，在学生已经观察到当前证据、执行过已有动作并形成当前 DecisionState 后，再让对应 Harness module 对“这一状态下下一步应该如何决策”提供局部指导。保证训练状态与部署时学生真正会遇到的状态尽可能一致，减少 teacher trajectory 与 student trajectory 之间的状态分布偏移。
- 受可见信息约束的局部指导（visibility-constrained artifact）：Harness 不直接把完整推理过程、后续搜索结果或额外外部知识交给学生，而只针对当前决策生成一个结构化的局部指导 artifact，例如“当前 claim 缺少直接证据，应继续搜索”“已有证据存在冲突，应寻找独立来源”“当前覆盖已经充分，可以停止搜索”。同时，通过 visibility mask 强制要求该指导只能引用学生当前已经获得的 observation、evidence 和真实 runtime metadata。尽量将 Harness 带来的决策能力优势与其额外搜索、外部验证器、持久 memory 等带来的信息优势分离开，使蒸馏目标更加接近真正可被模型参数内化的 capability。
- 按需局部恢复（recovery-on-demand）：仅依赖学生自己的主轨迹仍存在一个问题：一旦学生在关键位置做出严重错误，例如过早停止、进入重复搜索或遗漏关键证据，后续本应出现的有效状态就不会再被访问。SCOPE 因此只在少量高影响错误处，从当前状态额外建立一个 recovery branch：先执行一次经过验证的纠正动作，将轨迹拉回合理区域，随后立即重新交还给学生继续 rollout，而不是让 Harness 长时间接管。这样既能补充“发生错误之后如何恢复”的训练状态，又尽量避免完整 teacher takeover 改变学生的状态分布或同时混入多个 Harness capability。
因此，SCOPE 的核心并不是让学生复制一个更强 Harness 的完整行为，而是在学生真实访问的状态上，从 Harness 中抽取仅依赖当前可见信息的局部决策能力，并在必要时补充错误边界附近的恢复状态，从而训练模型逐步获得原本由 Harness 提供的可迁移能力。
2. 方法
方法总览
                   ┌──────────────────────────────┐
                   │  Full Search Harness H       │
                   └──────────────┬───────────────┘
                                  │ module decomposition
                                  ▼
          Distillability Probe: procedural / hybrid / runtime-only
                                  │
                                  ▼
Bare student πθ ── on-policy ──> student state s_t
                                  │
                                  ▼
                      DecisionState d_t = ψ(s_t)
                                  │
                       same-state shadow query
                                  ▼
                         typed module h_m
                                  │
                                  ▼
               local information-safe artifact z_t^m
                                  │
                         validate / compare
                     ┌────────────┴────────────┐
                     ▼                         ▼
                ENDORSE                    CORRECT
              label = a_t^-           label = verified a_t^+
                     └────────────┬────────────┘
                                  ▼
                  Selective Decision Imitation
                 + optional outcome RL / KL stab.
                                  │
                     high-impact correction?
                      ┌───────────┴───────────┐
                      │ no                    │ yes
                      ▼                       ▼
             continue main rollout      fork recovery branch
                                       execute a_t^+ once
                                       student continues K turns
                                  │
                                  ▼
                        capability statistics
                      reliability / gain / ρ_m
                                  │
                                  ▼
                         module lifecycle
             internalize → downweight → retire / retain runtime
Step 0：同状态决策蒸馏
对于模块$$m$$，定义三种运行形态：
- $$H_{-m}$$：关闭模块$$m$$；
- $$H_m^{\text{proc}}$$：模块可读取学生当前已经观察到的信息和 runtime metadata，但禁止额外获取新事实；
- $$H_m^{\text{full}}$$：模块拥有完整原始能力，包括必要的外部访问。
定义 procedural share：
$$P_m = \operatorname{clip}\left(
\frac{R(H_m^{\text{proc}})-R(H_{-m})}
{R(H_m^{\text{full}})-R(H_{-m})+\epsilon}, 0, 1
\right).$$
直觉：
- $$P_m\approx 1$$：模块收益主要来自程序性策略，优先蒸馏；
- $$0<P_m<1$$：Hybrid，蒸馏决策部分但保留 runtime；
- $$P_m\approx 0$$：收益主要来自新信息/外部执行，不应以“内化”作为目标。
初步实验表明：Search Harness 的能力具有明显异质性，并非都适合内化。 其中 external_verification 等模块的收益主要依赖外部信息或执行能力，应保留在 runtime；duplicate_evidence 观察到明确程序性恢复信号；deterministic_truncation 因测试中未实际触发，暂无法判断。
Step 1：学生 on-policy rollout 与 DecisionState
对于任务$$x$$：
$$\tau^- \sim \pi_\theta(\cdot\mid x,H_{\min}),$$
其中$$H_{\min}$$只保留不可取消的 executor、状态存储和硬约束，不启用待蒸馏的认知控制模块。
时间步$$t$$的真实环境状态为：
$$s_t=(x,o_{\le t},a_{<t},E_t,B_t),$$
其中，$$o_{\le t}$$表示学生真实获得的 observation；$$a_{<t}$$表示此前动作；$$E_t$$表示当前 Evidence / Verification state；$$B_t$$表示真实 runtime budget metadata。
学生执行动作：
$$a_t^-\sim\pi_\theta(\cdot\mid s_t).$$
为了让不同 Harness module 对齐到统一接口，将原始交互状态压缩为 DecisionState：
$$d_t=\psi(s_t).$$
实现上，$$\psi$$对应 `harness/capability/state.py` 中的 `DecisionState`（schema：`scope.decision_state.v2`）。在线 rollout（如 `train_rl.py`）从 WorkingMemory、action history、evidence/verification 记录直接构造；claim 分桶可通过 `with_derived_claims()` 从 `evidence_claims.status` 推导。核心字段示例：
{
  "schema_version": "scope.decision_state.v2",
  "episode_id": "...",
  "task_id": "...",
  "turn_id": 12,
  "event_id": "ep:12",
  "query": "...",
  "goal": "...",
  "active_subgoal": "",
  "observation_ids": ["obs_3", "obs_6"],
  "pool_document_ids": ["d1", "d2"],
  "curated_document_ids": ["d1"],
  "evidence_claims": [{"claim_id": "c1", "text": "...", "status": "supported"}],
  "supported_claims": ["c1"],
  "unsupported_claims": ["c3"],
  "conflicting_claims": [],
  "verification_records": [],
  "last_action_type": "search",
  "last_action_arguments": {"query": "..."},
  "last_query": "...",
  "repeated_query_score": 0.6,
  "repeated_query_count": 1,
  "remaining_turns": 23,
  "remaining_search_calls": null,
  "remaining_open_calls": null,
  "token_budget_used": 1200,
  "token_budget_total": 32768,
  "wm_snapshot_hash": "..."
}
字段约定：
- `goal` 默认等于 `query`；`active_subgoal` 预留，当前在线路径常为空。
- 证据候选用 `pool_document_ids`（别名 `candidate_evidence_ids`），观测用 `observation_ids`（别名 `observed_ids`），而非笼统的 `candidate_sources`。
- 上一步结果不单独存 `last_result_summary`，而由 `last_action_type` / `last_action_arguments` / `last_query` 与 `rendered_context` 表达。
- `supported_claims` / `unsupported_claims` / `conflicting_claims` 为 `DERIVED_VISIBLE`：按 claim status 分桶（supported/verified/linked → supported；conflict* → conflicting；其余 → unsupported）。
- `repeated_query_count` 由相邻 search 的 token overlap（`repeated_query_score`）阈值化得到；`remaining_search_calls` / `remaining_open_calls` 字段存在，但当前在线构造常填 `null`，预算主要靠 `remaining_turns` 与 token budget。
- 每个字段有 provenance（`OBSERVED` / `RUNTIME` / `DERIVED_VISIBLE`），并由 `check_info_safety()` 强制 $$\operatorname{Info}(d_t)\subseteq\operatorname{Info}(s_t)$$。
这里的 DecisionState 不是新的 privileged context，作用是把“这一刻需要作什么控制决策”显式化，使 Evidence、Verification、Budget 模块可以在同一状态接口上提供局部监督。
Step 2：Same-State Shadow Guidance
对于选中的模块$$m$$，在不改变主环境轨迹的情况下运行：
$$z_t^m=h_m(d_t).$$
其中 $$z_t^m$$是一个局部 typed artifact，而不是另一条完整 Harness trajectory。
推荐统一 schema（对齐 `PrivilegedArtifact` / `VerificationShadow` 产出）：
{
  "schema_version": "scope.artifact.v3",
  "module_id": "verification",
  "capability_id": "premature_stop",
  "mode": "correct",
  "target": "claim_3",
  "target_claim_id": "claim_3",
  "diagnosis": "missing_direct_evidence",
  "recommended_operation": "rewrite_query",
  "operation_args": {
    "query": "when was X founded",
    "target_claim": "when was X founded",
    "query_intent": "fill_missing_claim"
  },
  "evidence_ids": ["obs_3", "obs_6"],
  "document_ids": [],
  "runtime_fields_used": ["remaining_turns"],
  "reason_code": "MISSING_DIRECT_EVIDENCE",
  "confidence": 0.9,
  "teacher_source": "VerificationShadow"
}
字段约定：
- 代码字段名是 `module_id`（取值 `"verification"`），不是文档旧称 `verification_control`。
- `reason_code` 用闭集大写枚举（如 `MISSING_DIRECT_EVIDENCE`）；`diagnosis` 常为其小写形式。
- 学生准备 stop 且证据不足时：无 curated docs → `rewrite_query` + `query_intent=fill_missing_claim`；有 curated 但未 verify → `verify_claim`；冲突 → `search`。
- `evidence_ids` ⊆ `DecisionState.observation_ids`；`document_ids` 来自 curated 可见文档。
- 完整序列化还含 `artifact_id` / `episode_id` / `turn_id` / `student_action` / `recommended_action` / `metadata`。
然后由轻量 `ActionRealizer`（`harness/shadow/action_realizer.py`）将 artifact 映射为候选动作：
$$a_t^m = g(d_t,z_t^m)=\operatorname{ActionRealizer.realize}(d_t,z_t^m).$$
$$g$$是确定性 runtime mapping，不是新的认知模块；只用学生可见的 $$d_t$$，不调 LLM、不读隐藏 teacher 信息。
映射优先级（代码实际路径）：
1. `mode=endorse`：恒等映射，直接返回 `student_action`（`source=artifact_recommended`）。
2. `capability_id=duplicate_evidence`：把 `skip_curate` 实现为 `curate_document(add_ids=[], remove_ids=[...])`；若 artifact 已带可执行 `recommended_action`，优先用之。
3. `capability_id=premature_stop`：优先用已填好的 `recommended_action`（`search` / `rewrite_query` / `continue_search` / `verify_claim`）；若缺 `query` 但有 `query_intent`，则用 `_intent_to_query(d_t, intent)` 从 `goal` + unsupported claims 拼出可执行 query（如 `fill_missing_claim` → `"{goal} evidence for {claim_id}"`）。
4. 其余：回退到 `recommended_action`；再不行则按 `recommended_operation` + `operation_args`（含 `skip_curate`→`curate_document` 等别名）构造 `CapabilityAction`。
产出是 `CandidateAction{action, source, notes}`，再交给 Step 4 的 `route_decision` 做 endorse/correct/ignore。很多 shadow 模块（如 `VerificationShadow`）已在 artifact 里写入 `recommended_action`，此时 $$g$$接近恒等；只有 operation 仍是抽象意图时才真正“realize”。
1. 如果$$z_t^m=h_m(d_t)$$是一条完整轨迹，“same-state”从第二步开始就不存在了。
2. 完整轨迹会把 procedural capability 和新 observation 重新纠缠起来。
3. artifact 是模块能力的最小归因单位。
4. artifact 是“模块接口”，不是“第二个 Agent”。
5. 更容易兼容 black-box / API teacher。
Step 3：Information-Safe Gate
定义 artifact 的可见性约束：
$$\operatorname{evidence\_ids}(z_t^m)\subseteq\operatorname{observed\_ids}(s_t).$$
模块$$m$$在状态$$s_t$$生成的 artifact $$z_t^m$$中引用的所有证据，都必须已经被学生在当前状态真实看过。
并要求 artifact 中的 runtime metadata 必须来自真实$$B_t$$，而非 module 自行推测。
定义 hard mask：
$$M_t^m=M_{\text{visible}}M_{\text{schema}}M_{\text{executable}}M_{\text{module}}.$$
任一条件不满足，样本不进入蒸馏。
- $$M_{visible}$$检查有没有使用学生不可见的信息。
- $$M_{schema}$$检查 artifact 格式是否合法。
- $$M_{executable}$$检查建议动作在当前环境中是否真的可以执行。
- $$M_{module}$$检查这个建议是不是属于模块$$m$$的职责范围。
Step 4：Verified Decision Routing
定义：
$$\tilde a_t^m=
\begin{cases}
a_t^-, & \text{module endorses }a_t^- \\
a_t^m, & \text{module rejects }a_t^-\text{ and }a_t^m\text{ is verified} \\
\varnothing, & \text{otherwise.}
\end{cases}$$
Endorse
若：
$$h_m(d_t)\text{ agrees with }a_t^- \quad\land\quad V_m(d_t,a_t^-)=1,$$
则保留学生已有正确动作作为 label。此时强调当前学生已经进入正确行为区域，训练目标是提高该决策的稳定性。
Correct
若 module 拒绝学生动作，则获得$$a_t^+=a_t^m$$，并要求$$V_m(d_t,a_t^+)=1$$。
学生动作 $$a_t^-$$
模块诊断
verified target $$\tilde a_t$$
证据不足直接回答
unsupported claim
搜索直接支持该 claim 的来源
重复原 query
low information gain
加入缺失实体 / 时间条件改写 query
引用二手摘要页
primary source missing
打开原始来源
冲突证据下停止
unresolved conflict
搜索独立来源消歧
coverage 足够仍继续搜
marginal gain low
stop and answer
预算将耗尽仍宽泛检索
budget-risk
聚焦当前最关键缺口
Step 5：Action-Level Imitation
DAgger 对多轮 LLM Agent 的实验已经表明：在真实 visited state 上直接使用 expert action 的 supervised label 是一个强而简单的训练信号。
因此 SCOPE 默认采用 decision/action-level cross entropy：
$$\mathcal L_{\text{SDI}}
= - \sum_{t,m} w_t^m M_t^m
\log \pi_\theta(\tilde a_t^m\mid s_t).$$
只对该 turn 的 action span 计算 loss，不对 observation、Harness artifact、tool result 文本反向传播。这样有三个好处：
1. black-box compatible：只需要 teacher sampled action / module action，不需要 logits；
2. 避免低价值 token 蒸馏：DOPD 已验证 token supervision 非均匀，SCOPE进一步把粒度提升到“关键决策 span”；
3. 与目标一致：我们要学的是 search control decision，不是模仿 Harness 的自然语言格式。
Step 6：Capability-Bearing Decision
DOPD 的重要经验是：不要假设所有 privileged supervision 都等价；需要辨认真正携带能力差异的信号。SCOPE 在模块 / 决策级定义四个统计量。
Procedural purity $$P_m$$ 
来自 Step 0，衡量模块收益中有多少在不增加新事实信息的情况下仍存在。
Reliability $$U_m$$
$$U_m=
\frac{\#\{\text{module target passes verifier}\}}
{\#\{\text{module proposes target}\}}.$$
Local decision gain $$G_m$$
若 verifier 只有 0/1，则用“verified correction rate”近似即可。
Internalization $$\rho_m$$
在 held-out student states 上：
$$\rho_m=\mathbb E_{d_t}
\left[\mathbb I\big(\pi_\theta(d_t)\text{ agrees with }h_m(d_t)\big)\right].$$
可以用 action-type agreement、module-specific accuracy 或 target preference rate 实现。
动态样本/模块权重
没有标量 verifier 时：
$$w_t^m=P_mU_m(1-\rho_m).$$
Step 7：Shadow-first, Recovery-on-Demand
纯 same-state shadow guidance 有一个潜在问题：如果学生在第 5 步就错误停止，主 rollout 根本不会出现“正确继续搜索以后”的第 6–10 步状态。DAgger 通过 teacher/student mixture 和 teacher takeover 解决这一点，但 SCOPE 不希望让完整 teacher 长时间接管，因为这会：改变状态分布；混入多个 Harness capability；增加信息泄漏；使模块归因变差。因此采用局部恢复分支。当满足：
$$M_t^m=1,\qquad \delta_t^m>\tau_{\text{recover}},$$
且环境允许 fork / replay 时：
1. 主轨迹继续执行学生原动作$$a_t^-$$，保持 strict on-policy；
2. 从同一$$s_t$$ fork 一个辅助分支；
3. 只执行一次 verified corrective action $$a_t^+$$；
4. 从$$s_{t+1}^{+}$$起重新交给学生策略继续$$K$$步：
$$a_{t+k}\sim\pi_\theta(\cdot\mid s_{t+k}^{+}),\quad k=1,\ldots,K;$$
5. 在这些 recovery states 上继续做 same-state module labeling。
定义 recovery loss：
$$\mathcal L_{\text{rec}}
=-\omega_{\text{rec}}
\sum_{(s,a^*)\in\mathcal D_{\text{rec}}}
\log\pi_\theta(a^*\mid s),$$
其中建议$$0<\omega_{\text{rec}}<1$$，以避免 recovery branch 反客为主。
首版 toy experiment 可先不加 recovery；当 full rollout 明显存在 premature stop / dead-end 时再加入。
Step 8：Module Lifecycle——不是“蒸馏完就全部删掉 Harness”
对模块$$m$$，根据$$P_m,U_m,\rho_m$$以及真实 external dependency 将模块划分为：
1. Internalize 满足：$$P_m$$高；$$U_m$$高；$$\rho_m$$仍低。继续给予高训练权重。
2. Distilled / Retire 满足：$$P_m$$高；$$\rho_m>\tau_{\text{retire}}$$；移除模块后 held-out 性能无显著下降。则逐步降低该模块蒸馏频率，并在 inference 端退役。
3. Hybrid Runtime 例如 Evidence Graph、Budget Control：策略层已被模型学会；但真实状态存储、预算计数仍不可替代。因此保留轻量 runtime，仅去掉“认知决策”部分。
4. Runtime-only 例如：live search execution；exact tool execution；external factual verifier；deterministic truncation / accounting。这些模块不把“删除 Runtime”作为目标，只训练模型学习何时调用、如何调用、如何消费结果。
因此最终部署变为
Full Harness
   ↓ capability internalization
Hybrid Harness
   ↓ module retirement
Minimal Runtime + stronger model
最终优化目标
$$\mathcal L
=
\mathcal L_{\text{RL}}
+\lambda\mathcal L_{\text{SDI}}
+\gamma\mathcal L_{\text{rec}}
+\xi\mathcal L_{\text{stab}}.$$
其中，$$\mathcal L_{\text{RL}}$$表示最终任务 outcome reward；$$\mathcal L_{\text{SDI}}$$表示主 same-state verified decision imitation；$$\mathcal L_{\text{rec}}$$表示可选 recovery branch；$$\mathcal L_{\text{stab}}$$表示对旧策略的轻量 KL，防止少量局部标签导致策略漂移。
若当前任务 reward 不可靠，可以先只训练：
$$\mathcal L=\mathcal L_{\text{SDI}}+\xi\mathcal L_{\text{stab}},$$
先验证“可学习 Harness decision 是否能独立内化”，再叠加 RL。
训练算法
Input:
    student policy πθ
    minimal runtime Hmin
    typed harness modules {h1, ..., hM}
    local verifiers {V1, ..., VM}
    training tasks D

Stage 0: module distillability probe
    for each module m:
        estimate procedural share Pm
        classify m as procedural / hybrid / runtime-only

for each training iteration:

    1. Pure student rollout
        τ- ~ πθ under Hmin

    2. Select key decision states
        build DecisionState dt = ψ(st)

    3. Same-state shadow guidance
        for relevant module m:
            ztm = hm(dt)
            check visibility / schema / executability
            if invalid: mask sample

    4. Verified decision routing
        if module endorses student action at-:
            target ãt = at-
        else:
            realize candidate at+ from ztm
            if Vm(dt, at+) passes:
                target ãt = at+
            else:
                discard sample

    5. Capability weighting
        update Um, Gm, ρm
        wtm = Pm * Um * (1 - ρm) * φ(δtm)

    6. Optional recovery branch
        if correction is high-impact and environment is forkable:
            fork st
            execute at+ once
            let student continue K steps
            collect recovery states + local labels

    7. Optimize
        L = LRL + λ LSDI + γ Lrec + ξ Lstab

    8. Module lifecycle update
        if ρm is high and removal causes no held-out drop:
            reduce module sampling / mark module for retirement
        if module remains externally dependent:
            keep corresponding minimal runtime component
3. 注意事项
3.1 关键状态选择
不在所有 token 上运行 Harness。只选择“会改变环境分支或证据状态”的 decision points：
1. 生成 / 改写 query 前；
2. 选择搜索结果 / 打开文档前；
3. claim status 更新前后；
4. 冲突出现时；
5. verifier 调用决策点；
6. stop / answer 决策点；
7. 连续 query 高度相似时；
8. evidence coverage 长时间无提升时；
9. budget 即将进入关键阈值时。
这与 DOPD 的“关键 token 非均匀”结论一致，但 SCOPE 将粒度从 token 提升到 capability-bearing decision event。
3.2 Harness Artifact 设计
1. Evidence 模块输出
{
  "module": "evidence_state",
  "target_claim": "claim_2",
  "current_status": "unsupported",
  "missing_evidence_type": "primary_source",
  "recommended_operation": "search",
  "reason_code": "no_direct_support",
  "evidence_ids": ["obs_2", "obs_5"]
}
2. Verification 模块输出
{
  "module": "verification_control",
  "target_claim": "claim_4",
  "verification_status": "conflict",
  "conflicting_evidence_ids": ["obs_3", "obs_7"],
  "recommended_operation": "search_independent_source",
  "reason_code": "unresolved_conflict"
}
3. Budget 模块输出
{
  "module": "budget_control",
  "remaining_search_calls": 3,
  "evidence_coverage": 0.85,
  "estimated_information_gain": "low",
  "recommended_operation": "stop_and_answer",
  "reason_code": "coverage_sufficient"
}
Artifact 不应包含：学生未看到网页中的句子；external verifier 的隐藏结论文本；teacher 完整 CoT；Harness 后续完整 trajectory；仅为了复现某套 Harness prompt 的自由文本模板。
4. 实验设计
1. Benchmark 组合
Model：Qwen3-1.7B, Qwen2.5-7B-Instruct, Llama-3-8B-Instruct, Qwen3-30B-A3B-Instruct-2507.
Benchmark：BrowseComp Plus, HotpotQA FullWiki, LoHoSearch(OOD).
2. Baseline 至少包括：
- SFT on full Harness trajectories；
- GRPO；
- OPD；
- OPHSD-style full Harness self-distillation：用完整 Harness terminal context 作为 privileged teacher；
- DOPD-style selective OPD：若 teacher logits 可用，在相同 privileged setup 下实现；
- DAgger-style agent imitation：teacher/student turn-level mixture + expert label；
- Pure same-state expert labeling：无 capability weighting；
- Full SCOPE。
3. 主指标
最终任务指标：Answer Accuracy；Exact Match / F1；Task Success Rate。
证据指标：Supporting Document Recall；Citation Precision；Citation Entailment；Unsupported Answer Rate；Evidence Coverage。
搜索过程指标：Search Calls；Open Calls；Trajectory Length；Repeated Query Rate；Premature Stop Rate；Conflict Resolution Rate。
内化指标：对模块$$m$$定义：
$$\operatorname{Retention}_m = \frac{ R(\theta_{\text{trained}},H_{\text{minimal}}) - R(\theta_{\text{base}},H_{\text{minimal}}) }{ R(\theta_{\text{base}},H_{\text{full}}) - R(\theta_{\text{base}},H_{\text{minimal}}) }.$$
它衡量完整 Harness 原本带来的收益中，有多少在移除可学习模块后被模型保留。
Runtime reduction：被成功退役的认知模块数；inference token cost；Harness LLM calls；latency；minimal runtime 下性能保留比例。
4. 实验 todo list
- E0：Module Distillability Map
比较 M1 / M2 / M3：
module off
procedural-only module
full module
得到$$P_m$$，验证 Search Harness 的收益确实同时包含 procedural 与 informational 成分。
- E1：Full Harness distillation vs Same-State local distillation
比较：SFT on Harness trace；OPHSD-style full Harness context；Same-state local label；Same-state + information-safe gate。
重点看：fresh corpus；unseen facts；citation hallucination；action decision accuracy。
目标：证明 local artifact 不是工程简化，而是在 Search 场景中用于分离 procedural capability 与 privileged information 的必要机制。
- E2：为什么需要 Correct，而不只是 Endorse
比较：endorse-only；reject/mask only；corrective CE；pairwise corrective preference。
主方法建议先用 corrective CE，因为它与 DAgger 的 expert label 更直接、更稳定；pairwise 作为 ablation。
重点错误：premature answer；repeated query；missing primary source；unresolved conflict；over-search after sufficient evidence。
- E3：Capability Weighting 是否真的解决 privilege illusion
比较：uniform module weight；reliability only：$$U_m$$；reliability + internalization：$$U_m(1-\rho_m)$$；full：$$P_mU_m(1-\rho_m)$$。
重点观察：fresh corpus transfer；teacher/harness format perturbation；entropy / policy collapse；低 procedural-share 模块是否被过度蒸馏。
- E4：DAgger-style mixing vs Shadow-first Recovery-on-Demand
比较：pure student OPD；DAgger turn-level mixture；student-prefix → teacher completion；SCOPE shadow-only；SCOPE shadow + local recovery。
重点：success；teacher/Harness calls；recovery state coverage；deployment-state divergence；模块监督归因纯度。
目标：证明局部 recovery 可以获得 DAgger 的恢复收益，同时减少 full teacher takeover。
- E5：Black-box Teacher Compatibility
构建两类 teacher：
1. white-box local model，能返回 logits；
2. API / rule / retriever / verifier 混合 Harness，只能返回 sampled action / structured artifact。
比较 logit-OPD 与 action-level SCOPE。
目标：证明方法不依赖 teacher distribution，适合真实 Harness 系统。
- E6：Module Retirement / Minimal Runtime
测试：
1. Bare Model；
2. Minimal Executor；
3. Minimal Executor + hard verifier / state store；
4. Partially retired Harness；
5. Full Harness。
目标不是证明“所有 Harness 都可以删除”，而是得到一条 Pareto curve：
$$\text{Task Quality} \quad\text{vs.}\quad \text{Runtime Complexity / Cost}.$$
- Fresh-corpus / Cross-Harness 泛化
  - Fresh Corpus
    - BM25 → dense retriever；
    - train corpus → newly indexed corpus；
    - fixed source → new source distribution。
如果模型只记住 Harness 给出的 facts，性能会明显崩溃；如果学到程序性搜索能力，应继续知道何时 search / open / verify / stop。
  - Cross-Harness Representation
  训练时改变：JSON 字段顺序；reason code 命名；evidence renderer；context serialization；相同模块的不同实现。
  目标：证明模型不是在模仿 artifact 模板，而是在学习模块表达的决策能力。
