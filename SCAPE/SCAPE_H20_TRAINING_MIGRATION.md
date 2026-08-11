# SCAPE H20 指令：Canonical Repo + Learnability + Capability Migration

## 任务定位

本机是 SCAPE 主开发/训练节点。

资源：

```text
8 × H20 144G
完整权限
允许长期运行
允许安装/修改环境
允许 Cursor 持续读取 .md 并循环执行
```

本机负责：

```text
0. 新建 canonical SCAPE repo
1. 集成并 pin Harness-1
2. Same-state snapshot/dual-view/tool-mask 基础设施
3. Local Learnability
4. Objective / same-state ablation
5. Single-component migration
6. Post-training component retirement
7. Multi-component annealing
8. 最终 quality-cost Pareto
```

**停止继续旧 SCOPE rollback / duplicate capability 工程。**

---

# 1. 新仓库

路径：

```bash
cd /data/ppnm/Capability_Evolution
mkdir -p SCAPE
cd SCAPE
```

初始化独立 git：

```bash
git init
git checkout -b main
```

建立：

```text
external/harness-1
scape/
configs/
scripts/
tests/
manifests/
outputs/
docs/
```

推荐把 upstream Harness-1 pin 成 submodule 或固定 commit。

记录：

```text
upstream repo
upstream commit
model checkpoint revision
dataset revision
retrieval/index revision
CUDA
driver
torch
vllm
transformers
uv lock hash
```

生成：

```text
docs/ENVIRONMENT.md
docs/UPSTREAM_PROVENANCE.md
```

---

# 2. 旧 SCOPE 只读迁移

允许从：

```text
/data/ppnm/Capability_Evolution/SCOPE
```

复制通用工具，但必须逐文件人工/agent审查。

可以移植：

```text
manifest freezing
stable hash split
paired W/L/T
paired bootstrap
SHA256SUMS
GPU launcher
resume/orphan cleanup
STATUS_LIVE.md writer
result-record updater
```

禁止移植到主方法：

```text
P_m
KEEP/SKIP
rollback classifier
O7
ModuleRetirementGate
Information-Safe Gate
capability-specific artifact schemas
```

把历史结果摘要写：

```text
docs/LEGACY_SCOPE_NOTES.md
```

只保存“经验/负结果”，不形成代码依赖。

---

# 3. 先建立自动测试

必须通过：

```text
test_component_mask_only_changes_target_flag
test_snapshot_no_future_information
test_snapshot_roundtrip_hash
test_dual_view_same_snapshot
test_full_teacher_does_not_step_environment
test_tool_name_span
test_argument_key_span
test_argument_value_span
test_end_search_span
test_reduced_rollout_owns_state_distribution
test_full_vs_minus_replay_parity
test_metric_pairing_by_query_id
```

任何训练前：

```text
pytest -q
```

全绿。

---

# 4. 输入 H100 Pre-stage 结果

约定同步目录：

```text
imports/h100_1/
imports/h100_2/
imports/h100_3/
```

读取：

```text
COMPONENT_CONTRIBUTION.json
LOO_REPLICATION.csv
COALITION_INTERACTION.csv
INFLUENCE_BY_COMPONENT.csv
```

如果还没有：

不要等待。

本机用：

```text
LOCAL_CAL64
```

跑最小：

```text
Full / -m
+
same-state I_name
```

只用于 provisional ranking。

正式 H100 结果同步后重新排序。

---

# 5. Candidate Selector

生成：

```text
outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.csv
outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.md
outputs/scape_prestage/CANDIDATE_SELECTION.json
```

只选：

```text
Candidate A
Candidate B
```

首轮最多两个。

排序：

```text
Contribution
×
Influence above null
×
semantic-decision fraction
÷
runtime cost to keep
```

但不要硬写成论文最终公式；这是实验 scheduler heuristic。

强制排除作为“完全内化目标”的：

```text
retrieval executor
exact accounting
persistent store
hard budget enforcement
cheap deterministic runtime checks
```

`content_dedup` 若表现好，也默认先作为 Runtime/Hybrid 对照，不作为 Candidate A。

---

# 6. Stage L：Micro-Learnability

数据：

```text
Candidate A:
  512
  2k
  8k

Candidate B:
  512
  2k
  8k
```

所有数据必须：

```text
来自 H_-m 下 student on-policy rollout
保存 xi_t
从同一 xi_t render full/minus
teacher 不执行未来 action
```

micro-distillation 首版：

```text
tool-token KL
+
light anchor KL
```

先不加入 RL。

teacher：

```text
EMA / lagged copy
```

固定具体策略并写 manifest，不允许不同 seed 偷换 teacher。

---

# 7. Stage L：8 卡并行队列

## GPU 0 — Candidate A Learnability seed42

顺序：

```text
A-L512
A-L2K
A-L8K
held-out divergence eval
```

每一级从同一个 base checkpoint 独立训练，不能把 512 checkpoint 继续训成 2k 后再声称 sample scaling。

如果要做 cumulative curriculum，另起实验名。

---

## GPU 1 — Candidate A Learnability seed43

同 GPU0：

```text
A-L512
A-L2K
A-L8K
held-out eval
```

---

## GPU 2 — Candidate B Learnability seed42

```text
B-L512
B-L2K
B-L8K
held-out eval
```

---

## GPU 3 — Candidate B Learnability seed43

同上。

---

## GPU 4 — Same-State Action CE baseline

针对 Candidate A：

```text
2K same-state sampled teacher tool-call CE
8K same-state sampled teacher tool-call CE
```

回答：

```text
logit/token OPD 是否优于 sampled action imitation？
```

---

## GPU 5 — Full-Response OPD baseline

Candidate A：

```text
2K
8K
```

teacher/student state 与主方法一致，但 loss mask 覆盖完整 assistant response，而非 tool span。

回答：

```text
tool-call-only mask 是否更干净？
```

---

## GPU 6 — Off-policy Harness Trace baseline

Candidate A：

```text
Full Harness 独立 trajectory
→ SFT / trajectory distillation
```

严格与主方法使用相同 task split 和近似相同 update token budget。

回答：

```text
为什么必须 student-state same-environment-state？
```

---

## GPU 7 — One-shot Full→Slim baseline

Candidate A：

```text
从第一步训练就固定 H_-m
不做 component availability curriculum
same-state tool-token KL
```

回答：

```text
Harness annealing/dropout 是否必要？
```

---

# 8. Learnability Gate

聚合：

```text
D_pre
D_post_512
D_post_2k
D_post_8k
L_m
tool_name agreement
argument-token NLL/KL
invalid tool rate
```

Candidate 进入 full migration，至少满足：

```text
1. held-out divergence 明确下降
2. 两 seed 方向一致
3. 2k/8k 不比 512 系统性恶化
4. invalid tool call 不上升
5. 没有明显 base capability collapse
```

若 A 失败：

```text
不要救 objective 超过一轮
```

最多允许：

```text
uniform KL
→ teacher-confidence weighting
```

仍失败则 Candidate A 标：

```text
CURRENTLY_NOT_LEARNABLE
```

切 Candidate B/C。

不要重演 rollback 线连续多轮救火。

---

# 9. Stage S：Single-Component Migration

优先对通过 Gate L 的 Candidate A。

必须有四格：

```text
S0 theta0 + H_full
S1 theta0 + H_-A
S2 theta_A + H_-A
S3 theta_A + H_full
```

训练主方法 V0：

```text
student rollout under H_z
same-state full-view teacher
tool-token KL
anchor KL
component dropout / annealing
```

第一轮先做 distillation-only causal proof。

若：

```text
S2 接近 S0
```

再从 best checkpoint 启动：

```text
RL + tool-token OPD
```

追求真正：

```text
S2 > S0
```

---

# 10. Stage S：8 卡并行队列

完成 Learnability Gate 后重排 GPU。

## GPU 0

```text
SCAPE-A distill-only seed42
→ four-grid eval
→ post-training N_A_post / CCR_A
```

## GPU 1

```text
SCAPE-A distill-only seed43
→ four-grid eval
```

## GPU 2

```text
SCAPE-A distill-only seed44
→ four-grid eval
```

## GPU 3

若 Candidate B 通过 Gate L：

```text
SCAPE-B distill-only seed42
```

否则：

```text
SCAPE-A + teacher-confidence weighting seed42
```

## GPU 4

若 B 通过：

```text
SCAPE-B seed43
```

否则：

```text
SCAPE-A name-only KL ablation
```

## GPU 5

若 B 通过：

```text
SCAPE-B seed44
```

否则：

```text
SCAPE-A args-only / name+args ablation
```

## GPU 6

```text
SCAPE-A best configuration
+ RL
seed42
```

只有 distill-only 已证明 positive compensation 才启动 RL。

## GPU 7

```text
SCAPE-A best configuration
+ RL
seed43
```

---

# 11. Stage S Gate

对 A/B 分别计算：

```text
CCR_m
N_m_post
Harness context token reduction
tool-call change
latency
state operation reduction
quality
paired bootstrap 95% CI
```

Strong pass：

```text
J(theta', H_-m) > J(theta0, H_full)
AND
C(H_-m) < C(H_full)
```

Acceptable pass：

```text
J(theta', H_-m) non-inferior
AND
C(H_-m) materially lower
```

Fail：

```text
只恢复 teacher agreement
但闭环 task quality 没补回来
```

这种不能写“component retired”。

---

# 12. Stage M：Multi-Component Annealing

只有至少一个 Candidate 通过 Stage S。

如果 A/B 都通过：

比较：

```text
M0 sequential A -> B
M1 sequential B -> A
M2 joint coalition dropout A+B
M3 random component dropout
M4 pre-stage-guided annealing
```

优先只做 2–3 个 seed，不要一开始 10 个 module 全关。

---

# 13. Stage M：GPU 队列

## GPU 0

```text
A->B sequential seed42
```

## GPU 1

```text
A->B sequential seed43
```

## GPU 2

```text
B->A sequential seed42
```

## GPU 3

```text
joint A+B dropout seed42
```

## GPU 4

```text
joint A+B dropout seed43
```

## GPU 5

```text
random dropout control seed42
```

## GPU 6

```text
pre-stage-guided annealing seed42
```

## GPU 7

```text
post-training runtime mask sweep
+
Pareto frontier
```

若只有 A 通过：

不要强行造 B。

改成：

```text
GPU0-2: A 三 seed
GPU3: confidence weighting
GPU4: name/args weighting
GPU5: same-state CE
GPU6: off-policy trace
GPU7: runtime recomposition sweep
```

---

# 14. 最终 Runtime Recomposition

训练后对所有候选 mask 重新跑：

```text
theta0 + H_z
theta' + H_z
```

构造：

```text
Task Quality vs Runtime Cost
```

至少 cost 维度：

```text
enabled cognitive components
rendered context tokens
state serialization tokens
extra Harness LLM calls
tool calls
latency
memory/state operations
wall-clock
```

最终主表：

```text
Original: theta0 + H_full
No-train removal: theta0 + H_slim
Trained full: theta' + H_full
SCAPE: theta' + H_slim
```

---

# 15. 必做消融优先级

按重要性：

```text
P0 same-state vs off-policy Harness trace
P0 tool-token KL vs sampled action CE
P0 Full->Slim one-shot vs annealing
P0 no-train removal
P1 full-response KL
P1 tool-name only vs name+args
P1 uniform vs teacher-confidence
P2 student uncertainty
P2 step-divergence weighting
P2 coalition-aware schedule
```

不要让 selective weighting 抢走主方法。

---

# 16. OOD / Robustness

主闭环成立后再做：

```text
held-out Harness-1 benchmark
fresh query split
renderer field-order perturbation
budget marker paraphrase
```

tool schema rename 只作为附录 stress test，不做主实验。

---

# 17. 每张卡的运行纪律

每个 GPU 都必须：

```text
独立 output root
独立 pid
独立 stdout/stderr log
独立 RUN_MANIFEST
可 resume
```

Agent 每轮检查：

```text
n_expected
n_finished
errors
last progress timestamp
GPU process
vLLM orphan
disk usage
checkpoint integrity
```

发现卡死：

```text
先保存 manifest/status
只杀本 run 的 orphan
resume missing shard
禁止删除已有 completed shard
```

---

# 18. result-record.md 更新格式

每完成一个 stage，追加：

```markdown
## YYYY-MM-DD SCAPE <stage>

### Setting
- repo commit:
- upstream Harness-1 commit:
- model:
- benchmark:
- split:
- component:
- harness mask:
- seed:
- decode:
- output:

### Results
| metric | value |
|---|---:|

### Paired
...

### Gate
PASS / FAIL / UNRESOLVED

### Decision
下一步只写一个明确动作。
```

---

# 19. 最重要的自动停止规则

1. **一个 candidate micro-learnability 连续两个合理 objective 都失败：停止救。**
2. **没有 Contribution + Influence，不启动训练。**
3. **只有 local KL 改善、没有 closed-loop compensation：不扩 seed。**
4. **single-component 尚未成立：不做 multi-component。**
5. **deterministic runtime anchor：不为删而删。**
6. **不要再把一个负方向连续迭代成旧 rollback 式长期黑洞。**

---

# 20. 本轮最理想产物

```text
Figure A:
Contribution Δ vs Influence I
bubble = Learnability L
category = Retire / Hybrid / Runtime

Figure B:
512 / 2k / 8k Learnability curve

Table:
single-component four-grid

Figure C:
before/after Quality-Cost Pareto frontier

Table:
Original theta0+Hfull
No-train theta0+Hslim
Trained theta'+Hfull
SCAPE theta'+Hslim
```

只要 Figure A + single-component four-grid 成立，SCAPE 就已经从旧 SCOPE 的 capability-specific toy line 转成了一个真正可写论文的 system-level capability migration experiment。
