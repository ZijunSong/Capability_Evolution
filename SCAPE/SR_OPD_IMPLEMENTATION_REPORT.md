# SR-OPD Implementation Report

> SCAPE 先将 Full-Harness Teacher 中异构的 Harness intervention 投影/编译为 Reduced-Harness Student 可实现的原生行为监督，再使用统一的 token-level SR-OPD CE 完成能力内化；无法可靠实现的 privileged behavior 被显式 reject，而不是通过 component-specific loss 强行拟合。

## 1. 修改文件

新增：

```text
SCAPE/scape/training/action_codec.py
SCAPE/scape/training/opd_events.py
SCAPE/scape/training/opd_realizability.py
SCAPE/scape/training/opd_projection.py
SCAPE/scape/training/opd_dataset.py
SCAPE/scape/training/sr_opd_loss.py
SCAPE/scripts/run_sr_opd_train.py
SCAPE/tests/test_opd_projection.py
SCAPE/tests/test_opd_realizability.py
SCAPE/tests/test_sr_opd_loss.py
SCAPE/tests/test_opd_no_privilege_leak.py
```

修改：

```text
SCAPE/scape/training/tool_mask.py
SCAPE/scape/training/hf_tool_opd.py
SCAPE/scape/training/train_tool_opd.py
SCAPE/scape/collection/same_state.py
SCAPE/scripts/run_opd_hf_save.py
SCAPE/tests/test_tool_mask.py
```

未覆盖用户未提交的 `SCAPE/scape/eval/projected_action.py` 与 0818 H20 实验脚本。

## 2. Deprecated legacy path

| Path | Status |
|---|---|
| `tool_token_kl` / `weighted_tool_token_kl` / `tool_name_only_kl` / `args_only_kl` / `full_response_kl` | Ablation / regression only |
| `--opd-mode legacy_same_action` | 显式回归旧 same-action OPD |
| `SCOPE/training/opd/*` via `scripts/run_opd_hf_save.py` | Legacy。`--opd-mode sr_opd` 会直接退出并指向新 launcher |
| Teacher-token KL | `--legacy-teacher-kl-weight`，默认 `0.0` |

正式路径：

```text
--opd-mode sr_opd
loss_impl=scape.training.hf_tool_opd:sr_opd_ce
projection_schema_version=scape_projection_v1
```

## 3. 新 schema

- `scape_snapshot_v2`：Student start snapshot + hash
- `scape_projection_v1`：`DIRECT / MACRO / SKIP / REJECT`
- Formal row：`teacher_events` + `projection` + `projected_steps` + `audit`
- `SKIP` 不产生 loss row；`REJECT` 不产生 loss row，但写入 audit

## 4. Project / Skip / Reject 实现位置

```text
StudentActionSpaceProjector.project_segment
  SCAPE/scape/training/opd_projection.py
```

Realizability gate：`SCAPE/scape/training/opd_realizability.py`  
Materialize（每一步推进 Student shadow）：`SCAPE/scape/training/opd_dataset.py`

## 5. 10-component handlers

| Component | Handler | 第一版决策 |
|---|---|---|
| `auto_populate_first_search` | curated-set delta → `curate(add/remove)` | MACRO |
| `subtractive_curation` | observable curated delta → `curate` | MACRO |
| `importance_tagging` | latent table SKIP；eviction delta → `curate` | SKIP then MACRO |
| `sentence_compress` | transform SKIP → downstream action | SKIP then DIRECT |
| `content_dedup` | extra dups OK；无 equivalence map 的 id remap REJECT | SKIP then DIRECT |
| `verify_tool` | 有全文 DIRECT；可 `review_docs` 则 MACRO；否则 REJECT | MACRO / REJECT |
| `evidence_graph` | graph event SKIP → downstream | SKIP then DIRECT |
| `token_budget_marker` | marker SKIP；外部 exact accounting REJECT | SKIP then DIRECT |
| `adaptive_rerank_instruction` | 同名 search ≠ DIRECT；结果不可复现 REJECT | REJECT |
| `chunk_neighbors` | neighbor expansion SKIP；可 read/review 则 MACRO/DIRECT | SKIP then DIRECT |

没有任何 handler 会 `NotImplementedError`。  
`importance_tagging()` / `verify()` 不会进入 reduced Student 工具空间。

## 6. Component-wise projection coverage（fixture smoke）

来自 `outputs/sr_opd_migration/projection_samples.json` 的代表性 segment，不是 100% 全量 Teacher trace。

| Component | Kind | Coverage on fixture |
|---|---|---|
| auto_populate_first_search | MACRO curate | 1/1 |
| subtractive_curation | MACRO curate | 1/1 |
| importance_tagging | SKIP + MACRO curate | delayed eviction 可投影 |
| sentence_compress | SKIP + DIRECT | 1/1 |
| content_dedup | SKIP + DIRECT | 1/1 |
| verify_tool | MACRO review_docs→curate | 可恢复时 1/1 |
| evidence_graph | SKIP + DIRECT search | 1/1 |
| token_budget_marker | SKIP + DIRECT | 1/1 |
| adaptive_rerank_instruction | REJECT | 0（故意，soundness） |
| chunk_neighbors | SKIP + DIRECT | 1/1 |

原则：soundness > coverage。adaptive rerank 在 Student search 无法恢复 Teacher docs 时显式 REJECT。

## 7. Reject distribution（当前单测 / fixture）

| Reason | 何时出现 |
|---|---|
| `TEACHER_ONLY_INFORMATION` | verify oracle 且 Student 不可访问所需 docs |
| `TRANSITION_NOT_REPRODUCIBLE` | adaptive rerank 同名 search、结果不同 |
| `DOC_NOT_ACCESSIBLE` | curate 引用 Student 当前拿不到的 id |
| `ILLEGAL_TOOL` | target 含 `verify` / `importance_tagging` |
| `INVALID_ARGUMENT_SCHEMA` | `curate(..., importance=...)` 且 mask 关闭 |
| `NO_SEMANTIC_ANCHOR` / `ANCHOR_HORIZON_EXCEEDED` | 扫描窗口内没有可投影 anchor |
| `MACRO_TOO_LONG` | 超过 `--projection-max-macro-actions` |
| `UNKNOWN_COMPONENT_EFFECT` | 未知 component，不 silent fallback |

## 8. Leakage audit

单测 `tests/test_opd_no_privilege_leak.py`：

- Teacher verify judgment / `teacher_only_observation` 不进入 Student prefix
- `importance_tagging=False` 时 importance table 不进入 prompt
- MACRO 每一步 `assert_no_future`
- Teacher / Student shadow 不 merge

硬门槛在 fixture 上：

```text
Student-Executable Target Rate = 100%（仅 DIRECT/MACRO materialized steps）
teacher-only observation leak rate = 0
future leakage rate = 0
```

## 9. Unit / integration tests

```text
tests/test_opd_projection.py
tests/test_opd_realizability.py
tests/test_sr_opd_loss.py
tests/test_opd_no_privilege_leak.py
tests/test_tool_mask.py   # 增补 verify mask
```

覆盖指令要求的 1–14 条核心 case（DIRECT、compress SKIP、auto_populate、subtractive、importance delayed eviction、verify MACRO、verify REJECT、dedup superset、adaptive rerank 不同 transition、verify 不输出、importance 不进 curate、no future leak、mask/weight 真正 backward、deterministic overfit）。

全量：`pytest -q` → **66 passed**。

真实 Harness-1 16–32 example/component smoke **尚未跑**。下一步应先审计 projection JSON，再扩训练，不能用 fixture 结果宣称真实 integration 完成。

## 10. Formal smoke 命令

投影 / audit（不走 SCOPE）：

```bash
python SCAPE/scripts/run_sr_opd_train.py \
  --opd-mode sr_opd \
  --component-id auto_populate_first_search \
  --projection-max-events 8 \
  --projection-max-macro-actions 3 \
  --reject-nonrealizable \
  --projection-jsonl path/to/teacher_rows.json \
  --projection-audit-path SCAPE/outputs/sr_opd_migration/audit.json
```

训练入口：

```bash
python -m scape.training.train_tool_opd \
  --opd-mode sr_opd \
  --component-id auto_populate_first_search \
  --n-samples 32 \
  --base-checkpoint <ckpt> \
  --out outputs/sr_opd_migration/train \
  --no-dry-run
```

回归旧 same-action OPD：

```bash
python -m scape.training.train_tool_opd --opd-mode legacy_same_action ...
```

## 11. 与旧 OPD 的行为差异

| 旧 | 新 |
|---|---|
| 同一 `response_text` / Teacher tokens 对齐 | Projector 生成 Student-legal `a*` |
| 每 component 不同 loss mask / KL | 统一 `sr_opd_ce` |
| `legal_tool_names()` 永远含 `verify` | `verify_tool=False` ⇒ `verify ∉ A_S` |
| MACRO 拼成一条 CE | 每步独立 `(xi^S, a)`，中间状态来自 Student shadow |
| SCOPE `HFTrainBackend` 可能 log 一套、backward 另一套 | logger loss 与 `loss.backward()` 同一 tensor |
| Teacher KL 是正式目标 | 默认关闭 |

## 12. 尚不能安全投影的 case

- Teacher verify 依赖 Student 原理上拿不到的 oracle 信息
- adaptive rerank 改变 retrieval transition，且第一版禁止 LLM 编造 query rewrite
- content_dedup 改变 canonical doc id 但没有经过验证的 duplicate map
- token_budget 依赖外部 exact accounting 而不是 Student 可从当前 sequence 推导的 budget
- 超过 `max_macro_actions=3` 或 `max_anchor_scan_events=8` 的长程等价
- 真实 Harness-1 live trace 中尚未审计的未知 side effect（会落到显式 REJECT，而不是猜一个 loss）
