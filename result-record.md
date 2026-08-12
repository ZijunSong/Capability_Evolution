# SCAPE result-record

> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。

---

## 本轮总览（更新于 2026-08-12 15:55）

### Setting（双线）
| 线 | 机器 / repo | model | retrieval | Candidate A/B |
|---|---|---|---|---|
| **Canonical True-SCAPE（H20）** | 8×H20；`sync/h20-20260812` @ `a765290` | `/data/ppnm/models/Qwen2.5-7B-Instruct` | LOCAL_COMPAT plumbing | **待 H100 V2 Gate**（旧 A/B 已归档） |
| **归档 provisional（H20）** | 同上 | Qwen2.5-7B-Instruct | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool`（Gate S FAIL） |
| **H100** | 8×H100；shared tree | `pat-jj/harness-1` | local BM25 / offline stub | 新一轮 confirm/repl/real-influence **进行中（H20 侧未收到 imports）** |

### 进度板 — Canonical True-SCAPE（H20）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| Git sync `sync/h20-20260812` | **已完成（local+bundle）** | push 被 SSH pubkey 阻塞；`artifacts/git/scape-h20-20260812.bundle` |
| pytest + `preflight_scape.py` | **已完成** | 16 passed；preflight ok；`legacy_scope_path_used=false` |
| True same-state / tool-mask / HF tool-OPD | **已完成（plumbing）** | `scape/collection/same_state.py` + `scape/training/hf_tool_opd.py` |
| Part C 8-GPU smoke Group A P0–P3 | **已完成** | `outputs/true_scape_pipeline_smoke/group_a/` |
| Part C 8-GPU smoke Group B Q0–Q3 | **已完成** | 三 loss path 数值可区分；`LOSS_PATH_AUDIT.md` pass |
| Candidate V2 / Stage L–S–M | **阻塞：等 H100 imports** | `imports/h100_{1..4}/` 仅 `.gitkeep`；daemon 监控中 |

### 进度板 — 归档 provisional（勿与 canonical 混写）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| LOCAL_CAL64 LOO + 旧 A/B + Gate L/S | **已完成 / 归档** | Gate S FAIL；**不再扩 seed / Stage M** |

### 进度板 — H100（自 `result-record-from-h100.md` 同步）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| H100-1 Phase 0/1 contribution LOO | **已完成（local BM25）** / 官方 Chroma **阻塞** | 10 组件 n=200 errors=0；见下方 H100-1 表 |
| H100-2 replication + coalition | **已完成（frozen consolidation）** | 4 modules + 6 coalition rows；非原 10-component REPL200 全量 |
| H100-3 same-state influence | **已完成（offline INF_CAL64）** | 10 组件 × 256 states；deterministic stub |
| H100-1 × H100-3 quadrant map | **已完成** | `CONTRIBUTION_INFLUENCE_MAP.md`；四象限 |
| H100-2 placement stability | **已完成** | `PLACEMENT_STABILITY.md` |
| Harness-1 restore + vLLM smoke | **已完成（smoke）** | 9 shards；`/v1/models` 200 |
| 官方 BrowseComp+ Chroma eval | **阻塞** | 缺 `OPENAI_API_KEY` / `CHROMA_API_KEY` / `CHROMA_DATABASE` |
| H100-3 confirm/targeted 扩展 | **未开始** | `INF_CONFIRM128` 等未跑 |
| 官方 Chroma H100-1/2 全量 parity | **未开始/阻塞** | 不可用 local/offline 证据冒充 |

### 结论（一句话）
- **非 H100**：LOCAL_CAL64 + BM25+Qwen 下 A/B **不可 retirement**（Gate S FAIL）；Stage M 已停。
- **H100**：local/offline 贡献·复现·影响力地图已齐；强平衡候选为 `evidence_graph` / `chunk_neighbors`；**不可**据此宣称官方 Harness-1 Chroma parity 或 released-checkpoint retirement。下一步仍需官方凭证或换候选。

详细数字：非 H100 见 `## 2026-08-12 SCAPE non-H100 round final`；H100 见 `## 2026-08-12 SCAPE H100-1/2/3 synced status`。

---

## 2026-08-11 SCAPE repo_bootstrap

### Setting
- repo path: `/data/ppnm/Capability_Evolution/SCAPE`
- upstream Harness-1 pin: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- scope: code + tests + docs + scripts (no GPU experiments yet)
- model: unset (not launched)
- benchmark: unset

### Results
| metric | value |
|---|---:|
| pytest | 14 passed |
| experiments_started | 0 |

### Paired
- (none)

### Gate
UNRESOLVED

### Decision
完成 canonical repo + 测试全绿 + 推送 Github；暂不启动 Stage L/S/M 训练。

---

## 2026-08-11 LOCAL_CAL64 LOO aggregate + candidate select

### Setting
- path: `/data/ppnm/Capability_Evolution/SCAPE/outputs/local_cal64_loo`
- model: Qwen2.5-7B-Instruct (vLLM TP=1, CAL64 BM25 provisional)
- n_queries: 64 unique / job; quality gate unique≥64 & err_rate≤0.15
- jobs: full + 8 minus_* (9/9 quality-complete)

### Results
| metric | value |
|---|---:|
| quality_complete | 9/9 |
| Candidate A | auto_populate_first_search |
| Candidate B | verify_tool |
| placement_map | outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.md |
| selection_json | outputs/scape_prestage/CANDIDATE_SELECTION.json |

### Paired
- LOO contribution from full vs minus_* CAL64 rollouts
- influence values are provisional proxies pending real same-state influence probe

### Gate
PARTIAL — LOO aggregate done; Stage L scaffolding + dry_run distill started; real OPD data path not yet wired

### Decision
Proceed Stage L learnability for A=`auto_populate_first_search`, B=`verify_tool`. Prefer waiting was satisfied (9/9). Next: wire real reduced-harness same-state collection → tool-OPD training cells.

---

## 2026-08-11 Stage L B-verify provisional OPD L64_seed42

### Setting
- path: `outputs/stage_l/B_verify_opd_provisional/`
- stack: SCOPE `smoke_opd_vllm_hf` + `train_opd` (provisional LOCAL_CAL64; H100 not required)
- GPUs: 2–5 (TP=4 vLLM rollout → HF train)
- cell: L64 seed=42 · target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1

### Results
| metric | value |
|---|---:|
| smoke DONE | yes |
| smoke opd_loss | 0.0486 |
| L64 n_transitions | 64 |
| L64 epoch0 loss | 0.1220 |
| L64 opd_loss | 0.7293 |
| checkpoint | `L64_seed42/checkpoint.json` status=saved |

### Paired
- （当时）Collect A/B H_-m 未完成 → **后续已于 2026-08-12 完成 512**（见本轮总览）
- Next cell: L64_seed43 started on freed GPU2–5

### Gate
PARTIAL（条目当时）→ **已被本轮总览覆盖**：Gate L 后续 PASS；collect 已完成

### Decision
Record L64_seed42 metrics; advance seed43 on free GPUs. Do not stop for empty H100 imports.

---

## 2026-08-11 Stage L B-verify provisional OPD L200

### Setting
- path: `outputs/stage_l/B_verify_opd_provisional/`
- stack: SCOPE `train_opd` (provisional LOCAL_CAL64; H100 not required)
- cells: L200 seed42 (GPU2–5 TP4 :8769); L200 seed43 (GPU6–7 TP2 :8770)
- target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1

### Results
| metric | seed42 | seed43 |
|---|---:|---:|
| n_transitions | 200 | 200 |
| epoch0 loss | 0.1220 | 0.1296 |
| opd_loss | 0.7293 | 0.8308 |
| checkpoint | saved | saved |
| status | DONE | DONE |

### Paired
- Prior L64: s42 loss=0.122 / opd=0.729; s43 loss=0.119 / opd=0.860; s44 loss=0.130 / opd=0.988
- （当时）A/B collect → **后续已完成**（见本轮总览）

### Gate
PARTIAL（条目当时）→ **已被本轮总览覆盖**：held-out×2 + L200×3 已完成；Gate L PASS；closed-loop Gate S FAIL

### Decision
Record L200 seed42/43; free GPU2–7; start B L64 held-out (`--split test`) while collect continues.


## 2026-08-12 SCAPE non-H100 round final

> 覆盖并取代同日自动追加的 `non_h100_closed_loop_complete` / `non_h100_completion` 草稿（其中仍写 collect 进行中 / S2S3 proxy 的条目作废）。
> 状态：**本轮非 H100 主线已完成**；Stage M **不启动**。

### Setting
- repo: `/data/ppnm/Capability_Evolution/SCAPE`
- model: Qwen2.5-7B-Instruct（vLLM serve / HF train）
- retrieval: BM25 provisional（BrowseComp-Plus index）；**非**官方 Harness-1 Chroma
- benchmark: BrowseComp-Plus
- H100: unavailable — 全程不依赖 `imports/h100_*`
- Candidate A: `auto_populate_first_search` · OPD `target_module=evidence_state` · student=`ablate_auto_seed.yaml`
- Candidate B: `verify_tool` · OPD `target_module=verification` · student=`ablate_verification.yaml`
- teacher harness: `modules_full.yaml` / LOO full V8D mask
- Stage S eval: CAL64 `split=test` n=64；S0/S1=LOO；S2/S3=served `L64_seed42_hf/hf_model`
- H_-m collect: `split=train` limit=512 · mask 去掉对应组件
- output roots:
  - LOO: `outputs/local_cal64_loo/`
  - collect: `outputs/stage_l_hminus_data/`
  - B OPD/Gate L: `outputs/stage_l/B_verify_opd_provisional/` + `GATE_L_B.json`
  - A OPD: `outputs/stage_l/A_auto_opd_provisional/`
  - B four-grid: `outputs/stage_s/B_verify_fourgrid/`
  - A four-grid: `outputs/stage_s/A_auto_fourgrid/`
  - narrative: `outputs/NON_H100_FINAL_REPORT.md`

### Results

#### 状态汇总
| item | status | note |
|---|---|---|
| LOO 9/9 | **已完成** | quality-complete |
| Candidate select A/B | **已完成** | A score≈0.0072；B score≈0.0011 |
| B Gate L | **已完成 · PASS** | provisional SCOPE-OPD；非 full tool-token |
| A/B HF student ckpt | **已完成** | 可 vLLM 服务 |
| A/B H_-m collect 512 | **已完成** | A 512 uniq；B 512 uniq（834 lines w/ resume dups） |
| B Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.855；不可 retirement |
| A Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.536；不可 retirement |
| Stage M / Pareto | **未开始** | 按 auto-stop 规则停止 |
| H100 / Chroma 官方线 | **阻塞 / 未开始** | 本轮不依赖 |

#### B = verify_tool（closed-loop 四格，n_shared=64）
| cell | J (curated_recall) | C (tool-call proxy) | source |
|---|---:|---:|---|
| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
| S1 θ0+H_-verify | 0.0275 | 32.95 | LOO |
| S2 θ'+H_-verify | 0.0358 | 34.98 | closed-loop HF |
| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |

| metric | value |
|---|---:|
| CCR_m | 0.855 |
| HRR | 0.152 |
| Gate S verdict | **FAIL** |
| can_claim_retired | false |

#### A = auto_populate_first_search（closed-loop 四格，n_shared=64）
| cell | J | C | source |
|---|---:|---:|---|
| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
| S1 θ0+H_-auto | 0.0084 | 34.16 | LOO |
| S2 θ'+H_-auto | 0.0238 | 34.67 | closed-loop HF |
| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |

| metric | value |
|---|---:|
| CCR_m | 0.536 |
| HRR | 0.152 |
| Gate S verdict | **FAIL** |
| can_claim_retired | false |

#### Stage L（B，摘录）
| cell | status |
|---|---|
| L64 seeds 42/43/44 + L64_seed42_hf | DONE |
| L200 seeds 42/43/44 | DONE |
| L64 heldout seeds 42/43 | DONE |
| L200_seed45 / L512_seed42 | **未跑**（Gate S 已 FAIL，不再扩） |

### Paired
- S0/S1：同一 CAL64 query 集上 full vs minus 组件的 LOO paired quality
- S2/S3：同一 query 集上 θ'（OPD HF）vs θ0 的 closed-loop paired 评测
- B：去掉 verify 后缺口约 0.0097 J，OPD 恢复约 85%（CCR），但仍低于 S0，且 C 未实质下降
- A：去掉 auto_populate 后缺口更大；OPD 仅恢复约 54%，远未非劣于 S0

### Gate
- Gate L (B): **PASS**（provisional）
- Gate S (B): **FAIL**
- Gate S (A): **FAIL**
- Stage M: **不进入**
- Overall round: **COMPLETED (non-H100 line)** / retirement claim: **REJECTED**

### Decision
停止对 A/B 的 retirement 宣称与 Stage M；本轮 provisional 线归档。下一步只做其一：**(1)** 等 H100/官方 Chroma 后重跑 LOO→Gate；或 **(2)** 换下一候选组件（遵守「连续两失败则停救」）。不在当前 BM25+Qwen 线上继续扩 seed / multi-component。

---

## 2026-08-12 SCAPE H100-1/2/3 synced status

> 自 `result-record-from-h100.md` 同步的实验 setting / 结果 / 结论。
> 路径以 H100 机为准：`/mnt/songzijun/Capability_Evolution/SCAPE`（对应本机 `/data/ppnm/Capability_Evolution/SCAPE` 同源树）。
> 状态词汇：**已完成** / **进行中/阻塞** / **未开始**。

### Overall status
| Workstream | Todo target | Current status | Output / evidence | Notes |
|---|---|---|---|---|
| H100-1 Phase 0/1 | Harness-1 reproduction + 10-component LOO contribution map | **已完成（local BM25 compat）/ 进行中（official Chroma parity）** | `outputs/h100_1_contribution/{RUN_MANIFEST.json,STATUS_LIVE.md,COMPONENT_CONTRIBUTION.*,SHA256SUMS}` | Local BM25 compatibility contribution sweep finished for all 10 components, n=200, errors=0. Official Chroma Cloud eval blocked by missing credentials. |
| H100-2 | independent replication + coalition interaction | **已完成（frozen consolidation）/ 部分偏离原 10-component REPL200 plan** | `outputs/h100_2_replication_coalition/{RUN_MANIFEST.json,STATUS_LIVE.md,LOO_REPLICATION.csv,COALITION_INTERACTION.csv,REPLICATION_REPORT.md,PLACEMENT_STABILITY.md,SHA256SUMS}` | 4 replicated modules + 6 coalition rows, errors=0. No new training/retrieval. |
| H100-3 | same-environment-state policy influence map | **已完成（offline deterministic INF_CAL64）** | `outputs/h100_3_influence/{RUN_MANIFEST.json,STATUS_LIVE.md,INFLUENCE_BY_COMPONENT.*,INFLUENCE_PER_STATE.jsonl,H100_3_INFLUENCE_REPORT.md,SHA256SUMS}` | 64 queries × 4 states/query = 256 states/component; deterministic offline scorer. |
| H100-1 × H100-3 | contribution/influence quadrant map | **已完成** | `outputs/CONTRIBUTION_INFLUENCE_MAP.md` | 10 components → four quadrants. |
| Official Harness-1 serving | restore model and local vLLM smoke | **已完成（smoke）/ 进行中（official eval）** | `outputs/h100_1_official_vllm` | Restored from `harness-1.tar.gz`, 9 shards, vLLM smoke passed. |
| H100-3 confirm/targeted | `INF_CONFIRM128`, targeted influence/mining | **未开始** | none | Optional follow-ups not launched. |
| H100-1/2 official parity | Chroma-backed BrowseComp+ LOO/replication | **未开始/阻塞** | none beyond local/proxy | Requires official retrieval credentials. |

### H100-1 setting
- Run id: `h100_1_local_bm25_contribution_20260811`
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`（git `61f7741a…` dirty at manifest）
- Env: `/opt/bishop-harness/bin/python`；Python 3.11.6；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
- Backend: `local_bm25_compat`（**非**官方 Chroma Cloud）
- Split/seed: BrowseComp+ CAL200，seed 1101；smoke 1/5/20 亦 errors=0
- Decode: deterministic compatibility；无训练 / 无改权重
- Status: `n_expected=10`，`n_finished=10`，`remaining=0`，`errors=0`

### H100-1 results
| component | n | Δ curated | Δ trajectory | Δ final | Δ reward | Status |
|---|---:|---:|---:|---:|---:|---|
| subtractive_curation | 200 | +0.001556 | +0.000000 | +0.000000 | +0.000700 | 已完成 |
| importance_tagging | 200 | +0.001000 | +0.000000 | +0.000000 | +0.000450 | 已完成 |
| auto_populate_first_search | 200 | +0.000000 | +0.010298 | +0.000000 | +0.004634 | 已完成 |
| evidence_graph | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
| sentence_compress | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
| chunk_neighbors | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
| content_dedup | 200 | +0.000833 | +0.004583 | +0.000000 | +0.002438 | 已完成 |
| verify_tool | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
| token_budget_marker | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
| adaptive_rerank_instruction | 200 | -0.001250 | +0.000000 | +0.002917 | -0.000271 | 已完成 |

#### H100-1 conclusion
- LOO 仅对 **local BM25 compatibility** 路径完成。
- 综合 `Δ curated + Δ trajectory + Δ final` 最强：`auto_populate_first_search` → `content_dedup` → `adaptive_rerank_instruction` / `evidence_graph` / `chunk_neighbors`。
- `sentence_compress`、`verify_tool`、`token_budget_marker` 在本地质量指标上中性。
- **不可**用本 run 宣称官方 Harness-1 reproduction/parity。

### H100-2 setting
- Run id: `h100_2_replication_coalition_20260811`
- Env: `/opt/vllm-qwen3-1.7b/bin/python`；Python 3.12.13；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
- Replication input: `SCOPE/outputs/h100_2_module_utility`（fresh200 module-utility）
- Coalition input: `SCOPE/outputs/h100_2_exact_budget_factorial`（exact-budget factorial）
- Seed/decode: seed 42；temperature=0, top_p=1, do_sample=false
- Status: `n_expected=5`，`n_finished=5`，`errors=0`

### H100-2 results — replication
| module | ablated condition | n | Δ final-answer recall | Δ trajectory recall | Δ reward | paired final W/L/T | paired trajectory W/L/T | Status |
|---|---|---:|---:|---:|---:|---|---|---|
| context_budget | minus_context_budget | 200 | +0.003345 | -0.000671 | +0.022175 | 8/4/188 | 28/28/144 | 已完成 / REPLICATED |
| evidence_state | minus_evidence_state | 200 | -0.001786 | +0.002148 | +0.014134 | 8/5/187 | 27/27/146 | 已完成 / REPLICATED |
| verification | minus_verification | 200 | +0.010575 | +0.016813 | +0.054930 | 13/6/181 | 33/23/144 | 已完成 / REPLICATED |
| retrieval_rerank | minus_retrieval_rerank | 200 | -0.005571 | -0.007124 | -0.008528 | 6/6/188 | 25/30/145 | 已完成 / REPLICATED |

### H100-2 results — coalition
| model | budget | N | Q | QS | sequential interaction gap | interpretation | Status |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3_1p7b | 256 | 0.0300 | 0.0300 | 0.0200 | -0.0100 | diminishing_returns | 已完成 |
| qwen3_1p7b | 512 | 0.0300 | 0.0500 | 0.0400 | -0.0300 | diminishing_returns | 已完成 |
| qwen3_1p7b | 1024 | 0.0400 | 0.0400 | 0.0400 | +0.0000 | near_additive | 已完成 |
| qwen3_30b | 256 | 0.0100 | 0.0000 | 0.0000 | +0.0100 | super_additive | 已完成 |
| qwen3_30b | 512 | 0.0200 | 0.0200 | 0.0000 | -0.0200 | diminishing_returns | 已完成 |
| qwen3_30b | 1024 | 0.0300 | 0.0100 | 0.0100 | +0.0200 | super_additive | 已完成 |

#### H100-2 conclusion
- `verification` 是最清晰的稳定正复现模块（final / trajectory / reward 皆正）。
- `context_budget`、`evidence_state` 跨轴符号不一致 → placement/domain-sensitive。
- `retrieval_rerank` 两路 recall 皆负 → interaction/benchmark-sensitive。
- Coalition 多为 diminishing/near-additive，仅作交互备注，非强协同证据。
- 本 run ≠ 原 H100-2 10-component REPL200 全量计划；是 frozen SCOPE 输出的 consolidation。

### H100-3 setting
- Run id: `h100_3_influence_offline_cal64`
- Env: `/root/miniforge3/bin/python`；Python 3.13.13；offline scorer（无 torch/vLLM 依赖）
- Scale: INF_CAL64；64 queries/component；max 4 states/query；256 states/component；共 2560 per-state records
- Scorer: `deterministic_offline_stub`；无训练
- Status: `n_expected=10`，`n_finished=10`，`errors=0`
- H100 侧 A/B 候选：`subtractive_curation` / `importance_tagging`（与非 H100 线 A/B 不同）

### H100-3 results
| component | n_queries | n_states | event_support | normalized influence | Status |
|---|---:|---:|---:|---:|---|
| subtractive_curation | 64 | 256 | 256 | 0.134885 | 已完成 |
| importance_tagging | 64 | 256 | 256 | 0.107081 | 已完成 |
| verify_tool | 64 | 256 | 256 | 0.010138 | 已完成 |
| chunk_neighbors | 64 | 256 | 256 | 0.009933 | 已完成 |
| evidence_graph | 64 | 256 | 256 | 0.007756 | 已完成 |
| content_dedup | 64 | 256 | 256 | 0.007324 | 已完成 |
| auto_populate_first_search | 64 | 256 | 256 | 0.005417 | 已完成 |
| token_budget_marker | 64 | 256 | 256 | 0.005255 | 已完成 |
| sentence_compress | 64 | 256 | 256 | 0.003571 | 已完成 |
| adaptive_rerank_instruction | 64 | 256 | 256 | 0.001980 | 已完成 |

#### H100-3 conclusion
- 最高 same-state influence：`subtractive_curation`、`importance_tagging`。
- 中档：`verify_tool`、`chunk_neighbors`、`evidence_graph`、`content_dedup`。
- 最低：`adaptive_rerank_instruction`、`sentence_compress`、`token_budget_marker`。
- 本图有效为 offline deterministic same-state 产物；**不是** released Harness-1 logprob 枚举。
- `INF_CONFIRM128` / targeted 扩展 **未开始**。

### Cross-map conclusions（H100-1 + H100-3）
- Source: `outputs/CONTRIBUTION_INFLUENCE_MAP.md`
- Thresholds: contribution median `0.001611`；influence median `0.007540`

| quadrant | components | conclusion | Status |
|---|---|---|---|
| High Δ, High I | `evidence_graph`, `chunk_neighbors` | 冻结 local/offline 证据下最强平衡迁移候选 | 已完成 |
| High Δ, Low I | `auto_populate_first_search`, `content_dedup`, `adaptive_rerank_instruction` | 质量/运行时效应清晰，same-state 策略位移弱 | 已完成 |
| Low Δ, High I | `subtractive_curation`, `importance_tagging`, `verify_tool` | 改策略但本地质量提升弱；保留/移除前需复核 | 已完成 |
| Low Δ, Low I | `sentence_compress`, `token_budget_marker` | 本分析下的直接移除候选 | 已完成 |

### H100 lightweight / proxy 附注（同源记录）
| item | gate / key metric | Decision |
|---|---|---|
| H20 lightweight torch L/S/M/Pareto | PASS / LIGHTWEIGHT_TORCH_COMPLETE；best L_m≈0.962；S2 quality≈0.030 | 可作为 lightweight 产物；**非** official checkpoint retirement |
| qrel-backed pre-stage + H20 torch | PASS / LIGHTWEIGHT_TORCH_PROXY_COMPLETE；A/B Gate L PASS | 同上；官方 Chroma 评测仍独立 |
| Official model restore + vLLM smoke | PASS / MODEL_RESTORED_AND_VLLM_SMOKE_COMPLETE | 可继续接官方 eval；缺 3 个 secret vars |

### Final decision / next actions（H100）
- **已完成**：H100-1 local BM25 contribution；H100-2 frozen replication/coalition + placement；H100-3 offline influence；贡献×影响力四象限；Harness-1 restore + vLLM smoke；lightweight torch proxy L/S/M。
- **进行中/阻塞**：官方 BrowseComp+（缺 Chroma/OpenAI 凭证）。
- **未开始**：官方 Chroma H100-1/2 parity；`INF_CONFIRM128`；targeted influence/mining；released-checkpoint retirement 宣称。
- **禁止宣称**：不可把 local BM25 / offline / proxy 证据写成官方 Harness-1 Cloud/Chroma parity 或最终 retirement。

---

## Canonical True-SCAPE Line

### 2026-08-12 H20 Part A/B/C — sync + true tool-token OPD plumbing smoke

#### Setting
- canonical commit: `a765290` (`sync/h20-20260812`)
- upstream Harness-1 pin: `8ac4012167858f6478fb2a8fd840e4550e2af161` (submodule)
- model revision: `/data/ppnm/models/Qwen2.5-7B-Instruct`
- retrieval backend: LOCAL_COMPAT_ONLY (structured same-state plumbing; not official Chroma)
- component mask (plumbing only): `evidence_graph` minus vs full
- snapshot schema version: `scape_snapshot_v1`
- tool-mask version: `scape_tool_mask_v1`
- teacher backend: full-view prompt teacher-forced logprobs (HF)
- loss implementation: `scape.training.hf_tool_opd` (`tool_token_kl` / `action_ce` / `full_response_kl` / `offpolicy_matched`)
- legacy_scope_path_used: **false**
- GPUs: Group A=`0-3`, Group B=`4-7` (parallel)

#### Results
| cell | metric | value |
|---|---|---:|
| pytest | passed | 16 |
| preflight_scape | ok | true |
| P0 same-state audit | pass | true (16/16) |
| P1 dual-view mean div | D | 0.1679 |
| P2 tool-token KL micro-overfit | D_pre → D_post | 0.1679 → -0.0072 |
| P3 heldout32 | D | -0.0175 |
| Q0 action_ce mean_loss | | 0.9757 |
| Q1 full_response_kl mean_loss | | -0.0020 |
| Q2 offpolicy_matched mean_loss | | 0.1162 |
| Q3 syntax/serve | serve_rate / mask_pass | 1.0 / true |
| LOSS_PATH_AUDIT | pass | true |

#### Artifacts
- `outputs/true_scape_pipeline_smoke/{PIPELINE,TOOL_MASK,SAME_STATE,LOSS_PATH}_AUDIT.md`
- `artifacts/git/scape-h20-20260812.bundle`
- `GITHUB_SYNC_BLOCKED.md` (origin push blocked)

#### Gate
PLUMBING_PASS — not a scientific candidate Gate L/S claim.

#### Decision
True SCAPE training stack is wired and 8-GPU smoke is green. Stage L/S/M wait for H100 V2 imports under `imports/h100_{1..4}/` (watcher `scape_wait_h100` running). Old provisional auto_populate/verify line remains archived.
