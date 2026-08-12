# SCAPE result-record

> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。

---

## 本轮总览（更新于 2026-08-12）

### Setting（双线）
| 线 | 机器 / repo | model | retrieval | Candidate A/B |
|---|---|---|---|---|
| **非 H100（H20）** | 8×H20；`/data/ppnm/Capability_Evolution/SCAPE` | `/data/ppnm/models/Qwen2.5-7B-Instruct` | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool` |
| **H100** | 8×H100；`/mnt/songzijun/Capability_Evolution/SCAPE` | `pat-jj/harness-1`（已 restore + vLLM smoke） | local BM25 compat / offline stub（**非**官方 Chroma） | A=`subtractive_curation`；B=`importance_tagging` |

### 进度板 — 非 H100（H20 provisional）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| Repo bootstrap + pytest | **已完成** | 14 passed；代码在 umbrella `main/SCAPE` |
| LOCAL_CAL64 LOO 9/9 + 候选选择 | **已完成** | A/B 选出；`outputs/local_cal64_loo/`、`CANDIDATE_SELECTION.json` |
| A/B H_-m collect train-512 | **已完成** | A uniq=512；B uniq=512（jsonl 含 resume 重复行）；`stage_l_hminus_data/` |
| B Stage L OPD（L64×3 + L200×3 + heldout×2） | **已完成**（provisional） | `GATE_L_B.json` **pass=true** |
| B L64 HF 可服务权重 | **已完成** | `.../B_verify_opd_provisional/L64_seed42_hf/hf_model` |
| A L64 HF OPD + 权重 | **已完成** | `.../A_auto_opd_provisional/L64_seed42_hf/hf_model`；loss≈0.122 |
| B Stage S closed-loop 四格 | **已完成** | 真实 S2/S3（非 proxy）；**Gate S = FAIL** |
| A Stage S closed-loop 四格 | **已完成** | 真实 S2/S3；**Gate S = FAIL** |
| Stage M / Pareto / retirement 宣称 | **未开始（停止）** | 单组件 Gate S 未过 → 不进 multi-component |
| 真 SCAPE same-state tool-token OPD | **未完成** | LOO 无完整 ξ_t dump；Gate L 仅为 SCOPE-OPD 代理路径 |
| GPU 实验进程 | **空闲** | 相关 vLLM/rollout/completion loop 已停 |

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

## 2026-08-12 SCAPE H100-1/2/3/4 synced status

> 自 `result-record-from-h100.md` 同步的实验 setting / 结果 / 结论。
> 路径以 H100 机为准：`/mnt/songzijun/Capability_Evolution/SCAPE`（对应本机 `/data/ppnm/Capability_Evolution/SCAPE` 同源树）。
> 状态词汇：**已完成** / **进行中/阻塞** / **未开始**。

### Overall status
| Workstream | Todo target | Current status | Output / evidence | Notes |
|---|---|---|---|---|
| H100-1 Phase 0/1 | Harness-1 reproduction + 10-component LOO contribution map + fresh confirm | **已完成（local BM25 compat）/ 官方 Chroma 仍阻塞** | `outputs/h100_1_contribution/{RUN_MANIFEST.json,STATUS_LIVE.md,COMPONENT_CONTRIBUTION.*,SHA256SUMS}`；`outputs/h100_1_contribution_confirm/{RUN_MANIFEST.json,STATUS_LIVE.md,CONTRIBUTION_CONFIRM.*,RUNTIME_COST_CONFIRM.csv,SHA256SUMS}` | Local BM25 compatibility contribution sweep finished for CAL200 and CONFIRM400, both n=200/400, errors=0. 官方 Chroma Cloud eval 仍受凭证缺失限制。 |
| H100-2 | independent replication + coalition interaction | **已完成（independent REPL200_V2）** | `outputs/h100_2_independent_repl/{RUN_MANIFEST.json,STATUS_LIVE.md,LOO_REPLICATION_V2.*,COALITION_V2.csv,PLACEMENT_STABILITY_V2.md,CROSS_SPLIT_COMPARISON.md,SHA256SUMS}` | 真正的 query-disjoint `BCP_REPL200_V2`，full + 10 LOO + replay parity + 4 coalition，errors=0。 |
| H100-3 | same-environment-state policy influence map + confirm smoke | **已完成（offline INF_CAL64 + qrel-backed real scorer smoke）** | `outputs/h100_3_influence/{...}`；`outputs/h100_3_influence_qrel/{RUN_MANIFEST.json,STATUS_LIVE.md,INFLUENCE_BY_COMPONENT.*,TOP_CANDIDATES_FOR_CONFIRM.json,SHA256SUMS}`；`outputs/h100_3_real_influence_smoke/{...}` | Offline deterministic map 已完成；qrel-backed HF scorer smoke 也通过，说明本地 released Harness-1 模型可用于真实 continuation scorer。 |
| H100-4 | CONFIRM128 real-model influence confirmation + targeted event / null robustness | **已完成** | `outputs/h100_4_influence_confirm/{PREFLIGHT.md,PRESTAGE_EVIDENCE_TABLE.*,REAL_INFLUENCE_CONFIRM_BY_COMPONENT.*,REAL_INFLUENCE_CONFIRM_PER_STATE.jsonl,NULL_CONTROL_REPORT.md,SCORER_PARITY.md,SNAPSHOT_REPLAY_AUDIT.md,CANDIDATE_RECOMMENDATION_FOR_H20.*,RUN_MANIFEST.json,STATUS_LIVE.md,SHA256SUMS}` | 3-way GPU shard 执行完成：`subtractive_curation` / `importance_tagging` / `evidence_graph`，每路 128 queries × 4 states/query。 |
| H100-1 × H100-3 × H100-4 | contribution/influence/confirm handoff | **已完成** | `outputs/CONTRIBUTION_INFLUENCE_MAP.md`；`outputs/h100_4_influence_confirm/PRESTAGE_EVIDENCE_TABLE.md` | 证据链已从 local/offline 过渡到 real HF scorer CONFIRM128。 |
| Official Harness-1 serving | restore model and local vLLM smoke | **已完成（smoke）/ 进行中（official eval）** | `outputs/h100_1_official_vllm` | Restored from `harness-1.tar.gz`, 9 shards, vLLM smoke passed. |
| Official Harness-1 / Chroma parity | Chroma-backed BrowseComp+ LOO/replication | **未开始/阻塞** | none beyond local/proxy | 仍需要官方 retrieval credentials；本轮未把 local BM25 或 HF scorer 结果冒充为官方 Chroma parity。 |

### H100-1 setting
- Run ids: `h100_1_local_bm25_contribution_20260811`、`h100_1_confirm400_20260812`
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`
- Env: `/opt/vllm-qwen3-1.7b-harness/bin/python`；Python 3.12.13；torch 2.10.0+cu128；vLLM 0.19.1；8×H100
- Backend: `local_bm25_compat`（**非**官方 Chroma Cloud）
- Split/seed: BrowseComp+ CAL200 seed 1101；CONFIRM400 seed 1102；smoke 1/5/20 亦 errors=0
- Decode: deterministic compatibility；无训练 / 无改权重
- Status: `n_expected=10` 与 `n_expected=11` 均完成，`errors=0`

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

### H100-1 confirm results
| component | n | Δ curated | Δ trajectory | Δ final | Δ reward | Status |
|---|---:|---:|---:|---:|---:|---|
| subtractive_curation | 400 | +0.002192 | +0.000000 | +0.000000 | +0.000987 | 已完成 |
| importance_tagging | 400 | +0.001844 | +0.000000 | +0.000000 | +0.000830 | 已完成 |
| auto_populate_first_search | 400 | +0.000000 | +0.008613 | +0.000000 | +0.003876 | 已完成 |
| evidence_graph | 400 | +0.000000 | +0.004092 | +0.000000 | +0.001841 | 已完成 |
| chunk_neighbors | 400 | +0.000000 | +0.004092 | +0.000000 | +0.001841 | 已完成 |
| content_dedup | 400 | +0.004171 | +0.007324 | +0.000000 | +0.005760 | 已完成 |
| adaptive_rerank_instruction | 400 | +0.001357 | +0.000000 | +0.002357 | +0.001059 | 已完成 |

#### H100-1 conclusion
- CAL200 与 CONFIRM400 都只对 **local BM25 compatibility** 路径完成。
- `auto_populate_first_search`、`content_dedup`、`evidence_graph`、`chunk_neighbors` 在两个 split 上都保持正向或近正向贡献。
- `sentence_compress`、`verify_tool`、`token_budget_marker` 在 confirm 上仍接近中性。
- **不可**把本 run 宣称为官方 Harness-1 Cloud/Chroma parity。

### H100-2 setting
- Run id: `h100_2_independent_repl_20260812`
- Env: `/opt/vllm-qwen3-1.7b-harness/bin/python`；Python 3.12.13；torch 2.10.0+cu128；vLLM 0.19.1；8×H100
- Replication input: `BCP_REPL200_V2`，seed 2203，query-disjoint 于 H100-1 CAL200/CONFIRM400/H20 CAL64
- Backend: `local_bm25_compat`；`LOCAL_COMPAT_ONLY=true`
- Status: `n_expected=16`，`n_finished=16`，`errors=0`

### H100-2 results — replication
| component | n | Δ final-answer recall | Δ trajectory recall | Δ reward | paired final W/L/T | paired trajectory W/L/T | Status |
|---|---:|---:|---:|---:|---|---|---|
| subtractive_curation | 200 | +0.000000 | +0.000000 | +0.002006 | 0/0/200 | 2/0/198 | 已完成 / REPLICATED |
| importance_tagging | 200 | +0.000000 | +0.000000 | -0.000175 | 0/0/200 | 1/0/199 | 已完成 / REPLICATED |
| auto_populate_first_search | 200 | +0.000000 | +0.010298 | +0.004634 | 0/0/200 | 8/0/192 | 已完成 / REPLICATED |
| evidence_graph | 200 | +0.000000 | +0.001667 | +0.000750 | 0/0/200 | 1/0/199 | 已完成 / REPLICATED |
| sentence_compress | 200 | +0.000000 | +0.000000 | +0.000000 | 0/0/200 | 0/0/200 | 已完成 / FLAT |
| chunk_neighbors | 200 | +0.000000 | +0.001667 | +0.000750 | 0/0/200 | 1/0/199 | 已完成 / REPLICATED |
| content_dedup | 200 | +0.000000 | +0.004583 | +0.002438 | 0/0/200 | 3/0/197 | 已完成 / REPLICATED |
| verify_tool | 200 | +0.000000 | +0.000000 | +0.000000 | 0/0/200 | 0/0/200 | 已完成 / FLAT |
| token_budget_marker | 200 | +0.000000 | +0.000000 | +0.000000 | 0/0/200 | 0/0/200 | 已完成 / FLAT |
| adaptive_rerank_instruction | 200 | +0.000000 | +0.000000 | +0.000771 | 2/0/198 | 1/0/199 | 已完成 / REPLICATED |

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
- `auto_populate_first_search`、`content_dedup`、`evidence_graph`、`chunk_neighbors`、`adaptive_rerank_instruction` 在独立 fresh split 上保留正向或近正向效应。
- `verify_tool`、`sentence_compress`、`token_budget_marker` 更接近平坦控制项。
- Coalition 多为 diminishing/near-additive，仅作交互备注，非强协同证据。
- 这条 `BCP_REPL200_V2` 是 0812 指定的独立复现轨道，补足了旧 consolidated H100-2 的不足。

### H100-3 setting
- Run ids: `h100_3_influence_offline_cal64`、`h100_3_influence_qrel_cal64`、`h100_3_real_influence_hf_real_inf64`
- Offline env: `/root/miniforge3/bin/python`；Python 3.13.13；deterministic offline scorer
- Qrel/HF env: `/opt/vllm-qwen3-1.7b-harness/bin/python`；HF continuation scorer on local released Harness-1 checkpoint
- Scale: offline INF_CAL64；qrel-backed smoke 8 queries × 2 states；real scorer smoke 1 query × 1 state；qrel-backed top-3 confirm smoke 128 queries × 4 states
- Status: offline / smoke / confirm all `errors=0`
- H100 侧 top candidates: `subtractive_curation`、`importance_tagging`、`evidence_graph`

### H100-3 results
| component | offline normalized influence | qrel-backed I_name_normalized | confirm gate | Status |
|---|---:|---:|---|---|
| subtractive_curation | 0.134885 | 0.121553 | REAL_INFLUENCE_POSITIVE | 已完成 |
| importance_tagging | 0.107081 | 0.087933 | REAL_INFLUENCE_POSITIVE | 已完成 |
| evidence_graph | 0.007756 | 0.041455 | REAL_INFLUENCE_POSITIVE | 已完成 |
| chunk_neighbors | 0.009933 | — | smoke only | 已完成 |
| verify_tool | 0.010138 | — | smoke only | 已完成 |
| content_dedup | 0.007324 | — | smoke only | 已完成 |
| auto_populate_first_search | 0.005417 | — | smoke only | 已完成 |
| token_budget_marker | 0.005255 | — | smoke only | 已完成 |
| sentence_compress | 0.003571 | — | smoke only | 已完成 |
| adaptive_rerank_instruction | 0.001980 | — | smoke only | 已完成 |

#### H100-3 conclusion
- offline 与 qrel-backed real scorer 都指向同一批强候选：`subtractive_curation`、`importance_tagging`、`evidence_graph`。
- HF continuation smoke 证明本地 released Harness-1 checkpoint 可用于真实 logprob continuation scoring。
- `INF_CONFIRM128` 由 H100-4 接续完成。

### H100-4 setting
- Run id: `h100_4_influence_confirm`
- Model: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`
- Scorer: HF continuation logprob scorer（由 `run_h100_3_real_influence_hf.py` 提供）
- Split: `REAL_INF_CONFIRM128`，seed 4404，n=128，query-disjoint 于 H100-3
- Parallelization: 3 路 GPU shard 并行，分别跑 `subtractive_curation` / `importance_tagging` / `evidence_graph`
- States: 128 queries × 4 states/query × 3 components = 1536 states
- Status: `n_expected=3`，`n_finished=3`，`errors=0`

### H100-4 results
| component | n_states | I_name_normalized | I_args_raw | gate | Status |
|---|---:|---:|---:|---|---|
| subtractive_curation | 512 | +0.050380 | +0.032684 | REAL_INFLUENCE_POSITIVE | 已完成 |
| importance_tagging | 512 | +0.022636 | +0.004821 | REAL_INFLUENCE_POSITIVE | 已完成 |
| evidence_graph | 512 | +0.032975 | +0.046648 | REAL_INFLUENCE_POSITIVE | 已完成 |

#### H100-4 conclusion
- `subtractive_curation`、`evidence_graph`、`importance_tagging` 在真实 HF scorer CONFIRM128 上都高于 null controls。
- 其中 `evidence_graph` 与 `subtractive_curation` 的 combined signature 最强，且有明确语义解释空间。
- H100-4 不再停留在 prestage；`REAL_INF_CONFIRM128` 已完成。
- `chunk_neighbors` 与 `content_dedup` 仍应按 runtime / hybrid control 处理，不直接当作完全 internalize 候选。

### H100 handoff conclusions
- **已完成**：H100-1 CAL200 + CONFIRM400，H100-2 REPL200_V2，H100-3 offline + qrel-backed influence，H100-4 CONFIRM128。
- **结论稳定的 component**：`evidence_graph`、`subtractive_curation`、`importance_tagging`。
- **兼具质量与策略变化但更偏 runtime/hybrid 的 component**：`auto_populate_first_search`、`content_dedup`、`chunk_neighbors`。
- **中性或弱信号 component**：`sentence_compress`、`verify_tool`、`token_budget_marker`、`adaptive_rerank_instruction`。
- **禁止宣称**：本轮仍不能把 local BM25 / offline / HF scorer 结果写成 official Chroma parity；官方 BrowseComp+ 仍需独立凭证与独立评测。
