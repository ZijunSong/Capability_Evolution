# SCAPE result-record

> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。

---

## 本轮总览（更新于 2026-08-13 晚）

### Setting（双线）
| 线 | 机器 / repo | model | retrieval | Candidate A/B |
|---|---|---|---|---|
| **H20 true-SCAPE（主线）** | 8×H20；`/data/ppnm/Capability_Evolution/SCAPE` | `pat-jj/harness-1`（local `/data/ppnm/models/harness-1`） | local_bm25_compat | A=`evidence_graph`（L **FAIL**）；B=**未冻结**（三候选 micro tournament 全 **MICRO_FAIL**） |
| **H20 provisional（已归档）** | 同上 | Qwen2.5-7B-Instruct | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool` |
| **H100** | 8×H100；`/mnt/songzijun/Capability_Evolution/SCAPE` + worktrees `SCAPE-wt-h100-*` | `pat-jj/harness-1`（HF continuation-logprob scorer；vLLM smoke 已通过） | local BM25 compat / HF same-state scorer（官方 Chroma 阻塞） | A=`evidence_graph`；B 未冻结；H100-2 utility 推荐 `subtractive_curation` 但 H20 learnability 未过 |

### 进度板 — H20 true-SCAPE evidence_graph（0813 主线）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| Canonical pipeline + pytest | **已完成** | 16 passed；`legacy_scope_path_used=false` |
| Same-state 数据 EG_TRAIN_8K / VALID_1K / TEST_1K | **已完成** | query-disjoint；`outputs/true_scape_evidence_graph/data/` |
| Stage L V0 uniform tool-token KL（8 卡 × 18 cells） | **已完成** | Gate L **FAIL**（`divergence_not_down`） |
| Stage L retry weighted KL（8 卡 × 13 cells） | **已完成** | Gate L **FAIL**（`scaling_regression`）→ **`CURRENTLY_NOT_LEARNABLE`** |
| Stage L baselines（action_ce / full_response / offpolicy / name_only / args_only） | **已完成** | name_only L8K L_m≈11.7 离线最优，但未挽救主路径 Gate L |
| Stage S closed-loop 四格 | **未运行** | Gate L 未过；S0/S1 仅有 LOO proxy，**不可**宣称 retirement |
| Stage M / Candidate B micro tournament | **已完成 · 全 FAIL** | 见 `## 2026-08-13 SCAPE H20 Candidate-B micro-learnability tournament final` |
| Runtime recomposition | **未开始** | 等 H100-1 decomposition |
| GPU 实验进程 | **空闲** | `TOURNAMENT_ALL_DONE`；evidence_graph retry + Candidate-B micro 均完毕 |

### 进度板 — H20 true-SCAPE Candidate-B micro tournament（0813 H20 §1–§5）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| 三候选 same-state 数据（SC/IT/VT） | **已完成** | TRAIN_8K + VALID_512 + TEST_512；query-disjoint；`outputs/true_scape_candidate_b_tournament/data/` |
| Micro Stage L L512/L2K（8 卡 × 14 cells） | **已完成** | uniform `tool_token_kl`；seeds 42/43；SC 基线 action_ce + name_only @2K |
| Micro Gate（三候选） | **全 MICRO_FAIL** | reason=`divergence_not_down`（IT/VT 另有 `scaling_regression`） |
| Candidate B 冻结 | **未冻结** | `winner=null`；`CANDIDATE_B_FINAL.json` |
| Winner 8K 扩展 | **跳过** | 无 MICRO_PASS；符合 §6 禁止三候选齐跑 8K |
| Stage S 四格 | **跳过** | 无 winner + Gate L 未过 |
| Attribution-guided retry | **未触发** | 无 MICRO_WEAK（仅 FAIL） |
| PROBE_VALIDATION_V2 | **已完成** | C+I 仍不能预测 L；四组件 learnability 均负 |

### 进度板 — H20 provisional（已归档，不再扩展）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| LOCAL_CAL64 LOO 9/9 + 候选选择 | **已完成** | A/B 选出；Gate S **FAIL** → 归档 |
| A/B Stage L/S（Qwen+BM25） | **已完成 · FAIL** | 不可 retirement；见 `## 2026-08-12 SCAPE non-H100 round final` |
| 真 SCAPE same-state tool-token OPD | **已被 true-SCAPE 线取代** | 旧线不再救 |

### 进度板 — H100（0812 fresh confirm / 0813 attribution + sync）
| 阶段 | 状态 | 结论 / 产物 |
|---|---|---|
| Git/code canonicalization + GitHub sync | **已完成** | snapshot commit `0f0934bd9f7a985af747e18dda9c2c666a9c24ba`；sync branch `sync/h100-20260812` pushed to GitHub at `66047fc5d4f7ee20c3111d90c0fea13f0c44c88e`，后续 0813 sync head `31a05e9e63339d62f5ac78e743ed28ef6effe093` |
| H100-1 CAL200 historical contribution | **已完成（local BM25 compat）** | 10 组件 n=200 errors=0；保留为历史基线 |
| H100-1 fresh contribution confirm | **已完成（local BM25 compat / LOCAL_COMPAT_ONLY）** | `outputs/h100_1_contribution_confirm/`；BCP_CONFIRM400 seed1102 n=400；10/10 errors=0；SHA OK |
| H100-1 graph placement decomposition | **已完成（LOCAL_COMPAT_ONLY）** | `outputs/h100_1_graph_decomp/`；G0/G1/G2/G3/G4 对比显示 `G3` 接近 `G4`，`G2` 保留少数测得 utility，结论为 `Semantic-migratable` |
| H100-2 independent replication | **已完成（local BM25 compat / LOCAL_COMPAT_ONLY）** | `outputs/h100_2_independent_repl/`；BCP_REPL200_V2 seed2203 n=200；full + 10 LOO + 4 coalition；16/16 errors=0；SHA OK |
| H100-2 candidate-B utility resolution | **已完成（LOCAL_COMPAT_ONLY）** | `outputs/h100_2_candidate_b_utility/`；UTILITY_STATE256 对 3 component × K={2,4}；Candidate B 推荐为 `subtractive_curation`，但总体决策为 `Behavior-only` |
| H100-3 real-model same-state influence | **已完成（HF continuation-logprob）** | `outputs/h100_3_real_influence/`；7 components × 64q × 16 states = 7168 states；7/7 errors=0；SHA OK |
| H100-3 influence attribution | **已完成（CPU 聚合；GPU rescore skipped）** | `outputs/h100_3_influence_attribution/`；evidence_graph/importance_tagging/verify_tool 各 1024 states；已生成 tool/turn/argument 分层和 H20 loss recommendation |
| H100-4 real influence confirmation | **已完成（HF continuation-logprob）** | `outputs/h100_4_influence_confirm/`；REAL_INF_CONFIRM128 n=128；3 components × 512 states；3/3 positive；SHA OK |
| H100-4 verify_tool follow-up confirm | **已完成（HF continuation-logprob / `/opt` env）** | `outputs/h100_4_verify_confirm/`；VERIFY_INF_CONFIRM128 seed4414 n=128；natural 2048 states + targeted 512 states；errors=0；decision=`CONFIRMED`；`H1004_VERIFY_HANDOFF.json` 已更新 |
| Harness-1 restore + vLLM smoke | **已完成（smoke）** | 9 shards；`/v1/models` 200 |
| 官方 BrowseComp+ Chroma eval / parity | **阻塞** | 缺 `OPENAI_API_KEY` / `CHROMA_API_KEY` / `CHROMA_DATABASE`；不可用 local/HF evidence 冒充 official Chroma |

### 结论（一句话）
- **H20 true-SCAPE A-side**：`evidence_graph` 在 harness-1 + same-state tool-token KL 下 **Gate L 双次 FAIL**（uniform → weighted retry）→ **`CURRENTLY_NOT_LEARNABLE`**；停止 Evidence Graph full migration。
- **H20 true-SCAPE B-side**：`subtractive_curation` / `importance_tagging` / `verify_tool` 统一 micro tournament **三候选全 MICRO_FAIL** → **Candidate B 未冻结**；Contribution–Influence–Utility **仍未能预测** same-state tool-token OPD learnability。
- **H20 provisional**：LOCAL_CAL64 + BM25+Qwen 下 A/B **不可 retirement**（Gate S FAIL）；已归档。
- **H100**：fresh contribution + replication + real influence + H100-4 confirm 已齐；utility 排序 `subtractive_curation` > `importance_tagging` > `verify_tool`，但 **不能覆盖** H20 learnability gate。官方 Chroma 仍阻塞，所有 local/HF 结果标注 `LOCAL_COMPAT_ONLY`。

详细数字：evidence_graph Stage L 见 `## 2026-08-13 SCAPE H20 true-SCAPE evidence_graph Stage L final`；Candidate-B tournament 见 `## 2026-08-13 SCAPE H20 Candidate-B micro-learnability tournament final`；0813 H100 状态见 `## 2026-08-13 SCAPE 0813 execution status`；H20 provisional 见 `## 2026-08-12 SCAPE non-H100 round final`；H100 见 `## 2026-08-12 SCAPE H100 fresh confirm + real influence final`。

---

## 2026-08-13 SCAPE H20 true-SCAPE evidence_graph Stage L final

> 覆盖 `SCAPE-0813-五机协调.md` + `SCAPE-0813-H20.md` 中 H20 主线实验。
> 状态：**Stage L 已完成（V0 + 一次 weighted retry）**；**Gate L FAIL** → **`CURRENTLY_NOT_LEARNABLE`**；Stage S closed-loop **未运行**。

### Setting
- repo: `/data/ppnm/Capability_Evolution/SCAPE`
- machine: 8×H20-3e（单卡 LoRA 训练，每卡独立队列）
- model: released Harness-1 `pat-jj/harness-1`（local `/data/ppnm/models/harness-1`）
- component: `evidence_graph`（Candidate A）
- harness: Harness-1；student=`H_-evidence_graph` rollout；teacher=full view score-only
- retrieval: `local_bm25_compat`（官方 Chroma 阻塞）
- trainer: LoRA tool-token OPD（`scape.training.hf_tool_opd`）；`legacy_scope_path_used=false`
- data splits（query-disjoint）:
  - `EG_TRAIN_8K`（train pool，截取 512 / 2K / 8K）
  - `EG_VALID_1K`（训练期 held-out divergence）
  - `EG_TEST_1K`（post-train eval）
- loss V0: uniform tool-token KL + light anchor KL；mask = tool name + arg keys + arg values + end_search
- loss retry（Part H 唯一允许的一次）: `weighted_tool_token_kl`；span weights name=3.0, arg_key=0.5, arg_value=0.5, end_search=1.0（来自 Stage L baseline ablation：name_only >> args_only >> uniform）
- training: epochs=1, batch_size=1, lr=1e-5, LoRA r=8/alpha=16；每个 cell 从 base checkpoint 独立初始化
- output root: `outputs/true_scape_evidence_graph/`
- schedule（8 卡 Part F）:
  - GPU0: main seed42 L512→L2K→L8K（V0 uniform）
  - GPU1: main seed43 L512→L2K→L8K
  - GPU2: main seed44 L2K→L8K
  - GPU3–7: baselines（action_ce, full_response_kl, offpolicy_matched, name_only, args_only）@ 2K/8K
  - retry: GPU0–2 同主 seeds weighted；GPU3–7 额外 weighted L8K seeds 45–49

### Results — status summary
| phase | cells | loss | Gate L | note |
|---|---:|---|---|---|
| V0 uniform | 18 | `tool_token_kl` | **FAIL** (`divergence_not_down`) | 两 seed 8K held-out divergence 未稳定下降 |
| Retry weighted | 13 | `weighted_tool_token_kl` | **FAIL** (`scaling_regression`) | seed 方向一致，但 2K/8K 相对 512 系统性退化 |
| **Overall** | **31** | — | **`CURRENTLY_NOT_LEARNABLE`** | 停止 Evidence Graph full migration |

### Results — Gate L V0 uniform（seeds 42/43，摘录 8K）
| seed | d_pre | d_post@8K | L_m@8K | invalid_tool_rate |
|---:|---:|---:|---:|---|
| 42 | -0.0107 | -0.0485 | -3.54 | 0.0 |
| 43 | -0.0107 | -0.0101 | +0.050 | 0.0 |

Gate L reason: `divergence_not_down`（seed43@8K 未改善；seed42 scaling 退化）。

### Results — Gate L retry weighted（seeds 42/43，摘录）
| seed | d_pre | d_post@512 | d_post@2K | d_post@8K | L_m@8K |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.0228 | -0.0217 | -0.0292 | -0.0803 | -2.52 |
| 43 | -0.0228 | -0.0791 | -0.0402 | -0.0884 | -2.88 |

Gate L reason: `scaling_regression`（`seed_agree=true`，`invalid_ok=true`，但 2K/8K 不比 512 系统性更好）。

### Results — baselines（L8K seed42，离线 L_m 摘录）
| loss_path | L_m@8K | note |
|---|---:|---|
| `tool_name_only_kl` | +11.72 | 离线 divergence 下降最强，但未进入主 Gate L 判定 |
| `args_only_kl` | +2.21 | 次优 |
| `action_ce` | -1.55 | — |
| `full_response_kl` | -0.42 | — |
| `offpolicy_matched` | -0.32 | — |
| uniform main s42 | -3.54 | Gate L FAIL |

### Results — tool mask / data audit
| check | result |
|---|---|
| TOOL_MASK_AUDIT parsable_rate | 200/200 = 1.0 |
| DATA_AUDIT query_disjoint | pass（train/valid/test prefix 分离） |
| pytest | 16 passed |
| smoke16 / invalid tool rate | 0.0（全 cells） |

### Results — Stage S（未运行）
- Gate L 未 PASS → **不启动** closed-loop 四格（S2/S3 真实 rollout）。
- `FOUR_GRID_STAGE_S.md` 中 S2/S3 为 **LOO proxy**（`source: loo_proxy`），**不可**用于 retirement 宣称。
- 此前 Stage S vLLM 尝试因 GPU 显存被 Stage L 占用失败；按协议已在 Gate L FAIL 后跳过。

### Paired / probe check
| axis | pre-stage (H100) | post-stage (H20 true-SCAPE) |
|---|---|---|
| Contribution | ✅ positive（fresh + replicated） | — |
| Influence | ✅ positive（rank #1, H100-4 confirm） | — |
| Learnability | — | ❌ Gate L FAIL（两次） |
| Retirement | — | ❌ Stage S 未运行（仅 proxy） |

**PROBE 结论**：Contribution–Influence **未能预测** learnability。`evidence_graph` 仍是最完整的 probe-validation 目标，但 same-state tool-token KL 尚不能将 graph-aware policy 迁入 weights。

### Gate / Decision
- Gate L (V0 uniform): **FAIL** → 允许一次 weighted retry
- Gate L (weighted retry): **FAIL** → **`CURRENTLY_NOT_LEARNABLE`**
- Gate S: **未评测**（proxy 不可宣称）
- Stage M: **不进入**
- Candidate B: **未冻结**（evidence_graph Gate L FAIL 后转入 B-side tournament；三候选 micro 全 FAIL）
- Next: **不继续** Evidence Graph full migration；B-side micro tournament **已完成（全 FAIL）**；可选 follow-up：等 `H1001_GRAPH_HYBRID_HANDOFF.json` hybrid case，或新 loss/数据假设下的受控 retry

### Artifacts
- `outputs/true_scape_evidence_graph/STAGE_L_REPORT.md`
- `outputs/true_scape_evidence_graph/STAGE_L_CURVE.csv`（31 rows）
- `outputs/true_scape_evidence_graph/BASELINE_COMPARISON.md`
- `outputs/true_scape_evidence_graph/PROBE_PREDICTION_CHECK.md`
- `outputs/true_scape_evidence_graph/TOOL_MASK_AUDIT.md`
- `outputs/true_scape_evidence_graph/DATA_AUDIT.md`
- `outputs/true_scape_evidence_graph/CCR_EVIDENCE_GRAPH.json`
- `outputs/true_scape_evidence_graph/AGGREGATE.json`
- checkpoints: `stage_l/gpu*/main_L8K_s42/hf_merged`；`stage_l_retry/gpu0/weighted_L8K_s42/hf_merged`

---

## 2026-08-13 SCAPE H20 Candidate-B micro-learnability tournament final

> 覆盖 `SCAPE/0813/SCAPE-0813-H20.md` §0–§5、§10–§11（Candidate-B True-SCAPE Micro-Learnability Tournament）。
> 状态：**Micro Stage L 已完成（14 cells）**；三候选 **全 MICRO_FAIL**；**Candidate B 未冻结**；8K / Stage S **按协议跳过**。

### Setting
- repo: `/data/ppnm/Capability_Evolution/SCAPE`
- machine: 8×H20-3e（单卡 LoRA；每卡独立队列；`CUDA_VISIBLE_DEVICES` 绑定）
- model: Harness-1 `pat-jj/harness-1`（local `/data/ppnm/models/harness-1`）
- candidates（统一 tournament，不手工预冻结 B）:
  - `subtractive_curation`（SC）
  - `importance_tagging`（IT）
  - `verify_tool`（VT；Gate L 仅用 natural states，未混 targeted-eligible）
- harness: Harness-1；student=`H_-m` rollout 决定 state occupancy；same `xi_t` dual-view；teacher score-only；no future trajectory
- retrieval: `local_bm25_compat`（官方 Chroma 阻塞）
- trainer: LoRA tool-token OPD（`scape.training.hf_tool_opd`）；`legacy_scope_path_used=false`
- data（每候选 query-disjoint splits，前缀分离）:
  - `{component}_TRAIN_8K.jsonl`（训练池；micro 仅截取 512 / 2K）
  - `{component}_VALID_512.jsonl`
  - `{component}_TEST_512.jsonl`
- loss: uniform name+args tool-token KL + light anchor（`tool_token_kl`）；mask = tool name + arg keys + arg values + end_search
- training: epochs=1, batch_size=1, lr=1e-5, LoRA r=8/alpha=16；每 cell 从 **同一 base checkpoint 独立初始化**
- micro scale only: L512 + L2K per candidate × seeds {42, 43}（**不直接跑 8K**）
- output root: `outputs/true_scape_candidate_b_tournament/`
- 8-GPU schedule（§3）:
  - GPU0: SC seed42 L512→L2K
  - GPU1: SC seed43 L512→L2K
  - GPU2: IT seed42 L512→L2K
  - GPU3: IT seed43 L512→L2K
  - GPU4: VT seed42 L512→L2K
  - GPU5: VT seed43 L512→L2K
  - GPU6: SC `action_ce` @2K seed42
  - GPU7: SC `tool_name_only_kl` @2K seed42
- scripts（其他 agent 复跑/监控）:
  - `scripts/build_candidate_b_tournament_splits.py`
  - `scripts/launch_candidate_b_tournament_micro_8gpu.sh`
  - `scripts/launch_candidate_b_winner_8k_8gpu.sh`（仅 MICRO_PASS winner 时启用）
  - `scripts/aggregate_candidate_b_tournament.py`
  - `scripts/monitor_candidate_b_tournament.sh`
- ops note: 队列脚本初版 `shift 6` 错误（应为 `shift 5` per cell）导致 L2K 未自动接续；已修复并手动恢复 GPU0–5 L2K 训练。

### Results — Micro Gate summary
| component | verdict | Gate L reason | seed_agree | scaling_ok | invalid_ok |
|---|---|---|---|---|---|
| subtractive_curation | **MICRO_FAIL** | `divergence_not_down` | false | true | true |
| importance_tagging | **MICRO_FAIL** | `divergence_not_down` | false | false | true |
| verify_tool | **MICRO_FAIL** | `divergence_not_down` | false | false | true |

MICRO_PASS 要求（§4）：两 seed 方向一致、`D_post_2K < D_pre`、2K 不比 512 系统性退化、invalid tool 不升 — **无一满足**。
MICRO_WEAK / attribution-guided 2K retry：**未触发**（无候选处于一 seed 正、一 seed 近零）。

### Results — per-cell L_m@2K（uniform `tool_token_kl`）
| component | seed | d_pre | d_post@2K | L_m@2K | invalid_tool |
|---|---:|---:|---:|---:|---|
| subtractive_curation | 42 | -0.1344 | -0.0234 | 0.826 | 0.0 |
| subtractive_curation | 43 | -0.1344 | -0.0345 | 0.743 | 0.0 |
| importance_tagging | 42 | -0.0464 | -0.0326 | 0.296 | 0.0 |
| importance_tagging | 43 | -0.0464 | +0.0009 | 1.020 | 0.0 |
| verify_tool | 42 | -0.1731 | -0.0382 | 0.780 | 0.0 |
| verify_tool | 43 | -0.1731 | -0.0326 | 0.811 | 0.0 |

注：Gate L 判定 `d_post < d_pre - 1e-6`（held-out divergence 下降）；上表多数 cell 的 `d_post` 较负的 `d_pre` **更接近 0**，故 Gate 记为未改善，尽管部分 `L_m` 数值偏高（`L_m = 1 - D_post/D_pre` 在负 divergence 区间非单调）。

### Results — SC micro baselines @2K seed42
| loss_path | d_post@2K | L_m@2K |
|---|---:|---:|
| `action_ce` | -0.0284 | 0.789 |
| `tool_name_only_kl` | -0.0438 | 0.367 |

### Paired / PROBE_VALIDATION_V2
| component | Contribution | Influence | Utility | Learnability (H20 micro) | Placement |
|---|---|---|---|---|---|
| evidence_graph | + | + | semantic-migratable | **FAIL**（uniform + weighted 8K retry） | semantic-migratable / hybrid |
| subtractive_curation | +（最稳） | + | strong（H100-2） | **MICRO_FAIL** | semantic-migratable |
| importance_tagging | mixed | + | mid | **MICRO_FAIL** | semantic-migratable |
| verify_tool | neutral（local） | strong | weak vs subtractive | **MICRO_FAIL** | semantic-migratable |

**PROBE 结论**：Contribution–Influence–Utility **仍不能预测** harness-1 same-state tool-token OPD learnability。Pre-stage 升级为 Contribution–Influence–Utility–Learnability **仍缺乏 learnability 正向证据**。

### Gate / Decision
- Micro Gate（三候选）: **全 MICRO_FAIL**
- Candidate B freeze: **否**（`winner_component_id=null`）
- Winner 8K expansion（§6）: **跳过**（无 MICRO_PASS）
- Stage S 四格（§8）: **跳过**（无 winner + Gate L 未过）
- Stage M: **不进入**
- Evidence Graph full migration: **已停止**（A-side 先前结论不变）
- Next allowed: 等待 `H1001_GRAPH_HYBRID_HANDOFF.json` 作 hybrid case；或对新 loss/数据管线有明确假设后再开 retry（当前无 MICRO_WEAK，§7 attribution retry **不适用**）

### Artifacts
- `outputs/true_scape_candidate_b_tournament/DATA_AUDIT.md`
- `outputs/true_scape_candidate_b_tournament/MICRO_STAGE_L.csv`（14 rows）
- `outputs/true_scape_candidate_b_tournament/MICRO_STAGE_L_REPORT.md`
- `outputs/true_scape_candidate_b_tournament/CANDIDATE_B_FINAL.{json,md}`
- `outputs/true_scape_candidate_b_tournament/PROBE_VALIDATION_V2.md`
- `outputs/true_scape_candidate_b_tournament/BASELINE_COMPARISON.md`
- `outputs/true_scape_candidate_b_tournament/WINNER_8K_REPORT.md`（无 8K cells）
- `outputs/true_scape_candidate_b_tournament/FOUR_GRID_STAGE_S.md`（no_winner）
- `outputs/true_scape_candidate_b_tournament/RUN_MANIFEST.json`
- `outputs/true_scape_candidate_b_tournament/TOURNAMENT_ALL_DONE`
- per-cell checkpoints: `stage_l_micro/gpu*/{SC,IT,VT}_L{512,2K}_s*/hf_merged`

---

## 2026-08-13 SCAPE 0813 execution status

> 根据 `SCAPE/0813/SCAPE-0813-五机协调.md` 与 `SCAPE/0813/SCAPE-0813-H100-{1,2,3,4}.md`、`SCAPE/0813/SCAPE-0813-H20.md` 更新。本节记录 0813 调度下所有已执行/已 gate-block 的 setting、结果、结论及跨服务器 handoff 信息。官方 Chroma 仍因 credential 缺失阻塞；不会把 local/HF evidence 冒充 official Chroma parity。

### Setting
- repo: `/mnt/songzijun/Capability_Evolution/SCAPE` on branch `sync/h100-20260812`
- GitHub remote: `https://github.com/ZijunSong/Capability_Evolution.git`
- required H100 snapshot ancestor: `0f0934bd9f7a985af747e18dda9c2c666a9c24ba`
- GPU/env rule learned on 2026-08-13: **do not run torch/vLLM from `/mnt` JuiceFS environments**. GPU-heavy Python envs must live under `/opt`; current HF scorer env is `/opt/scape-hf-scorer/bin/python`.
- visible GPUs for final verify run: 4 GPUs exposed by current node; `device_map=auto` was added to the HF scorer and used to shard the Harness-1 checkpoint across visible GPUs. No leftover scorer/vLLM processes after completion.
- model for HF influence/confirm: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1` released checkpoint)
- official Chroma credentials: unavailable (`OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` missing) -> `OFFICIAL_CHROMA_BLOCKED=true`; continue local/HF mechanism experiments only
- H100-3 attribution input: `outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl`
- H100-4 verify follow-up setting: `VERIFY_INF_CONFIRM128`, component=`verify_tool`, seed=4414, n_queries=128, max_states_per_query=16, scorer=`hf_continuation_logprob`, output=`outputs/h100_4_verify_confirm/`

### Completed 0813 workstreams
| workstream | setting / scale | artifacts | result / conclusion |
|---|---|---|---|
| H100-1 Evidence Graph Placement Decomposition | `BCP_GRAPH_DECOMP200`, variants G0–G4; `LOCAL_COMPAT_ONLY=true` | `outputs/h100_1_graph_decomp/` | `G3` close to `G4`, `G2` retains minority utility → `Semantic-migratable` |
| H100-2 Candidate-B Utility Resolution | `importance_tagging`, `verify_tool`, `subtractive_curation`; `UTILITY_STATE256` | `outputs/h100_2_candidate_b_utility/` | Decision=`Behavior-only`; utility: `subtractive_curation` > `importance_tagging` > `verify_tool` |
| H100-3 Influence Attribution | 3 components × 1024 states = 3072 rows | `outputs/h100_3_influence_attribution/` | evidence_graph/importance_tagging/verify_tool attribution complete; H20 loss recommendation generated |
| H100-4 prior real influence confirm | REAL_INF_CONFIRM128, 3 components | `outputs/h100_4_influence_confirm/` | 3/3 positive, errors=0 |
| H100-4 `verify_tool` independent confirm | `VERIFY_INF_CONFIRM128`, seed=4414; `/opt/scape-hf-scorer/bin/python` | `outputs/h100_4_verify_confirm/`; `H1004_VERIFY_HANDOFF.json` | natural 2048 + targeted 512 states; errors=0; Decision=`CONFIRMED` |
| H100-4 `auto_populate` argument diagnostic | 128 states from HF per-state rows | `outputs/h100_4_verify_confirm/auto_populate_argument_diagnostic/` | I_args_raw_mean=-0.280; 98/128 negative args signal |
| H20 true-SCAPE Stage L evidence_graph | 8×H20, 31 cells (V0 18 + weighted retry 13) | `outputs/true_scape_evidence_graph/` | Gate L 双次 FAIL → **`CURRENTLY_NOT_LEARNABLE`**；见 `## 2026-08-13 SCAPE H20 true-SCAPE evidence_graph Stage L final` |
| H20 Candidate-B micro tournament | 8×H20, 14 micro cells (SC/IT/VT L512/L2K + SC baselines) | `outputs/true_scape_candidate_b_tournament/` | 三候选 **全 MICRO_FAIL**；Candidate B **未冻结**；见 `## 2026-08-13 SCAPE H20 Candidate-B micro-learnability tournament final` |
| 0813 status consolidation | reads completed artifacts only | `outputs/scape_prestage_v2/0813_STATUS_SUMMARY.{json,md}` | `missing={}` in required-artifact presence check |

### Key metrics / decisions
| item | value |
|---|---:|
| H100-4 verify natural states | 2048 |
| H100-4 verify targeted states | 512 |
| H100-4 verify natural I_name_normalized | 0.018325 |
| H100-4 verify natural I_args_raw | 0.039954 |
| H100-4 verify targeted I_name_normalized | 0.018523 |
| H100-4 verify gate | CONFIRMED |
| auto_populate diagnostic I_args_raw_mean | -0.280096 |
| auto_populate negative args states | 98/128 |
| required 0813 artifact presence check | missing = `{}` |
| final targeted tests | 3 passed |
| final GPU/process status | GPUs idle; no verify/vLLM/torchrun process remains |

### Candidate / placement conclusion
- Candidate A remains `evidence_graph`（但 H20 Gate L **FAIL** → 不做 full internalization migration）。
- `evidence_graph` placement decomposition supports a hybrid SCAPE target: external graph state should remain available, while graph-aware semantic decisions are migratable into weights and renderer/controller can be slimmed later.
- Candidate B is **not frozen** after H20 micro tournament:
  - `subtractive_curation`: H100-2 utility 最强，但 micro Gate **MICRO_FAIL**（`divergence_not_down`）。
  - `importance_tagging`: H100 influence 正、utility mid，但 micro **MICRO_FAIL**（`divergence_not_down` + `scaling_regression`）。
  - `verify_tool`: H100-4 **CONFIRMED**、influence strong，但 utility 最弱且 micro **MICRO_FAIL**（`divergence_not_down` + `scaling_regression`）。
  - **不得**仅凭 H100 utility/influence 覆盖 H20 learnability gate 冻结 B。
- Pre-stage 探针：Contribution–Influence–Utility **未能预测** harness-1 same-state tool-token OPD learnability（见 `PROBE_VALIDATION_V2.md`）。
- Runtime controls remain `chunk_neighbors` and `content_dedup`; do not promote them to first-round full internalization targets.
- H20 micro tournament 使用 uniform `tool_token_kl`；H100-3 attribution 仅作后续 ablation 参考，**未**触发 §7 attribution-guided retry（无 MICRO_WEAK）。

### Blocked / intentionally not continued
| workstream | status | reason |
|---|---|---|
| Official BrowseComp+ Chroma parity | **blocked** | missing `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`; checked once and not polled repeatedly. |
| H20 Evidence Graph Stage S/M | **not started** | Gate L FAIL (`CURRENTLY_NOT_LEARNABLE`); per auto-stop rule, no Stage S/M or retirement claim. |
| H20 Candidate-B 8K / Stage S | **skipped** | 三候选 micro 全 MICRO_FAIL；`winner=null`；per `SCAPE-0813-H20.md` §6–§8. |
| Old SCOPE rollback / KEEP-SKIP / P_m / old Stage M | **not continued** | explicitly forbidden by 0813 coordination. |

### Repo / worktree hygiene and handoff notes
- GPU workloads must use `/opt` Python envs, not `/mnt` JuiceFS conda/venv (torch/vLLM hang risk on JuiceFS).
- `scripts/run_h100_3_real_influence_hf.py` supports `--device auto` and `device_map=auto` for multi-GPU checkpoint loading.
- `scripts/run_h100_4_verify_confirm_hf.py` is the independent `verify_tool` confirm runner; `scripts/finalize_h100_4_verify_confirm.py` finalizes reports.
- `scripts/finalize_h100_4_auto_populate_diagnostic.py` produces auto_populate argument diagnostic from existing per-state rows.
- H20 Candidate-B tournament runners: `scripts/build_candidate_b_tournament_splits.py`, `launch_candidate_b_tournament_micro_8gpu.sh`, `launch_candidate_b_winner_8k_8gpu.sh`, `aggregate_candidate_b_tournament.py`, `monitor_candidate_b_tournament.sh`；产物根目录 `outputs/true_scape_candidate_b_tournament/`。
- Worktree checkout directories (`SCAPE-wt-h100-*`) must remain uncommitted; `CLAUDE.md` is gitignored and must not be committed.
- Outputs/checkpoints/models/indexes/secrets remain uncommitted; this record points other agents to artifact paths under `outputs/`.

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

## 2026-08-12 SCAPE H100 fresh confirm + real influence final

> 本节记录 0812 新调度（`SCAPE-0812-五机协调.md` + `SCAPE-0812-H100-1.md` 及 H100-2/3/4 补证据缺口）完成后的 H100 证据状态。官方 Chroma 凭证缺失，因此 retrieval quality 仍标注 `LOCAL_COMPAT_ONLY`；HF same-state influence 使用 released Harness-1 checkpoint 的 continuation logprob scorer，不等同于 Chroma parity。

### Setting
- code snapshot / barrier: `0f0934bd9f7a985af747e18dda9c2c666a9c24ba`；`/mnt/songzijun/Capability_Evolution/SCAPE/GIT_SYNC_H100_READY`
- worktrees:
  - H100-1: `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1` (`exp/h1001-contribution-confirm`)
  - H100-2: `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2` (`exp/h1002-independent-repl`)
  - H100-3: `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3` (`exp/h1003-real-influence`)
  - H100-4: `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4` (`exp/h1004-influence-confirm`)
- model: released Harness-1 checkpoint `pat-jj/harness-1` / local path `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`
- retrieval: `local_bm25_compat` for contribution/replication; official Chroma blocked by missing `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`
- scorer for real influence: HF continuation-logprob over legal Harness-1 tool names (`fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`)
- decode / contribution controls: temperature=0 / deterministic compatibility; same split manifest per cell; no training / no weight updates
- final status file: `outputs/scape_prestage_v2/0812_EXECUTION_STATUS.md`

### Results — status summary
| stream | split / scale | output | status |
|---|---|---|---|
| H100-1 contribution confirm | BCP_CONFIRM400 seed1102 n=400 | `outputs/h100_1_contribution_confirm/` | 10/10 complete, errors=0, SHA OK |
| H100-2 independent replication | BCP_REPL200_V2 seed2203 n=200 | `outputs/h100_2_independent_repl/` | full + 10 LOO + 4 coalition, 16/16 complete, errors=0, SHA OK |
| H100-3 real influence | REAL_INF64, 7 components × 64q × 16 states | `outputs/h100_3_real_influence/` | 7168 states, 7/7 complete, errors=0, SHA OK |
| H100-4 confirm | REAL_INF_CONFIRM128, 3 components × 128q × 4 states | `outputs/h100_4_influence_confirm/` | 1536 states, 3/3 complete, errors=0, SHA OK |

### Results — H100-1 fresh contribution confirm (n=400)
| component | Δ curated | Δ trajectory | Δ final | Δ reward | note |
|---|---:|---:|---:|---:|---|
| subtractive_curation | +0.004870 | +0.000000 | +0.000000 | +0.002192 | stable positive curated/reward |
| importance_tagging | +0.003277 | +0.000938 | -0.000521 | +0.001844 | positive curated/reward; final tiny negative |
| auto_populate_first_search | +0.000000 | +0.008605 | +0.000000 | +0.003872 | strong trajectory effect |
| evidence_graph | +0.000000 | +0.004086 | +0.000000 | +0.001839 | positive trajectory/reward |
| chunk_neighbors | +0.000000 | +0.004086 | +0.000000 | +0.001839 | runtime/hybrid control |
| content_dedup | +0.002610 | +0.006659 | +0.000000 | +0.004171 | deterministic/runtime control despite positive local effect |
| adaptive_rerank_instruction | +0.002545 | +0.002568 | +0.000563 | +0.002357 | positive local effect |

`sentence_compress` / `verify_tool` / `token_budget_marker` are absent from the compact confirm table because their local contribution rows are neutral in the generated markdown; full rows remain in `CONTRIBUTION_CONFIRM.{csv,json}`.

### Results — H100-2 independent 10-component replication (n=200)
| component | Δ curated | Δ trajectory | Δ final | Δ reward | quality_positive |
|---|---:|---:|---:|---:|---|
| subtractive_curation | +0.004458 | +0.000000 | +0.000000 | +0.002006 | true |
| importance_tagging | +0.001333 | -0.001722 | +0.000000 | -0.000175 | true / split-sensitive |
| auto_populate_first_search | +0.000000 | +0.006387 | +0.000000 | +0.002874 | true |
| evidence_graph | +0.000000 | +0.004887 | +0.000000 | +0.002199 | true |
| sentence_compress | +0.000000 | +0.000000 | +0.000000 | +0.000000 | false |
| chunk_neighbors | +0.000000 | +0.004887 | +0.000000 | +0.002199 | true / runtime-hybrid |
| content_dedup | -0.001333 | +0.003000 | +0.000000 | +0.000750 | true / runtime control |
| verify_tool | +0.000000 | +0.000000 | +0.000000 | +0.000000 | false |
| token_budget_marker | +0.000000 | +0.000000 | +0.000000 | +0.000000 | false |
| adaptive_rerank_instruction | +0.001667 | +0.000048 | +0.000000 | +0.000771 | true |

Coalition V2 artifacts (`COALITION_V2.csv`) were generated and used only as scheduling/interaction diagnostics, not as a strong causal claim.

### Results — H100-3 real-model same-state influence
| rank | component | n_states | I_name_normalized | I_args_raw | gate |
|---:|---|---:|---:|---:|---|
| 1 | evidence_graph | 1024 | 0.038704 | 0.117327 | REAL_INFLUENCE_POSITIVE |
| 2 | verify_tool | 1024 | 0.019043 | 0.050669 | REAL_INFLUENCE_POSITIVE |
| 3 | importance_tagging | 1024 | 0.028771 | 0.016560 | REAL_INFLUENCE_POSITIVE |
| 4 | subtractive_curation | 1024 | 0.013124 | 0.018983 | REAL_INFLUENCE_POSITIVE |
| 5 | content_dedup | 1024 | 0.035685 | -0.004602 | REAL_INFLUENCE_POSITIVE |
| 6 | chunk_neighbors | 1024 | 0.000000 | 0.000000 | NO_ABOVE_NULL_SIGNAL |
| 7 | auto_populate_first_search | 1024 | 0.048203 | -0.241941 | REAL_INFLUENCE_POSITIVE |

Top-3 by `I_name_normalized + I_args_raw`: `evidence_graph`, `verify_tool`, `importance_tagging`.

### Results — H100-4 CONFIRM128 real influence
| component | n_states | I_name_normalized | I_args_raw | gate |
|---|---:|---:|---:|---|
| evidence_graph | 512 | 0.032975 | 0.046648 | REAL_INFLUENCE_POSITIVE |
| importance_tagging | 512 | 0.022636 | 0.004821 | REAL_INFLUENCE_POSITIVE |
| subtractive_curation | 512 | 0.008787 | -0.000947 | REAL_INFLUENCE_POSITIVE |

H100-4 completed its originally selected confirm set. `verify_tool` became H100-3 real-influence rank 2 after the full seven-component HF sweep, but it was not part of this completed H100-4 CONFIRM128 set; mark it as high-priority follow-up confirm rather than H100-4-confirmed.

### Paired / controls
- H100-1/H100-2 contribution and replication are query-disjoint fresh splits with paired full vs `minus_m` metrics and bootstrap CIs in CSV/JSON artifacts.
- H100-3/H100-4 influence uses same-environment-state snapshots from reduced `H_-m` state occupancy; full view is score-only and does not step future trajectories.
- Null controls included: same-render identity controls and field-order perturbation fields; reported nulls are zero in the completed HF shard summaries.
- Official Chroma was checked once and blocked; no repeated polling.

### Gate / Decision
- Gate C/R: contribution direction is most stable for `evidence_graph` across H100-1 confirm and H100-2 replication; `importance_tagging` has positive H100-1 confirm but split-sensitive H100-2 reward/trajectory; `verify_tool` is neutral in local contribution but high in real influence.
- Gate I: `evidence_graph`, `importance_tagging`, `subtractive_curation` are H100-4 confirmed positive; H100-3 also elevates `verify_tool`.
- Placement sanity: do not pick `chunk_neighbors` or `content_dedup` as full internalization candidates by default; keep them as runtime controls.
- H20 handoff recommendation from completed confirmed evidence: Candidate A=`evidence_graph`, Candidate B=`importance_tagging`; runtime controls=`chunk_neighbors`, `content_dedup`.
- Follow-up if more H100 time is allocated: run H100-4 CONFIRM128 for `verify_tool`, because H100-3 real influence ranks it #2 but H100-4 did not confirm it in this pass.
- No training or retirement claim is made from these H100 experiments. Official Harness-1 Chroma parity remains unresolved.

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

### H100-1 graph placement decomposition（2026-08-13）

**Setting**
- 输入：`outputs/h100_1_contribution_confirm/` + SCAPE renderer audit
- 输出：`outputs/h100_1_graph_decomp/`
- 结论标签：`LOCAL_COMPAT_ONLY=true`，`official_chroma_parity=false`
- 目的：判断 `evidence_graph` 的收益来自 external state、renderer，还是 graph-aware decision surface

**Results**
| variant | quality delta | trajectory delta | state ops delta | latency delta ms | render tokens delta |
|---|---:|---:|---:|---:|---:|
| G0_FULL | 0.002376 | 0.005280 | 3.000 | 6.846 | 180 |
| G1_GRAPH_OFF | 0.000000 | 0.000000 | 0.000 | 0.000 | 0 |
| G2_GRAPH_STATE_ONLY | 0.000832 | 0.001848 | 3.000 | 4.450 | 0 |
| G3_GRAPH_STATE_PLUS_MINIMAL_RENDER | 0.001948 | 0.004329 | 3.000 | 5.613 | 72 |
| G4_GRAPH_FULL_RENDER | 0.002376 | 0.005280 | 3.000 | 6.846 | 180 |

**Conclusion**
- `G3` 接近 `G4`，`G2` 只保留少数测得 utility。
- 当前证据支持 `Semantic-migratable`：保留 external graph state，将训练重点放在 graph-aware tool decision，后续再收缩 renderer/controller，而不是直接删掉 graph state。
- `GRAPH_RUNTIME_COST.md` 显示 graph 有非零 state/latency 成本，因此 retirement 若发生，也应优先收缩 renderer/controller。
- `GRAPH_RENDERER_ROBUSTNESS.md` 未检测到明显 field-order 语义依赖。

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

### H100-2 candidate-B utility resolution（2026-08-13）

**Setting**
- 指令：`SCAPE/0813/SCAPE-0813-H100-2.md`
- 输出：`outputs/h100_2_candidate_b_utility/`
- 目标组件：`importance_tagging` / `verify_tool` / `subtractive_curation`
- Probe：same-state short-horizon utility，`UTILITY_STATE256`，每组件 K=2/K=4，natural states=256，targeted states=0
- 产物：`SHORT_HORIZON_UTILITY_PER_STATE.jsonl`、`SHORT_HORIZON_UTILITY_SUMMARY.csv`、`*_UTILITY.md`、`CANDIDATE_B_RECOMMENDATION.{md,json}`、`SHA256SUMS`；无 `RUN_MANIFEST.json`
- 标记：`LOCAL_COMPAT_ONLY=true`，`official_chroma_parity=false`

**Results**
| component | K | n_states | mean T-S | mean T | mean S |
|---|---:|---:|---:|---:|---:|
| importance_tagging | 2 | 256 | 0.001600 | 0.002095 | 0.000495 |
| importance_tagging | 4 | 256 | 0.001917 | 0.002412 | 0.000495 |
| verify_tool | 2 | 256 | 0.001046 | 0.001046 | 0.000000 |
| verify_tool | 4 | 256 | 0.001534 | 0.001534 | 0.000000 |
| subtractive_curation | 2 | 256 | 0.002014 | 0.002839 | 0.000825 |
| subtractive_curation | 4 | 256 | 0.002239 | 0.003064 | 0.000825 |

**Candidate-B ranking**
| rank | component | mean short-horizon T-S | real influence + args | local reward delta |
|---:|---|---:|---:|---:|
| 1 | subtractive_curation | 0.002126 | 0.032106 | 0.001929 |
| 2 | importance_tagging | 0.001758 | 0.045332 | 0.001069 |
| 3 | verify_tool | 0.001290 | 0.069712 | 0.000000 |

**Decision**
- `CANDIDATE_B_RECOMMENDATION.json` 的总体 decision 为 `Behavior-only`，推荐 Candidate B 为 `subtractive_curation`。
- `verify_tool` real influence 最高但 local reward delta 为 0，且短 horizon utility 最弱；不应因 influence rank #2 直接升级为主 B。
- `importance_tagging` 保持语义候选，但 utility 排名低于 `subtractive_curation`。
- `NULL_REPLAY_NOISE.md` 中 replay noise 暂按 deterministic local artifacts 记为 0；若未来有 live Harness-1 replay runner，需要替换为实测 noise。

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

## 2026-08-12 SCAPE H100-3 real-model same-state influence final

### Setting
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`
- Model: released Harness-1 local HF checkpoint at `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`
- Environment: `/opt/scape-h1003-hf-scorer`（新建于 `/opt`；未使用 `/mnt` 下 conda/venv 环境）
- Scorer: `hf_continuation_logprob`，对合法 Harness-1 tool names 计算 conditional sequence logprob 后归一化为 `P(tool_name | view)`。
- Tool names: `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`
- State occupancy: `xi_t` 来自每个 component 的 `H_-m` reduced same-state snapshot；full view 只 render/score 当前 snapshot，不继续执行 future trajectory。
- Scale: `REAL_INF64`；7 components × 64 queries/component × 16 states/query = 7168 per-state records。
- Parallelism: 7 GPU shards（GPU0–6 各 1 个 component）；GPU7 保留给 null/parity/监控；所有 7 个 shard completed、errors=0。
- Retrieval/data source for snapshot construction: local BrowseComp+ qrel/corpus compatibility path；**非官方 Chroma Cloud**。
- Output: `outputs/h100_3_real_influence/`

### Artifacts
- `REAL_INFLUENCE_PER_STATE.jsonl`
- `REAL_INFLUENCE_BY_COMPONENT.csv`
- `REAL_INFLUENCE_BY_COMPONENT.json`
- `REAL_INFLUENCE_BY_COMPONENT.md`
- `NULL_CONTROL_REPORT.md`
- `SCORER_PARITY.md`
- `SNAPSHOT_REPLAY_AUDIT.md`
- `TOP_CANDIDATES_FOR_CONFIRM.json`
- `RUN_MANIFEST.json`
- `STATUS_LIVE.md`
- `SHA256SUMS`

### Results
| rank | component | n_states | I_name_raw | I_name_null | I_name_normalized | I_args_raw | Gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `evidence_graph` | 1024 | 0.038704 | 0.000000 | 0.038704 | 0.117327 | `REAL_INFLUENCE_POSITIVE` |
| 2 | `verify_tool` | 1024 | 0.019043 | 0.000000 | 0.019043 | 0.050669 | `REAL_INFLUENCE_POSITIVE` |
| 3 | `importance_tagging` | 1024 | 0.028771 | 0.000000 | 0.028771 | 0.016560 | `REAL_INFLUENCE_POSITIVE` |
| 4 | `subtractive_curation` | 1024 | 0.013124 | 0.000000 | 0.013124 | 0.018983 | `REAL_INFLUENCE_POSITIVE` |
| 5 | `content_dedup` | 1024 | 0.035685 | 0.000000 | 0.035685 | -0.004602 | `REAL_INFLUENCE_POSITIVE` |
| 6 | `chunk_neighbors` | 1024 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `NO_ABOVE_NULL_SIGNAL` |
| 7 | `auto_populate_first_search` | 1024 | 0.048203 | 0.000000 | 0.048203 | -0.241941 | `REAL_INFLUENCE_POSITIVE` |

### Null / parity / audit
- Null controls in `NULL_CONTROL_REPORT.md`:
  - `N0 full-render vs full-render = 0`
  - `N1 reduced-render vs reduced-render = 0`
  - `N2 field-order-only perturbation = 0`
- `SNAPSHOT_REPLAY_AUDIT.md` records that every per-state row stores `snapshot_hash`, raw structured `xi_t`, reduced view, full view, and student executed action.
- `SCORER_PARITY.md` records that this final run used the canonical HF continuation scorer. vLLM arbitrary-continuation parity was not run because the inspected local vLLM eval path only returned generated token ids, not reliable arbitrary continuation scoring; this follows the H100-3 B2 fallback rule to use HF scorer rather than heuristic/string proxy.
- `SHA256SUMS` generated for all final artifacts.

### TOP_CANDIDATES_FOR_CONFIRM
`outputs/h100_3_real_influence/TOP_CANDIDATES_FOR_CONFIRM.json` contains at most 3 candidates:

1. `evidence_graph`
2. `verify_tool`
3. `importance_tagging`

### Conclusion
- The previous `deterministic_offline_stub` H100-3 ranking is superseded for real influence by this released-checkpoint HF logprob run.
- `evidence_graph` has the strongest combined signal because it has both positive tool-name influence and the largest positive argument influence.
- `verify_tool` is rank 2 in real influence and should be prioritized for H100-4 or follow-up confirmation despite being neutral in older H100-1 local contribution.
- `importance_tagging` remains a stable semantic candidate with positive real influence.
- `chunk_neighbors` is not above null in this real influence run and remains a runtime/hybrid control rather than a first-round internalization target.
- `content_dedup` and `auto_populate_first_search` show positive tool-name movement but have runtime/control placement concerns and/or negative argument-side score; they should not be selected by influence alone.
- These results are **not** official Chroma Cloud parity and must be labeled local/HF same-state evidence, not final Harness-1 Cloud/Chroma reproduction.

### H100-3 attribution details

**Tool-level / turn-level**
- `INFLUENCE_BY_TOOL.csv` keeps `evidence_graph`, `verify_tool`, `importance_tagging` as the top three by combined influence.
- `INFLUENCE_BY_TURN.csv` and `HIGH_INFLUENCE_ARCHETYPES.jsonl` show the strongest signal concentrated in later turns, especially around `read`, `curate`, `verify`, and `end` actions.

**Argument-class details**
| component | argument class | disagreement | n_states | I_name_mean | I_args_mean |
|---|---|---|---:|---:|---:|
| evidence_graph | doc ids | name change | 169 | 0.123880 | 0.235895 |
| evidence_graph | termination reason | no meaningful change | 769 | 0.021832 | 0.102246 |
| evidence_graph | doc ids | no meaningful change | 86 | 0.022191 | 0.019173 |
| importance_tagging | doc ids | name change | 131 | 0.104890 | 0.167589 |
| importance_tagging | doc ids | no meaningful change | 85 | 0.023446 | 0.035963 |
| importance_tagging | termination reason | no meaningful change | 808 | 0.016990 | -0.009967 |
| verify_tool | doc ids | name change | 127 | 0.069249 | 0.147595 |
| verify_tool | termination reason | no meaningful change | 764 | 0.011876 | 0.045385 |
| verify_tool | doc ids | no meaningful change | 133 | 0.012272 | -0.011529 |

**H20 handoff**
- `H20_LOSS_RECOMMENDATION.md` keeps first-run H20 V0 as **uniform name+args tool-token KL**.
- Follow-up weighting suggestions: `evidence_graph` name medium / args high; `importance_tagging` name high / args medium; `verify_tool` name medium / args high.
- This is for later ablations only; it does not override the initial uniform loss.
