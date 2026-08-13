# SCAPE result-record

> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。

---

## 本轮总览（更新于 2026-08-13）

### Setting（双线）
| 线 | 机器 / repo | model | retrieval | Candidate A/B |
|---|---|---|---|---|
| **非 H100（H20）** | 8×H20；`/data/ppnm/Capability_Evolution/SCAPE` | `/data/ppnm/models/Qwen2.5-7B-Instruct` | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool` |
| **H100** | 8×H100；`/mnt/songzijun/Capability_Evolution/SCAPE` + worktrees `SCAPE-wt-h100-*` | `pat-jj/harness-1`（HF continuation-logprob scorer；vLLM smoke 已通过） | local BM25 compat / HF same-state scorer（官方 Chroma 阻塞） | A=`evidence_graph`；B=`importance_tagging`；`verify_tool` 为高优先级 follow-up |

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
| H100-4 verify_tool follow-up confirm | **进行中 / 异常待恢复** | `outputs/h100_4_verify_confirm/`；VERIFY_INF_CONFIRM128 seed4414 n=128 setting 已冻结；首次运行因 `/opt/scape-h1003-hf-scorer/bin/python` 不存在失败，`n_finished=0/errors=1`；wrapper `scripts/run_h100_4_verify_confirm_hf.py` 已加入 |
| Harness-1 restore + vLLM smoke | **已完成（smoke）** | 9 shards；`/v1/models` 200 |
| 官方 BrowseComp+ Chroma eval / parity | **阻塞** | 缺 `OPENAI_API_KEY` / `CHROMA_API_KEY` / `CHROMA_DATABASE`；不可用 local/HF evidence 冒充 official Chroma |

### 结论（一句话）
- **非 H100**：LOCAL_CAL64 + BM25+Qwen 下 A/B **不可 retirement**（Gate S FAIL）；Stage M 已停。
- **H100**：fresh contribution confirm + independent replication + HF same-state real influence + H100-4 confirm 已齐；当前 H20 handoff 推荐 `evidence_graph` / `importance_tagging`，`verify_tool` 是 H100-3 real-influence rank2 但未被 H100-4 CONFIRM128 覆盖，列为高优先级 follow-up confirm。官方 Chroma 仍阻塞，所有 local/retrieval 贡献结果必须标注 `LOCAL_COMPAT_ONLY`。

详细数字：0813 状态见 `## 2026-08-13 SCAPE 0813 execution status`；非 H100 见 `## 2026-08-12 SCAPE non-H100 round final`；H100 历史同步见 `## 2026-08-12 SCAPE H100-1/2/3 synced status`；0812 新 H100 结果见 `## 2026-08-12 SCAPE H100 fresh confirm + real influence final`。

---

## 2026-08-13 SCAPE 0813 execution status

> 根据 `SCAPE/0813/SCAPE-0813-五机协调.md` 和 `SCAPE/0813/SCAPE-0813-H100-3.md` 更新。本节只记录 0813 调度下的 setting / 结果 / 结论，并明确区分 **已完成**、**进行中**、**未开始/阻塞**。

### Setting
- repo: `/mnt/songzijun/Capability_Evolution/SCAPE` on branch `sync/h100-20260812`
- GitHub remote: `https://github.com/ZijunSong/Capability_Evolution.git`
- pushed sync head: `31a05e9e63339d62f5ac78e743ed28ef6effe093` (`origin/sync/h100-20260812` after 0813 artifact summarizer commit); previous H100-3 sync head was `66047fc5d4f7ee20c3111d90c0fea13f0c44c88e`
- required H100 snapshot ancestor: `0f0934bd9f7a985af747e18dda9c2c666a9c24ba`
- official Chroma credentials: unavailable (`OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` missing) -> `OFFICIAL_CHROMA_BLOCKED=true`; continue local/HF mechanism experiments only
- H100-3 input: `outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl`
- H100-3 target components: `evidence_graph`, `importance_tagging`, `verify_tool`
- H100-3 GPU policy: skipped GPU0-7 rescore because per-state JSONL already contains `P_tool_name_full`, `P_tool_name_reduced`, `I_name`, `I_args`, and null statistics
- H100-4 verify follow-up setting: `VERIFY_INF_CONFIRM128`, component=`verify_tool`, seed=4414, n_queries=128, max_states_per_query=16, scorer=`hf_continuation_logprob`

### 已完成
| workstream | result | artifacts / evidence | conclusion |
|---|---|---|---|
| H100-3 Git audit / sync | branch `sync/h100-20260812` already pushed; remote head matches local `66047fc5d4f7ee20c3111d90c0fea13f0c44c88e` | `SCAPE/docs/H100_GITHUB_HANDOFF.md`; `git ls-remote --heads origin sync/h100-20260812` | H100 common scorer/attribution code is available on GitHub; `gh` CLI absent but HTTPS token push succeeded via temporary `GIT_ASKPASS` |
| H100-3 attribution aggregation | 3072 rows aggregated: 3 components x 1024 states | `outputs/h100_3_influence_attribution/RUN_MANIFEST.json`; `INFLUENCE_TOTALS.csv`; `INFLUENCE_BY_TOOL.csv`; `INFLUENCE_BY_ARGUMENT_CLASS.csv`; `INFLUENCE_BY_TURN.csv` | enough per-state statistics existed; no GPU rescore needed |
| evidence_graph attribution | `I_name_mean=0.038704`, `I_args_mean=0.117327`, late-turn args strongest | `outputs/h100_3_influence_attribution/EVIDENCE_GRAPH_ATTRIBUTION.md` | influence is not name-only; later ablations should retain name loss but emphasize argument tokens, especially read/end and doc-id / termination events |
| importance_tagging attribution | `I_name_mean=0.028771`, `I_args_mean=0.016560` | `outputs/h100_3_influence_attribution/IMPORTANCE_TAGGING_ATTRIBUTION.md` | positive influence is more tool-name than args-heavy in this aggregation; keep medium args loss for later ablation |
| verify_tool attribution | `I_name_mean=0.019043`, `I_args_mean=0.050669` | `outputs/h100_3_influence_attribution/VERIFY_TOOL_ATTRIBUTION.md` | H100-3 signal remains positive and args-heavy; still requires independent H100-4 verify confirmation before promoting over Candidate B |
| H20 loss recommendation | H20 V0 remains uniform name+args tool-token KL | `outputs/h100_3_influence_attribution/H20_LOSS_RECOMMENDATION.md` | attribution only informs later ablation/stratification; do not complicate first H20 V0 loss |
| H100-1/H100-2/H100-4 prior 0812 streams | confirmed still complete | `outputs/h100_1_contribution_confirm/STATUS_LIVE.md`; `outputs/h100_2_independent_repl/STATUS_LIVE.md`; `outputs/h100_4_influence_confirm/STATUS_LIVE.md` | no rerun needed under 0813 instructions |
| H20 lightweight torch record | complete artifact exists | `outputs/H20_LIGHTWEIGHT_TORCH_COMPLETE.json` | lightweight/proxy result exists but is not official Chroma parity |
| 0813 artifact summarizer | script committed and pushed | `SCAPE/scripts/generate_0813_required_artifacts.py`; `.gitignore` now ignores `SCAPE-wt-h100-*/` | summarizer only consolidates existing artifacts into `outputs/scape_prestage_v2/0813_STATUS_SUMMARY.{json,md}`; it does not synthesize per-state measurements or turn failed `verify_tool` confirm into success |

### Repo / worktree hygiene
- `SCAPE-wt-h100-1/` and `SCAPE-wt-h100-2/` are git worktree checkout directories, not experiment artifacts. Parent repo must not commit them as ordinary nested directories.
- Both worktrees were checked and are clean: `SCAPE-wt-h100-1` on `exp/h1001-evidence-graph-placement`; `SCAPE-wt-h100-2` on `exp/h1002-candidate-b-utility`.
- `.gitignore` includes `SCAPE-wt-h100-*/` so other agents and servers do not treat these checkout directories as missing commits.
- `SCAPE/scripts/generate_0813_required_artifacts.py` was initially untracked, then revised to preserve real status boundaries before commit. In particular, `h100_4_verify_confirm` remains **进行中 / 异常待恢复** until the HF scorer environment is fixed and rerun.
- Latest pushed branch state after this hygiene update chain: `origin/sync/h100-20260812` at `31a05e9e63339d62f5ac78e743ed28ef6effe093` before this record-only follow-up commit.

### 正在进行 / 异常待恢复
| workstream | current state | blocker / recovery |
|---|---|---|
| H100-4 verify_tool independent CONFIRM128 | `outputs/h100_4_verify_confirm/STATUS_LIVE.md` shows `n_expected=1`, `n_finished=0`, `errors=1` | scorer invocation failed because `/opt/scape-h1003-hf-scorer/bin/python` is missing. Setting and split audit are frozen in `outputs/h100_4_verify_confirm/PREFLIGHT.json`; recover by pointing `scripts/run_h100_4_verify_confirm_hf.py --python` to a valid HF scorer env or rebuilding `/opt/scape-h1003-hf-scorer`, then rerun verify_tool only. |

### 未开始 / 阻塞
| workstream | status | reason |
|---|---|---|
| Official BrowseComp+ Chroma parity | **阻塞** | missing `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`; do not poll repeatedly and do not label local/HF evidence as official parity |
| H20 true-SCAPE Stage L for evidence_graph | **未开始 on this host** | 0813 coordination assigns this to H20; current local evidence only records lightweight/proxy completion, not true same-state tool-token OPD migration |
| H100-1 evidence graph placement decomposition | **已完成（LOCAL_COMPAT_ONLY）** | `outputs/h100_1_graph_decomp/` contains audit, G0-G4 decomposition, runtime cost, renderer robustness, manifest, SHA256SUMS |
| H100-2 Candidate-B utility resolution beyond existing replication | **已完成（LOCAL_COMPAT_ONLY）** | `outputs/h100_2_candidate_b_utility/` contains short-horizon utility per-state/summary, per-component reports, recommendation, SHA256SUMS |
| Targeted verify-event confirmation | **未开始** | natural verify_tool CONFIRM128 has not recovered yet; targeted stream should remain separate and follow natural confirm |

### 结论
- Current completed evidence still supports Candidate A=`evidence_graph`.
- H100-2 utility resolution makes the short-horizon Candidate-B picture more conservative: `subtractive_curation` ranks first on local utility, but the overall decision is `Behavior-only`, so it is not a clean full internalization win.
- H100-3 attribution says `evidence_graph`, `importance_tagging`, and `verify_tool` are all args-sensitive enough that name-only training would be insufficient; H20 V0 still starts with uniform name+args KL per coordination doc.
- `importance_tagging` remains a semantic candidate from real influence, but it is now one of several B-side signals rather than a sole freeze point.
- Runtime controls remain `chunk_neighbors` and `content_dedup`; do not promote them to first-round internalization targets.
- The only active local recovery item is H100-4 `verify_tool` CONFIRM128 environment repair; official Chroma remains credential-blocked.

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
