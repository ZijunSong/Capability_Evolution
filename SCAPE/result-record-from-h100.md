# SCAPE result-record

> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).

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

## 2026-08-11 SCAPE h100_1_preflight

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 pin: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- model: `pat-jj/harness-1` (not served in this step)
- benchmark: BrowseComp+
- split: planned `BCP_CAL200` / `BCP_HOLD200` / `BCP_SMOKE20`
- component: 10 Harness-1 components
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_1_contribution`

### Results
| metric | value |
|---|---:|
| pytest | 14 passed |
| visible_gpus | 8 |
| torch | 2.11.0+cu130 |
| vllm | 0.25.1 |
| transformers | 5.14.1 |
| harness1_checkout | 1 |
| retrieval_backend_available | 0 |
| real_gpu_rollouts_started | 0 |

### Paired
- No paired BrowseComp+ LOO rows were produced because the compatible Chroma retrieval backend is unavailable.

### Gate
UNRESOLVED / BLOCKED_RETRIEVAL_BACKEND

### Decision
Do not start H100-1 BrowseComp+ LOO or H20 training from fake/old SCOPE retrieval. Resume after `SCAPE_CHROMA_PATH` or `HARNESS1_CHROMA_PATH` points to a compatible Harness-1 BrowseComp+ Chroma backend.

---

## 2026-08-11 SCAPE pre-stage bootstrap

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- branch: `main`
- scope: preflight + local CAL64 bootstrap + candidate selection + stage queue scaffolding
- model: unset (no training launched)
- benchmark: provisional synthetic CAL64 only; Harness-1 smoke blocked by external code permission

### Results
| metric | value |
|---|---:|
| pytest | 14 passed |
| preflight_hard_fail | true |
| local_cal64_rows | 10 |
| candidate_selection | A=`subtractive_curation`, B=`importance_tagging` |
| stage_l_queue | written |
| stage_s_queue | written |
| stage_m_queue | written |
| experiments_started | 0 |

### Paired
- (none)

### Gate
UNRESOLVED

### Decision
继续补齐 Harness-1 运行权限与真实 retrieval backend，再启动 H100-1/H100-2/H100-3 正式实验。

---

## 2026-08-11 SCAPE H100-3 influence offline

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- branch: `main`
- stage: `h100_3_influence`
- component axis: same-state policy influence
- query count: 64
- max states per query: 4
- scorer: deterministic_offline_stub
- training: false

### Results
| metric | value |
|---|---:|
| components | 10 |
| n_queries | 64 |
| per_state_records | 2560 |
| null_control_report | written |
| dual_view_parity | written |
| sha256sums | written |
| experiments_started | 1 |

### Paired
- (none)

### Gate
UNRESOLVED

### Decision
H100-3 offline influence artifacts are complete; next step is to add / repair H100-1 and H100-2 aggregation entrypoints and then request explicit Harness-1 smoke authorization.

---

## 2026-08-11 SCAPE H100-3 repaired pre-stage + Stage L dry-run

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 commit: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- model: `pat-jj/harness-1` reference only; no real model training launched
- benchmark: local CAL64 provisional + deterministic offline same-state influence
- split: `LOCAL_CAL64` / `INF_CAL64`
- component: all 10 SCAPE Harness-1 components
- harness mask: full vs `H_-m` same-snapshot render
- seed: Stage L dry-run seeds 42/43
- decode: deterministic offline stub scorer
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/{h100_3_influence,scape_prestage,stage_l}`

### Results
| metric | value |
|---|---:|
| pytest_base_env | 14 passed |
| pytest_scape_env | 14 passed |
| h100_3_components | 10 |
| h100_3_per_state_records | 2560 |
| h100_3_sha256sum_check | pass |
| candidate_A | `subtractive_curation` |
| candidate_B | `importance_tagging` |
| stage_l_dry_run_cells | 12 |
| stage_l_dry_run_gate_components | 2 |
| visible_gpus | 8 |
| active_gpu_processes | 0 |
| real_training_started | 0 |

### Paired
- Same-state full/minus dual-view parity report written.
- No closed-loop paired Harness quality rows yet because compatible Chroma retrieval backend is still missing.

### Gate
UNRESOLVED / REAL_TRAINING_BLOCKED_BY_RETRIEVAL_BACKEND_AND_DRY_RUN_TRAINER

### Decision
Do not claim component retirement or launch Stage S/M real training until real Harness-1 retrieval backend and non-dry model training entrypoints are available.

---

## 2026-08-11 SCAPE H100-1 local BM25 contribution + H20 proxy stages

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 commit: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- benchmark: BrowseComp+
- backend: `local_bm25_compat` using `/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25`
- official Chroma parity: false
- split: `BCP_SMOKE1`, `BCP_SMOKE5`, `BCP_SMOKE20`, `BCP_CAL200`, `BCP_HOLD200`
- component: 10 Harness-1 components
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/{h100_1_contribution,scape_prestage,stage_l,stage_s,pareto}`

### Results
| metric | value |
|---|---:|
| pytest | 14 passed |
| BCP_SMOKE1_errors | 0 |
| BCP_SMOKE5_errors | 0 |
| BCP_SMOKE20_errors | 0 |
| BCP_CAL200_rows_full | 200 |
| BCP_CAL200_minus_components | 10 |
| BCP_HOLD200_rows_full | 200 |
| candidate_A | `subtractive_curation` |
| candidate_B | `importance_tagging` |
| stage_l_dry_cells | 8 |
| stage_s_proxy_gate | written |
| pareto_proxy_table | written |
| active_scape_processes_after_run | 0 |

### Paired
- H100-1 local BM25 contribution rows are paired by query id on `BCP_CAL200` for all 10 components.
- H100-3 same-state influence remains deterministic offline scorer.
- Stage S and Pareto outputs are synthetic/proxy artifacts, not real post-training closed-loop evidence.

### Gate
PARTIAL_PASS / LOCAL_COMPAT_COMPLETE / REAL_TRAINING_NOT_CLAIMED

### Decision
Use `subtractive_curation` and `importance_tagging` as current A/B candidates. Official Chroma parity and real model-weight training remain separate follow-up work; do not present dry/proxy Stage L/S/M as real post-training retirement.

---

## 2026-08-11 SCAPE qrel-backed pre-stage + H20 torch completion

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 commit: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- model: `tiny_torch_tool_policy` executable proxy; released `pat-jj/harness-1` checkpoint not fine-tuned
- benchmark: BrowseComp+ qrel-aligned local corpus
- split: `INF_CAL64`, `LOCAL_CAL64`, H20 Stage L/S/M proxy stages
- component: A=`subtractive_curation`, B=`importance_tagging`; H100-3 measured all 10 components
- harness mask: same-snapshot full vs `H_-m`
- seed: Stage L seeds 42/43; Stage S seeds 42/43/44; Stage M seeds per GPU queue
- decode: deterministic qrel-backed scorer for influence; torch KL/CE objectives for proxy training
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs`

### Results
| metric | value |
|---|---:|
| local_qrel_corpus_docs | 5040 |
| h100_3_qrel_components | 10 |
| h100_3_qrel_queries | 64 |
| h100_3_qrel_states | 2560 |
| h100_3_qrel_sha256 | pass |
| candidate_A | `subtractive_curation` |
| candidate_B | `importance_tagging` |
| stage_l_torch_cells | 12 |
| stage_l_subtractive_curation_gate | PASS |
| stage_l_importance_tagging_gate | PASS |
| stage_s_torch_jobs | 8 |
| stage_m_torch_jobs | 8 |
| pareto_frontier_points | 2 |
| h20_lightweight_sha256 | pass |
| active_gpu_processes_after_run | 0 |

### Paired
- qrel-backed influence uses the same `EnvironmentSnapshot` for full/minus render and does not run an independent full trajectory.
- Stage S four-grid and Pareto table are generated from executable lightweight torch proxy metrics and measured cost proxies.
- Official Harness-1 Cloud Chroma evaluation remains distinct from this SCAPE local corpus backend.

### Gate
PASS / LIGHTWEIGHT_TORCH_PROXY_COMPLETE

### Decision
Use the qrel-backed A/B selection and H20 torch artifacts as the completed local SCAPE experiment set; only claim released-checkpoint component retirement after a future official Harness-1 Cloud/Chroma fine-tuning run.

---

## 2026-08-11 SCAPE official Harness-1 path preparation

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 commit: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- model: `pat-jj/harness-1`
- serving: vLLM localhost only, `--trust-remote-code` authorized by user
- benchmark: BrowseComp+
- evaluation code: `external/harness-1/inference/evaluate_harness1_vllm.py` authorized by user
- output/config: `docs/OFFICIAL_HARNESS1_STATUS.md`, `scripts/serve_harness1_vllm_local.sh`, `scripts/run_official_harness1_browsecompplus_vllm.sh`, `scripts/check_official_env_presence.py`

### Results
| metric | value |
|---|---:|
| official_dependencies_installed | 1 |
| vllm_import | 1 |
| torch_import | 1 |
| chromadb_import | 1 |
| tinker_import | 1 |
| baseten_client_import | 1 |
| hf_tls_with_system_ca | pass |
| browsecomp_paths_present | 4 |
| missing_required_secret_vars | 3 |
| model_download_started | 1 |
| official_vllm_started | 0 |
| official_eval_started | 0 |

### Paired
- Not yet applicable; official BrowseComp+ evaluation has not started.

### Gate
UNRESOLVED / WAITING_MODEL_WEIGHTS_AND_CLOUD_CHROMA_CREDENTIALS

### Decision
Continue monitoring `pat-jj/harness-1` weight download. Start localhost vLLM smoke after all shards are present; start official BrowseComp+ eval only after `OPENAI_API_KEY`, `CHROMA_API_KEY`, and `CHROMA_DATABASE` are provided.

---

## 2026-08-11 SCAPE official Harness-1 monitor tick

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- model target: `/mnt/songzijun/models/pat-jj_harness-1`
- benchmark: BrowseComp+
- monitor cadence: session cron `0ee9ffb5` every 17 minutes

### Results
| metric | value |
|---|---:|
| tokenizer_json_present | 0 |
| safetensor_shards_present | 0 |
| required_safetensor_shards | 9 |
| active_hf_download_processes | 0 |
| active_vllm_processes | 0 |
| active_official_eval_processes | 0 |
| gpu_processes | 0 |
| missing_required_secret_vars | 3 |
| local_qrel_proxy_complete | 1 |
| official_vllm_started | 0 |

### Paired
- Not applicable; no official run started during this monitor tick.

### Gate
UNRESOLVED / OFFICIAL_MODEL_WEIGHTS_ABSENT

### Decision
Do not start vLLM until tokenizer and all 9 safetensor shards are present. Continue session monitoring; provide `HUGGINGFACE_TOKEN`, a reachable mirror, or manually stage weights to unblock official serving.

---

## 2026-08-11 SCAPE H100-2 replication + coalition

### Setting
- repo path: /mnt/songzijun/Capability_Evolution/SCAPE
- module utility root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility
- exact budget root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial
- replication benchmark: fresh200 module utility
- coalition benchmark: exact-budget factorial
- seed: 42
- decode: temperature=0, top_p=1, do_sample=false
- output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_replication_coalition

### Results
| metric | value |
|---|---:|
| replication_modules | 4 |
| coalition_rows | 6 |
| canonical_floor | 0 |
| report_written | 1 |

### Paired
- replication_root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility
- interaction_root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial

### Gate
UNRESOLVED

### Decision
Completed H100-2 consolidation on existing fresh200/factorial outputs; no new training or retrieval was launched.

---

## 2026-08-11 SCAPE H20 lightweight torch experiments

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 pin: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- trainer: `tiny_torch_tool_policy` (`scripts/run_h20_lightweight_experiments.py`)
- benchmark: SCAPE lightweight same-state / tool-token OPD proxy
- split: Stage L `512/2000/8000`; Stage S `2000`; Stage M `2000`
- components: `subtractive_curation`, `importance_tagging`
- GPUs used: 0-7
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/{stage_l,stage_s,stage_m,pareto}`

### Results
| metric | value |
|---|---:|
| stage_l_seeded_cells | 12 |
| stage_l_pass_components | 2 |
| stage_s_real_checkpoint_jobs | 8 |
| stage_m_real_checkpoint_jobs | 8 |
| stage_s_gate_strong_pass | 2 |
| stage_m_best_L_m | 0.9620677517166649 |
| pareto_torch_table | written |
| checkpoint_artifacts | 28 |
| gpu_processes_at_end | 0 |
| pytest | 14 passed |

### Paired
- Stage L: two seeds per candidate, all three sample sizes improved divergence and held invalid-tool rate flat.
- Stage S: four-grid evaluation produced STRONG_PASS for both candidates.
- Stage M: coalition / annealing jobs all wrote checkpoints and summaries.

### Gate
PARTIAL_PASS / REAL_LIGHTWEIGHT_TRAINING_COMPLETE / OFFICIAL_HARNESS_PARITY_STILL_SEPARATE

### Decision
Lightweight torch training for Stage L/S/M is now complete and checkpointed. Next missing item for full official parity remains Harness-1 official Chroma Cloud + released model checkpoint wiring; do not confuse that with the completed lightweight training path.

## 2026-08-11 SCAPE H20 lightweight torch L/S/M/Pareto complete

### Setting
- repo path: /mnt/songzijun/Capability_Evolution/SCAPE
- trainer: tiny_torch_tool_policy
- dry_run: False
- candidate A: subtractive_curation
- candidate B: importance_tagging
- stage_l output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/stage_l/GATE_L_TORCH.json
- stage_s output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/stage_s/GATE_S_TORCH.json
- stage_m output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/stage_m/STAGE_M_TORCH.json
- pareto output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/pareto/MAIN_TABLE_TORCH.json
- checksum: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/H20_LIGHTWEIGHT_SHA256SUMS

### Results
| metric | value |
|---|---:|
| stage_l_subtractive_curation_pass | True |
| stage_l_importance_tagging_pass | True |
| stage_s_subtractive_curation_verdict | STRONG_PASS |
| stage_s_importance_tagging_verdict | STRONG_PASS |
| stage_m_best_L_m | 0.962068 |
| stage_m_best_agreement_post | 0.894531 |
| h20_elapsed_s | 2290.21 |
| sha256_verified | True |

### Paired
- subtractive_curation_S2_quality: 0.03021927059075972
- importance_tagging_S2_quality: 0.030179612797523736
- scape_quality: 0.027333067335609727
- scape_cost: 12280.0

### Gate
PASS / LIGHTWEIGHT_TORCH_COMPLETE

### Decision
H20 lightweight executable L/S/M/Pareto run is complete; use these as completed lightweight artifacts, not official Harness-1 checkpoint retirement evidence.

---

## 2026-08-12 SCAPE harness-1 model restore + preflight + smoke

### Setting
- repo path: `/mnt/songzijun/Capability_Evolution/SCAPE`
- upstream Harness-1 commit: `8ac4012167858f6478fb2a8fd840e4550e2af161`
- model source: `/mnt/songzijun/models/harness-1.tar.gz`
- restored model root: `/mnt/songzijun/models/harness-1-extracted/harness-1`
- serving env: `/opt/vllm-qwen3-1.7b`
- model: `harness-1`
- output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_1_official_vllm`

### Results
| metric | value |
|---|---:|
| restored_tokenizer_json | 1 |
| restored_safetensor_shards | 9 |
| pytest | 14 passed |
| preflight_ok | 1 |
| harness1_vllm_smoke | pass |
| /v1/models | 200 |
| active_gpu_processes_after_smoke | 8 |

### Paired
- Official BrowseComp+ evaluation environment variables are still absent, so no official eval started.
- `SCAPE_RETRIEVAL_CORPUS` is available for local preflight only; Chroma / OpenAI credentials are still required for official evaluation.

### Gate
PASS / MODEL_RESTORED_AND_VLLM_SMOKE_COMPLETE

### Decision
Continue by wiring or providing the official BrowseComp+ credentials and paths, then start the smoke or short official eval on the restored local harness-1 service.

## 2026-08-12 SCAPE H100-2 replication + coalition

### Setting
- repo path: /mnt/songzijun/Capability_Evolution/SCAPE
- module utility root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility
- exact budget root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial
- replication benchmark: fresh200 module utility
- coalition benchmark: exact-budget factorial
- seed: 42
- decode: temperature=0, top_p=1, do_sample=false
- output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_replication_coalition

### Results
| metric | value |
|---|---:|
| replication_modules | 4 |
| coalition_rows | 6 |
| canonical_floor | 0 |
| report_written | 1 |

### Paired
- replication_root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility
- interaction_root: /mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial

### Gate
UNRESOLVED

### Decision
Completed H100-2 consolidation on existing fresh200/factorial outputs; no new training or retrieval was launched.

## 2026-08-12 SCAPE H100-3 influence cross-map completion

### Setting
- repo path: /mnt/songzijun/Capability_Evolution/SCAPE
- inputs: H100-1 contribution + H100-3 influence
- output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/CONTRIBUTION_INFLUENCE_MAP.md

### Results
| metric | value |
|---|---:|
| map_written | 1 |
| quadrants | 4 |
| components | 10 |

### Paired
- h100_1_contribution: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_1_contribution/COMPONENT_CONTRIBUTION.md
- h100_3_influence: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_3_influence/H100_3_INFLUENCE_REPORT.md

### Gate
PASS

### Decision
Cross-node contribution/influence map completed from frozen results only.

## 2026-08-12 SCAPE H100-2 placement stability completion

### Setting
- repo path: /mnt/songzijun/Capability_Evolution/SCAPE
- inputs: H100-2 replication + coalition
- output: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_replication_coalition/PLACEMENT_STABILITY.md

### Results
| metric | value |
|---|---:|
| placement_report_written | 1 |
| replication_modules | 4 |
| coalition_rows | 6 |

### Paired
- replication_report: /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_replication_coalition/REPLICATION_REPORT.md

### Gate
PASS

### Decision
Placement-stability summary completed from frozen replication and coalition outputs only.

---

## 2026-08-12 SCAPE H100-1/2/3 todo1 consolidated status

### Scope
- Source instructions: `H100-1-0811-todo1.md`, `H100-2-0811-todo1.md`, `H100-3-0811-todo1.md`.
- Purpose: one consolidated status section for experiment settings, produced artifacts, quantitative results, conclusions, and remaining not-started work.
- Status vocabulary in this section:
  - **已完成**: artifact exists and the corresponding run/report finished with `errors=0` or an explicit `PASS`/`completed` status.
  - **进行中/阻塞**: prerequisites are prepared or partial artifacts exist, but the official/target run is waiting on external resources or credentials.
  - **未开始**: no corresponding run artifact was found, or the todo requested an optional follow-up that was not launched.

### Overall status table
| Workstream | Todo target | Current status | Output / evidence | Notes |
|---|---|---|---|---|
| H100-1 Phase 0/1 | Harness-1 reproduction + 10-component LOO contribution map | **已完成（local BM25 compat）/ 进行中（official Chroma parity）** | `outputs/h100_1_contribution/{RUN_MANIFEST.json,STATUS_LIVE.md,COMPONENT_CONTRIBUTION.*,SHA256SUMS}` | Local BM25 compatibility contribution sweep finished for all 10 components, n=200, errors=0. Official Harness-1 BrowseComp+ Chroma Cloud evaluation has not started because Chroma/OpenAI credentials are still missing. |
| H100-2 | independent replication + coalition interaction | **已完成（frozen consolidation）/ 部分偏离原 10-component REPL200 plan** | `outputs/h100_2_replication_coalition/{RUN_MANIFEST.json,STATUS_LIVE.md,SECONDARY_BENCHMARK_SELECTION.md,LOO_REPLICATION.csv,COALITION_INTERACTION.csv,REPLICATION_REPORT.md,PLACEMENT_STABILITY.md,SHA256SUMS}` | Consolidated existing fresh200 module-utility and exact-budget factorial outputs: 4 replicated modules + 6 coalition rows, errors=0. No new training/retrieval was launched. |
| H100-3 | same-environment-state policy influence map | **已完成（offline deterministic INF_CAL64）** | `outputs/h100_3_influence/{RUN_MANIFEST.json,STATUS_LIVE.md,INFLUENCE_BY_COMPONENT.*,INFLUENCE_PER_STATE.jsonl,H100_3_INFLUENCE_REPORT.md,SHA256SUMS}` | All 10 components measured, 64 queries × 4 states/query = 256 states/component, errors=0. Uses deterministic offline scorer rather than released Harness-1 model logprob path. |
| H100-1 × H100-3 | contribution/influence quadrant map | **已完成** | `outputs/CONTRIBUTION_INFLUENCE_MAP.md` | Merged frozen H100-1 local BM25 contribution with H100-3 offline influence; 10 components assigned to four quadrants. |
| Official Harness-1 serving | restore model and local vLLM smoke | **已完成（smoke）/ 进行中（official eval）** | `outputs/h100_1_official_vllm`; prior section “model restore + preflight + smoke” | Model restored from `/mnt/songzijun/models/harness-1.tar.gz`, 9 safetensor shards present, vLLM smoke passed. Official BrowseComp+ eval remains blocked by missing `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`. |
| H100-3 confirm/targeted extensions | `INF_CONFIRM128`, first-search/curate targeted influence, verify targeted state mining | **未开始** | none found | Todo permits confirm/targeted follow-ups only when CAL64 warrants expansion; no separate confirm/targeted artifact was found. |
| H100-1/H100-2 official full parity | official Chroma-backed BrowseComp+ LOO/replication across requested masks | **未开始/阻塞** | none found beyond local/proxy outputs | Must not be claimed from local BM25 or frozen SCOPE consolidation; requires official retrieval credentials and evaluator path. |

### H100-1 setting and results
- Run id: `h100_1_local_bm25_contribution_20260811`.
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`, git `61f7741a6be2e2e62a4c8b0da86a651791a9117f` with dirty worktree at manifest time.
- Python/env: `/opt/bishop-harness/bin/python`, Python 3.11.6, torch 2.11.0+cu130, vLLM 0.25.1, 8×H100 visible.
- Backend: `local_bm25_compat`; **not** official Chroma Cloud parity.
- Split/seed: local BrowseComp+ compatible CAL200, seed 1101; smoke rows for 1/5/20 also previously recorded with errors=0.
- Decode/budget: deterministic compatibility path; no training and no model-weight modification.
- Status artifact: `outputs/h100_1_contribution/STATUS_LIVE.md` reports `n_expected=10`, `n_finished=10`, `remaining=0`, `errors=0`.

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
- Baseline/component LOO is complete only for the **local BM25 compatibility** path.
- Strongest local contribution signal by combined `Δ curated + Δ trajectory + Δ final` is `auto_populate_first_search`, followed by `content_dedup`, then the small positive group around `adaptive_rerank_instruction`, `evidence_graph`, and `chunk_neighbors`.
- `sentence_compress`, `verify_tool`, and `token_budget_marker` are neutral in the local quality metrics.
- Do **not** claim official Harness-1 reproduction/parity from this run; official Chroma Cloud BrowseComp+ remains blocked.

### H100-2 setting and results
- Run id: `h100_2_replication_coalition_20260811`.
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`, git `61f7741a6be2e2e62a4c8b0da86a651791a9117f` with dirty worktree at manifest time.
- Python/env: `/opt/vllm-qwen3-1.7b/bin/python`, Python 3.12.13, torch 2.11.0+cu130, vLLM 0.25.1, 8×H100 visible.
- Replication input: `/mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility` fresh200 module-utility track.
- Coalition input: `/mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial` exact-budget factorial track.
- Secondary benchmark selection: source-domain BrowseComp+ compatibility fresh200 for replication plus transfer/diagnostic exact-budget factorial for interaction.
- Status artifact: `outputs/h100_2_replication_coalition/STATUS_LIVE.md` reports `n_expected=5`, `n_finished=5`, `remaining=0`, `errors=0`.

| module | ablated condition | n | Δ final-answer recall | Δ trajectory recall | Δ reward | paired final W/L/T | paired trajectory W/L/T | Status |
|---|---|---:|---:|---:|---:|---|---|---|
| context_budget | minus_context_budget | 200 | +0.003345 | -0.000671 | +0.022175 | 8/4/188 | 28/28/144 | 已完成 / REPLICATED |
| evidence_state | minus_evidence_state | 200 | -0.001786 | +0.002148 | +0.014134 | 8/5/187 | 27/27/146 | 已完成 / REPLICATED |
| verification | minus_verification | 200 | +0.010575 | +0.016813 | +0.054930 | 13/6/181 | 33/23/144 | 已完成 / REPLICATED |
| retrieval_rerank | minus_retrieval_rerank | 200 | -0.005571 | -0.007124 | -0.008528 | 6/6/188 | 25/30/145 | 已完成 / REPLICATED |

| model | budget | N | Q | QS | sequential interaction gap | interpretation | Status |
|---|---:|---:|---:|---:|---:|---|---|
| qwen3_1p7b | 256 | 0.0300 | 0.0300 | 0.0200 | -0.0100 | diminishing_returns | 已完成 |
| qwen3_1p7b | 512 | 0.0300 | 0.0500 | 0.0400 | -0.0300 | diminishing_returns | 已完成 |
| qwen3_1p7b | 1024 | 0.0400 | 0.0400 | 0.0400 | +0.0000 | near_additive | 已完成 |
| qwen3_30b | 256 | 0.0100 | 0.0000 | 0.0000 | +0.0100 | super_additive | 已完成 |
| qwen3_30b | 512 | 0.0200 | 0.0200 | 0.0000 | -0.0200 | diminishing_returns | 已完成 |
| qwen3_30b | 1024 | 0.0300 | 0.0100 | 0.0100 | +0.0200 | super_additive | 已完成 |

#### H100-2 conclusion
- `verification` is the clearest stable-positive replicated module: positive on final-answer recall, trajectory recall, and reward.
- `context_budget` and `evidence_state` are placement/domain-sensitive because metric signs differ across axes.
- `retrieval_rerank` is interaction-sensitive/benchmark-sensitive in this consolidation: both recall deltas are negative in the replicated table.
- Coalition evidence is mixed and mostly diminishing/near-additive; it should be treated as a reporting-level interaction note rather than strong synergy.
- This run does not equal the full original H100-2 10-component REPL200 plan; it is a completed consolidation of already-frozen module utility and factorial outputs.

### H100-3 setting and results
- Run id: `h100_3_influence_offline_cal64`.
- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`, git `61f7741a6be2e2e62a4c8b0da86a651791a9117f` with dirty worktree at manifest time.
- Python/env: `/root/miniforge3/bin/python`, Python 3.13.13; manifest records no torch/vLLM dependency for this offline scorer path.
- Data scale: INF_CAL64, 64 queries/component, max 4 states/query, 256 states/component, 2,560 per-state records overall.
- Scorer: `deterministic_offline_stub`; no training.
- Status artifact: `outputs/h100_3_influence/STATUS_LIVE.md` reports `n_expected=10`, `n_finished=10`, `remaining=0`, `errors=0`.

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
- Highest same-state influence: `subtractive_curation` and `importance_tagging`, well above the rest.
- Mid-tier influence: `verify_tool`, `chunk_neighbors`, `evidence_graph`, `content_dedup`.
- Lowest influence: `adaptive_rerank_instruction`, `sentence_compress`, `token_budget_marker`.
- The influence map is valid as an offline deterministic same-state renderer/scorer artifact; it is **not** a released Harness-1 logprob enumeration run.
- Optional confirm/targeted work (`INF_CONFIRM128`, first-search/curate targeting, verify targeted mining) was not found and is marked **未开始**.

### Cross-map conclusions from H100-1 + H100-3
- Source: `outputs/CONTRIBUTION_INFLUENCE_MAP.md`.
- Thresholds: contribution median `0.001611`, influence median `0.007540`.

| quadrant | components | conclusion | Status |
|---|---|---|---|
| High Δ, High I | `evidence_graph`, `chunk_neighbors` | strongest balanced migration candidates under frozen local/offline evidence | 已完成 |
| High Δ, Low I | `auto_populate_first_search`, `content_dedup`, `adaptive_rerank_instruction` | quality/runtime/state-mechanism effects clearer than same-state policy shift | 已完成 |
| Low Δ, High I | `subtractive_curation`, `importance_tagging`, `verify_tool` | policy-changing, but quality lift is weak in local contribution metrics; review before keeping/removing | 已完成 |
| Low Δ, Low I | `sentence_compress`, `token_budget_marker` | direct-removal candidates in this frozen local/offline analysis | 已完成 |

### Active/running process check
- A process check at consolidation time found no active SCAPE/H100/Harness/vLLM/evaluate process and no GPU compute process output from `nvidia-smi --query-compute-apps`.
- Therefore no SCAPE experiment from these three todo files is currently confirmed running in this session.

### Final decision / next actions
- **已完成**: H100-1 local BM25 compatibility contribution sweep; H100-2 frozen replication/coalition consolidation; H100-3 offline deterministic influence sweep; H100-1×H100-3 contribution/influence map; H100-2 placement stability summary; Harness-1 model restore + local vLLM smoke.
- **进行中/阻塞**: official Harness-1 BrowseComp+ evaluation path, because official Chroma/OpenAI-compatible credentials are absent even though the restored model and local vLLM smoke are ready.
- **未开始**: official Chroma-backed H100-1/H100-2 parity runs, H100-3 `INF_CONFIRM128`, targeted first-search/curate influence, targeted verify mining, and any real released-checkpoint component retirement/training claim.
- Do not present local BM25/offline/proxy evidence as official Harness-1 Cloud/Chroma parity or as final released-checkpoint component retirement evidence.
