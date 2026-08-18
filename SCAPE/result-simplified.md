# SCAPE Result Simplified

## 2026-08-18 H20 clean-init AUTO OPD (`h20_clean_auto_0817`) — final

Status: **已完成** GPU 主评测 + aggregate；`PHASE=DONE`. Machine `8×H20`. Repo `/data/ppnm/Capability_Evolution/SCAPE`. Spec: `todo/0817/H20_clean_init_AUTO_OPD_next_round_20260817.md`. Canonical outputs: `outputs/h20_clean_auto_0817/`. Handoff written `2026-08-18T14:17:06+0800`. This is the H20-only cross-initialization / cross-model line; it does **not** repeat H100-1/2/3/4 jobs.

Discard the `2026-08-18T00:51:01+0800` handoff. That run stacked OPD LoRA on raw `gpt-oss-20b` without merging Clean-SFT (`invalid≈0.94`, recall=0). Paper-grade numbers below use the reload-fix stack: `gpt-oss-20b → merge FULL_S42 → OPD LoRA`.

### Setting

```text
init:            openai/gpt-oss-20b + Harness-1 public SFT only (not released pat-jj/harness-1)
CLEAN_AUTO_BASE: CLEAN_FULL_S42
                 outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint
component:       auto_populate_first_search
objective:       reverse 8-way Route-KL; lambda_args=0; lambda_anchor=0.05
LoRA:            actual gpt-oss LLM weights, r=8, lr=1e-5, 1 epoch
Student input:   reduced / no privilege at train and inference
inference:       student_inference_privilege=false
retriever:       LOCAL_COMPAT_ONLY in-process overlap ranker over per-query doc_store
                 (not official Chroma)
reward:          evidence/qrel curated recall; final_answer gold = N/A (not written as 0)
max_steps:       6 (sanity 10 done; 12 not run)
splits:          BASE_EVAL n=128; AUTO unique states=438 (train 350 / valid 43 / test 45);
                 real DEV=128; real TEST=112 (all remaining unique queries, no resampling)
seeds:           unshuffled 42/43/44/45; shuffled-target 42/43/44/45
                 same unique states / query ids / update budget / target marginal /
                 reverse Route-KL / LoRA rank-alpha / lr / epochs
route space:     fan_out_search, search_corpus, grep_corpus, read_document,
                 review_docs, curate, verify, end_search
```

### Results

#### Step A — Harmony / Base Gate `[已完成]`

0814 n=4 smoke `parse≈0.75` was a **prompt/parser contract bug**. Official Harmony `build_context` + `render_conversation_for_completion`, stop on `<|call|>`/`<|return|>`:

```text
parser contract tests: 4/4 pass
  canonical Harmony tool call -> pass
  canonical end_search        -> pass
  malformed tool name         -> fail_legal
  analysis prose              -> unparsed

BASE_EVAL_128 (n=128):
  RAW_GPT_OSS      parse=0.047  legal=0.039  invalid=0.961  gate=FAIL
  CLEAN_FULL_S42   parse=1.000  legal=1.000  invalid=0.000  gate=PASS  ← CLEAN_AUTO_BASE
  CLEAN_FULL_S43   parse=1.000  legal=1.000  invalid=0.000  gate=PASS
  CLEAN_TOOL_S42   parse=0.727  legal=0.688  invalid=0.312  gate=FAIL
  CLEAN_TOOL_S43   parse=0.992  legal=0.953  invalid=0.047  gate=FAIL
```

FULL first-action mass is almost entirely `fan_out_search`/`search_corpus`. TOOL is diagnostic only.

#### Step B — FORMAT_REPAIR `[未开始 / 按规范跳过]`

Base Gate already PASS. Spec forbids another FULL/TOOL SFT and forbids FORMAT_REPAIR unless the gate fails.

#### Step C — fresh AUTO on-policy data `[已完成]`

```text
n_raw_unique=438   (target >=512; below target; no duplicated rows)
n_train=350  n_valid=43  n_test=45
n_effect_active=390
n_resampled_duplicate=0
query_disjoint_from_base_eval=true
used_rl=false
```

#### Step D — same-`xi_t` value K4/K8 `[已完成]`

```text
K4 n=350 mean=+0.0254  CI=[+0.0043, +0.0538]
K8 n=350 mean=+0.0115  CI=[-0.0102, +0.0323]
effect-active K4 mean=+0.0232 ci_low=-0.0060
effect-active K8 mean=+0.0079 ci_low=-0.0179
replay_noise_proxy=0
direction_consistent=true
AUTO_CLEAN_VALUE_GATE.pass=true
```

K8 95% CI includes 0. Gate pass is K4 `CI_low>0` + both means `> replay_noise` + K4/K8 sign agreement.

#### Step E/F — actual gpt-oss LoRA + shuffled-target `[已完成]`

Eight reloadable actual-LoRA cells (not `route_head.pt`):

```text
UNSHUFFLED Route-KL
  s42 loss=0.113  d_post=-0.0012  L_m=0.347
  s43 loss=0.109  d_post=+0.0019  L_m=2.017
  s44 loss=0.091  d_post=+0.0011  L_m=1.571
  s45 loss=0.128  d_post=+0.0026  L_m=2.376

SHUFFLED Route-KL (matched contract)
  s42 loss=0.261  d_post=-0.0015  L_m=0.196
  s43 loss=0.210  d_post=+0.0005  L_m=1.250
  s44 loss=0.233  d_post=+0.0058  L_m=4.080
  s45 loss=0.216  d_post=+0.0003  L_m=1.150
  shuffle fixed points: 1/350 = 0.0029
```

Same-state `L_m` / `d_post` are **not** the paper main result.

#### Step G — actual-model real multi-step closed-loop `[已完成]`

Contract: `real_eval/AUTO_CLEAN_REAL_EVAL_CONTRACT.md`. Reload-fix parent adapter = FULL_S42. Smoke: LoRA actually loaded; AUTO privilege absent; Base ≠ Student sequences; scorer non-constant.

16-query smoke:

```text
SMOKE_BASE   n=16  recall=0.2063  invalid=0.0729  search=3.31
SMOKE_UNSH   n=16  recall=0.1300  invalid=0.0521  search=2.94   (unshuffled s42)
```

DEV n=128, max_steps=6, evidence/qrel recall (reload-fix):

```text
CLEAN_BASE                 0.1913  invalid=0.0638  search=2.84
CLEAN_FULL_HARNESS         0.1913  invalid=0.0638  search=2.84   (same FULL_S42 weights)
AUTO_CLEAN_UNSHUFFLED_s42  0.1617  invalid=0.0443  search=2.70
AUTO_CLEAN_UNSHUFFLED_s43  0.1714  invalid=0.0273  search=2.69   ← best unshuffled
AUTO_CLEAN_UNSHUFFLED_s44  0.1710  invalid=0.0443  search=2.75
AUTO_CLEAN_UNSHUFFLED_s45  0.1663  invalid=0.0312  search=2.65
AUTO_CLEAN_SHUFFLED_s42    0.1745  invalid=0.0612  search=2.82
AUTO_CLEAN_SHUFFLED_s43    0.1998  invalid=0.0638  search=2.71   ← highest student cell
AUTO_CLEAN_SHUFFLED_s44    0.1866  invalid=0.0508  search=2.81
AUTO_CLEAN_SHUFFLED_s45    0.1575  invalid=0.0443  search=2.79

unshuffled mean            0.1676
shuffled mean              0.1796
```

TEST n=112, max_steps=6:

```text
CLEAN_BASE_TEST                 0.1497  invalid=0.0625  search=2.66
CLEAN_FULL_HARNESS_TEST         0.1497  invalid=0.0625  search=2.66
AUTO_CLEAN_UNSHUFFLED_s42_TEST  0.1221  invalid=0.0491  search=2.85
AUTO_CLEAN_UNSHUFFLED_s43_TEST  0.1476  invalid=0.0402  search=2.75
AUTO_CLEAN_UNSHUFFLED_s44_TEST  0.1387  invalid=0.0417  search=2.70
AUTO_CLEAN_UNSHUFFLED_s45_TEST  0.1281  invalid=0.0327  search=2.79
AUTO_CLEAN_SHUFFLED_s42_TEST    0.1167  invalid=0.0446  search=2.82
AUTO_CLEAN_SHUFFLED_s43_TEST    0.1423  invalid=0.0625  search=2.84
AUTO_CLEAN_SHUFFLED_s44_TEST    0.1621  invalid=0.0521  search=2.79
AUTO_CLEAN_SHUFFLED_s45_TEST    0.1083  invalid=0.0417  search=2.88

unshuffled mean                 0.1341
shuffled mean                   0.1323
```

Paired bootstrap 95% CI, DEV query-level vs CLEAN_BASE (n=128):

```text
UNSHUFFLED_s42  delta=-0.0296  CI=[-0.0764, +0.0107]
UNSHUFFLED_s43  delta=-0.0198  CI=[-0.0604, +0.0214]
UNSHUFFLED_s44  delta=-0.0203  CI=[-0.0666, +0.0150]
UNSHUFFLED_s45  delta=-0.0250  CI=[-0.0691, +0.0141]
SHUFFLED_s42    delta=-0.0168  CI=[-0.0587, +0.0211]
SHUFFLED_s43    delta=+0.0085  CI=[-0.0329, +0.0465]
SHUFFLED_s44    delta=-0.0046  CI=[-0.0455, +0.0355]
SHUFFLED_s45    delta=-0.0338  CI=[-0.0828, +0.0054]
UNSH_s42 vs SH_s42  delta=-0.0128  CI=[-0.0505, +0.0205]
```

All four unshuffled CIs include 0; point estimates are all negative vs Base. Unshuffled does not beat shuffled.

Sanity max_steps=10 (n=16 smoke queries):

```text
CLEAN_BASE_S10   recall=0.2140  invalid=0.0750  search=4.44
UNSH_S43_S10     recall=0.2050  invalid=0.0500  search=4.06
max_steps=12     not run
```

Longer horizon does not reverse the ranking. Completed DEV/TEST episodes terminate at `max_steps` (no early-`end_search` artifact). `final_answer=N/A`.

### Conclusion

```text
A  clean gpt-oss tool Base Gate PASS              true
B  AUTO proper value K4/K8 positive               true  (K8 CI includes 0)
C  actual gpt-oss LoRA real closed-loop > Base    false
D  >=2 unshuffled seeds same positive direction   false  (0/4 DEV, 0/4 TEST)
E  unshuffled > shuffled on real closed-loop      false
F  invalid-tool no material regression            true
G  student_inference_privilege=false              true

final_decision:              STOP_CLEAN_AUTO_REAL_TASK_NO_GAIN
recommended_for_main_table:  false
best_unshuffled_checkpoint:  phase_E/unshuffled_s43/lora_checkpoint
CLEAN_INIT_AUTO_TRANSFER_PASS: not written
```

What this round does support:

- Clean `gpt-oss-20b` + public SFT **can** emit a legal Harmony tool channel once the evaluator matches official Harmony. 0814 `CLEAN_BASE_BLOCKED` / parse≈0.75 does not survive the repaired contract.
- AUTO same-state value is a small positive K4 signal on clean occupancy; it does **not** become a real multi-step Search win after reverse Route-KL LoRA.
- The paper main result is actual Student weights + no-privilege inference + real Search + external metric. Under that contract, unshuffled AUTO OPD loses to CLEAN_BASE on both DEV and TEST, and does not beat the matched shuffled-target control.

What this round does **not** support, and must not be packaged as a main-table win:

- FORMAT_REPAIR skip / Base Gate PASS alone
- same-state Route-KL `L_m` / `d_post`
- value-positive K4/K8
- the discarded 00:51 broken-load run
- `CLEAN_FULL_HARNESS` as a distinct system (it is the same FULL_S42 weights as CLEAN_BASE on this clean-init machine)

Spec still allows **one** substantive redesign (continuation-aware state selection / multi-step windows / hard negatives, then a new shuffled control). That redesign was **not** launched in this round.

### Completed / not-started board

| Item | Status |
|---|---|
| Harmony runtime audit + parser tests | **已完成** |
| BASE_EVAL_128 five-way n=128 | **已完成** |
| FORMAT_REPAIR FR_A–D | **未开始** (Base Gate PASS, skipped) |
| Fresh AUTO collect / split | **已完成** (438 unique < 512 target) |
| Value K4/K8 fork-replay + gate | **已完成** |
| Unshuffled LoRA seeds 42–45 | **已完成** |
| Shuffled-target LoRA seeds 42–45 | **已完成** |
| Real-eval contract freeze | **已完成** |
| 16-query smoke (reload-fix) | **已完成** |
| DEV n=128 all required rows | **已完成** |
| TEST n=112 all required rows | **已完成** |
| max_steps=10 Base + unsh s43 | **已完成** |
| max_steps=12 | **未开始** (optional sanity) |
| Paired bootstrap 95% CI | **已完成** |
| Case analysis 20–25 per contrast class | **未开始** (stub only in `AUTO_CLEAN_CASE_ANALYSIS.md`) |
| `result-record.md` append | **未开始** |
| One substantive OPD redesign | **未开始** |
| Paper main-table recommendation | **否** (`recommended_for_main_table=false`) |

Artifacts: `outputs/h20_clean_auto_0817/` (`RUN_MANIFEST.json`, `STATUS_LIVE.md`, `H20_CLEAN_AUTO_HANDOFF.json`, `BEST_CLEAN_AUTO_STUDENT.json`, `SHA256SUMS`, `base_recovery/`, `auto_data/`, `value/`, `training/`, `real_eval/`).

## 2026-08-17 0816-2 final summary


Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`. This is the concise end-state summary for the 0816-2 round.

### Setting

- Main branch focus: `H100-1` AUTO actual-LoRA real closed-loop, `H100-2` importance_tagging proper K4/K8 gate, and H100-4 baseline reconciliation for Full Harness / Matched Text / OPHSD under the same actual-model contract.
- Shared contract target: actual Student model weights only, `student_inference_privilege=false`, real closed-loop execution, query-disjoint splits, matched max steps / reward / retriever / termination rules where applicable.
- Environment rule: GPU-heavy work used `/opt` Python/torch environments; no `/mnt` torch runtime for the formal finalizers.

### Results

- `H100-1 AUTO actual-LoRA`: actual LoRA weights and no-privilege inference were confirmed, but real closed-loop failed the gate. Paper-grade result: `student_beats_base=false`, `unshuffled_beats_shuffle=false`, `real_closed_loop_pass=false`.
- `H100-2 importance_tagging`: proper same-`xi_t` K4/K8 fork/replay completed with 2048 rows, but the formal gate failed. Paper-grade result: `proper_K4_positive=false`, `K8_direction_consistent_positive=false`, `gate_passed=false`.
- `H100-3 Structured`: current evidence remains route-head parity only. The safe claim is `Structured ~= Matched Text > Base` at route-level diagnostic granularity; there is no supported `Structured > Matched Text` claim from the available artifacts.
- `H100-4 baselines`: Full Harness same-contract reference is available only as a reference row; Matched Text and OPHSD remain route-level only and are not valid actual-LoRA main-table wins under the current contract.

### Conclusion

- The 0816-2 round does not produce a positive actual-model win for AUTO or importance_tagging.
- `importance_tagging` is not a valid second positive component under the formal fork gate and should not be launched into LoRA OPD from this artifact set.
- Structured privilege remains parity-like, not a superiority result.
- The remaining useful output is contract clarification and negative evidence: route-proxy gains, actual-LoRA gates, and baseline/reconciliation artifacts, not a new positive main-table claim.

### Completed / verified

#### Runtime

Allowed `/opt` runtime is available:

```text
/opt/scape-hf-scorer/bin/python
Python 3.12.13
torch 2.13.0+cu130
transformers 5.15.0
peft 0.20.0
pyserini: not installed
```

This satisfies the no-`/mnt` environment requirement for GPU/HF/PEFT workloads. `scripts/run_h100_2_live_fork_replay.py` has a deterministic local-corpus fallback when Lucene/pyserini is unavailable.

#### H100-1 AUTO actual-LoRA real closed-loop

Canonical controlling output directories:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_1_auto_lora_handoff_diagnostics_20260817/
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/h1001_actual_lora_sources/
```

Verified setting:

```text
actual_model_weights: true
student_inference_privilege: false
runtime: real executed BrowseComp+ BM25/chroma compatibility environment
candidate recipe: auto_populate_first_search relevant reverse Route-KL LoRA
paper-grade closed-loop scale: 256 queries
```

Paper-grade actual-model result:

```text
base_student                         reward=0.367756 trajectory_recall=0.152731 final_answer_recall=0.137915
AUTO relevant reverse-KL seed44       reward=0.082421 trajectory_recall=0.104482 final_answer_recall=0.090532
legacy shuffled control seed42        reward=0.362691 trajectory_recall=0.144899 final_answer_recall=0.139478
first-turn-only control seed42        reward=0.387497 trajectory_recall=0.146082 final_answer_recall=0.143248
invalid_tool_rate: 0.0 for all rows
```

Decision:

```text
DISCARD_RECIPE_FOR_MAIN_TABLE
student_beats_base: false
unshuffled_beats_shuffle: false
recommended_for_main_table: false
```

Interpretation for other servers/agents:

- The current AUTO OPD recipe has actual LoRA weights and no-privilege inference, but fails the real closed-loop gate.
- Do not promote same-state route-proxy results or the current AUTO actual-LoRA recipe to the main table as a positive result.
- A query-variant/repeat-action redesign smoke was attempted and still produced identical rewards; current instruction is to move to a substantively different OPD target/component rather than expand this recipe.
- H100-1 diagnostics SHA256 verification passed.

#### H100-2 importance_tagging proper K4/K8

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_formal_0817/
```

Formal contract and scale:

```text
same xi_t; importance ON first branch vs OFF reduced branch
both continuations use reduced policy
no full-harness takeover
2 seeds x K4/K8 x 512 states = 2048 rows
```

Gate result:

```text
seed8423 K4 mean_T_minus_S=-0.010283 ci95=[-0.013769, -0.006797]
seed8423 K8 mean_T_minus_S=-0.016494 ci95=[-0.021989, -0.010999]
seed8424 K4 mean_T_minus_S=-0.008408 ci95=[-0.011749, -0.005067]
seed8424 K8 mean_T_minus_S=-0.013740 ci95=[-0.018943, -0.008537]
proper_k4_positive: false
k8_direction_consistent_positive: false
gate_passed: false
```

Decision:

```text
do_not_start_importance_lora_opd
importance actual-LoRA real closed-loop: not_started_gate_blocked
recommended_for_main_table: false
```

The earlier approximate positive importance signal is not sufficient for this round. The formal directory SHA256 verification passed.

#### H100-3 structured parity / route diagnostic

Status remains as previously recorded: `Structured ~= Matched Text > Base` on the real BM25 route-head diagnostic, but it is auxiliary route-level only and not actual LLM Student weights. Do not claim `Structured > Textual` from the current artifacts.

#### H100-4 end-to-end baselines and novelty guard

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Generated required gap-aware deliverables include:

```text
END2END_BASELINE_PROTOCOL.md
FULL_HARNESS_REAL_CLOSED_LOOP.csv
MATCHED_TEXT_LORA_TRAINING.csv
MATCHED_TEXT_REAL_CLOSED_LOOP.csv
OPHSD_LORA_TRAINING.csv
OPHSD_REAL_CLOSED_LOOP.csv
STANDARD_OPSD_STATUS.md
END2END_MAIN_TABLE.csv/md
END2END_PAIRED_BOOTSTRAP.csv
END2END_COMPUTE_COST.csv
NOVELTY_MATRIX_20260816_LATE.md
NOVELTY_RED_LINES_LATE.md
BASELINE_GAP_LATE.md
H1004_END2END_BASELINE_HANDOFF.json
```

Main readout after corrected required-gap audit:

```text
Full Harness / Base full-modules Harness-1: completed same-contract vLLM test256, reward=0.367756
Ours AUTO Component OPD: completed actual LoRA real closed-loop, failed gate, reward=0.082421
Shuffle control: completed actual LoRA real closed-loop, reward=0.362691
First-turn-only control: completed actual LoRA real closed-loop, reward=0.387497
Matched Text OPD: actual-LoRA not run, training contract missing
OPHSD-style: route-level only, actual-LoRA training contract missing
```

Correction:

- `/mnt/songzijun/CLAUDE.md` was read and the contract-level closure lesson was applied.
- The prior `Base Student` row from H100-1 is not a reduced Base row: its manifest uses `modules_full_v2.yaml` with evidence_state, verification, and context_budget enabled, `policy_backend=vllm`, `model_path=/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`, `n=256`, and reward `0.367756`. It is now recorded as the local BM25 Full Harness / full-modules Harness-1 reference.
- A separate reduced no-privilege Base Student same-contract row is not available from these artifacts.
- Ours vs Full Harness: AUTO actual LoRA does not beat the full-modules reference.
- Ours vs Shuffle: AUTO actual LoRA does not beat shuffled control.
- Ours vs Matched Text / OPHSD: not comparable as actual LoRA; available baselines are blocked or route-level only.

H100-4 corrected artifacts were regenerated by `scripts/finalize_h1004_required_gaps_0817.py`; `SHA256SUMS` was refreshed.

### Current not-started / gap-blocked work

- Do not launch formal TEST/shuffle for the failed AUTO recipe as a positive main-table run.
- Do not launch importance LoRA/real closed-loop from the failed proper K4/K8 gate.
- Reduced Base Student same-contract row is not separately available because the prior Base label used full modules.
- Matched Text actual-LoRA remains blocked: `matched_v2_pairs.jsonl` has textualized state-time fields and reduced prompts, but not the teacher/ref route distribution plus optimizer/update-budget contract required by an actual HF/PEFT LoRA trainer.
- OPHSD actual-LoRA remains blocked: existing OPHSD artifacts are route-level `route_head.pt` cells, not reduced-prompt plus whole-harness-teacher actual LoRA data.
- H100-3 actual-LoRA structured V1/V2 should only continue if a valid actual-LoRA training/evaluation contract is implemented; two failed substantive redesigns should end the structured-superiority branch.

## 2026-08-16 H100-3 subtractive_curation all-zero root-cause audit

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; formal retry/training was intentionally not launched because the audit blocks it.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_3_subtractive_audit_0816_final/
```

Task scope:

- Follow-up to the 2026-08-15 `subtractive_curation` closed-loop gate where Base, Full/Student-style metrics, and best Student all reported `curated_evidence_recall=0`, `overall_reward=0`, and `final_answer_recall=0`.
- Goal was to determine whether the all-zero result is an evaluator/data/runtime-contract error or a real component failure.
- This run is an offline contract/root-cause audit, not a new OPD training run.
- Inputs were the current same-state subtractive test artifact and influence rows:
  - `outputs/true_scape_candidate_b_tournament/data/subtractive_curation_TEST_512.jsonl`
  - `outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl`
- Audited first `256` rows for oracle sanity and event coverage; case-level replay summary sampled `100` states.
- Heavy GPU training was not used because the spec only permits retraining after a verified evaluator/data repair or recollection of curate-event-positive data.

Oracle/evaluator sanity result:

```text
decision: EVALUATOR_NOT_CONSTANT_ZERO_BUT_REAL_TEST_LACKS_TERMINAL_GOLD_CONTRACT
rows audited: 256
synthetic base curated_evidence_recall:   0.000000
synthetic oracle curated_evidence_recall: 1.000000
real rows with documents:                 256/256
real rows with gold/reference contract:   0/256
student/parsed curate action rate:        0.113281
```

Event/argument coverage result:

```text
documents_nonempty:          256 / 256 = 1.000000
curated_ids_nonempty:        256 / 256 = 1.000000
teacher_curate_action:        29 / 256 = 0.113281
valid_add_ids:                29 / 256 = 0.113281
valid_remove_ids:              0 / 256 = 0.000000
event_active:                256 / 256 = 1.000000
value_positive:                0 / 256 = 0.000000
terminal_reward_available:     0 / 256 = 0.000000
```

Case-analysis result:

```text
cases analysed: 100
missing_terminal_gold_contract: 100
non_curate_route:                84
```

Code/audit finding:

- The scorer is not intrinsically constant-zero: the synthetic oracle curate action changes recall from `0.0` to `1.0`.
- The current real same-state test rows expose documents and curated ids, but do not expose terminal gold/reference fields required for terminal reward or final-answer recall.
- Current rows contain some valid `add_ids`, but no valid `remove_ids`; pointer/add-remove training on this artifact is not defensible.
- The named 0816 spec scripts `run_btp_subtractive.py`, `prepare_btp_subtractive_training.py`, `eval_btp_subtractive_closed_loop.py`, `train_route_opd.py`, and `scape/training/route_opd.py` are not present in this main checkout; this main-checkout audit uses `scripts/run_h100_3_subtractive_audit.py` and synthetic contract tests.
- The audit script checksum generation order was fixed so final `RUN_MANIFEST.json` and `STATUS_LIVE.md` are written before `SHA256SUMS` is generated.

Conclusion:

```text
REDESIGN_DATA_EVALUATOR_CONTRACT_BEFORE_RETRY
allowed_formal_retry: false
```

Interpretation for other servers/agents:

- Do **not** treat the previous all-zero closed-loop as proven component failure. The more precise diagnosis is missing evaluator/data contract for terminal scoring on the current same-state artifact.
- Do **not** launch the allowed one-time `value_weighted_route_kl` seeds `42,43,44,45` retry yet. The 0816 spec only permits this after an evaluator/data bug is fixed or after new true curate-event-positive data is collected.
- Do **not** train pointer/add-remove objectives from the current artifact. Valid pointer supervision is insufficient: `valid_remove_ids=0`, and terminal gold/reference is absent.
- If revisiting subtractive curation, first repair/recollect data with all of: curation event active, nonempty documents, meaningful teacher curate or route/gate decision, valid add/remove ids if pointer training is intended, and gold/reference fields available for terminal closed-loop metrics.
- A route/gate-only redesign may be defensible only after a repaired closed-loop evaluator can score nonzero terminal outcomes.
- Downstream writeups should update the 2026-08-15 `STOP subtractive_curation` wording: broad/unrepaired subtractive remains stopped, but the all-zero closed-loop itself is not a valid final failure signal until the evaluator/data contract is repaired.

Verification performed:

```text
pytest tests/test_subtractive_evaluator_nonzero_oracle.py \
       tests/test_subtractive_state_restore.py \
       tests/test_subtractive_curate_action_reachable.py \
       tests/test_subtractive_argument_id_contract.py -q
# 4 passed

cd outputs/btp_h100_3_subtractive_audit_0816_final && sha256sum -c SHA256SUMS
# all OK
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
SUBTRACTIVE_ORACLE_SANITY.md
SUBTRACTIVE_ORACLE_SANITY.json
SUBTRACTIVE_EVENT_COVERAGE.csv
SUBTRACTIVE_ARGUMENT_ROOT_CAUSE.md
SUBTRACTIVE_CODE_AUDIT.md
SUBTRACTIVE_ZERO_CASE_ANALYSIS.md
SUBTRACTIVE_ZERO_CASES.jsonl
REPAIRED_CLOSED_LOOP_RESULTS.csv
REPAIRED_CLOSED_LOOP_RESULTS.md
SUBTRACTIVE_REDESIGN_MANIFEST.json
H1003_SUBTRACTIVE_AUDIT_HANDOFF.json
SHA256SUMS
```

Synthetic tests present in this checkout:

```text
tests/test_subtractive_evaluator_nonzero_oracle.py
tests/test_subtractive_state_restore.py
tests/test_subtractive_curate_action_reachable.py
tests/test_subtractive_argument_id_contract.py
```

Operational notes:

- Outputs are under `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/`, following the global result/log location convention.
- No SCAPE/subtractive audit workers were left running after completion.
- GPU-heavy workloads were not started because the audit decision explicitly blocks formal retraining.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-16 H100-2 Beyond Textual Privilege auto_populate_first_search distillation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/btp_h100_2_auto_populate/
```

Task scope:

- Component: `V8D_AUTO_POPULATE_FIRST_SEARCH` / `auto_populate_first_search`.
- Goal was to test the second Beyond Textual Privilege component: whether high-level first-search initialization/search-control privilege can be distilled into a no-privilege Student using released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- This run intentionally distills route/tool-name behavior only. It does not distill query text, document ids, JSON arguments, or search-result text.
- All outputs remain local BM25/HF compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`. Do not describe these as official Harness-1 Chroma parity results.
- GPU-heavy Python used `/opt/vllm-qwen3-1.7b-harness/bin/python`; no `/mnt` JuiceFS torch environment was used for training/eval.
- H100-2 had 8 idle H100-class GPUs and used them for the first 8-cell training matrix; no related experiment processes remained after completion and all 8 GPUs were idle.

Value-confirm setting:

- Experiment: `AUTO_VALUE_CONFIRM512x2`.
- Strata: `NATURAL_FIRST_SEARCH`, `AUTO_EFFECT_ACTIVE`.
- Seeds: `2230`, `2231`.
- Horizons: `K4`, `K8`.
- Each `(stratum, seed)` K4/K8 pair used the same frozen 512-state manifest.
- Fork contract: same `xi_t`; Student executes reduced/student action; Teacher executes full-view action; both continue under reduced continuation policy; no full-harness takeover.
- `AUTO_EFFECT_ACTIVE` was determined from real runtime state/component code: full `auto_seed` present, reduced `auto_seed` absent, before first search, and no keyword/manual filtering.

Value-confirm result:

```text
decision: VALUE_POSITIVE
value rows: 4096 = 2 strata x 2 seeds x 2 horizons x 512 states
NATURAL_FIRST_SEARCH seed2230 K4 mean=0.009990 ci_low=0.008086 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2230 K8 mean=0.010430 ci_low=0.007646 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2231 K4 mean=0.009990 ci_low=0.008057 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2231 K8 mean=0.010430 ci_low=0.007617 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2230 K4 mean=0.009990 ci_low=0.008086 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2230 K8 mean=0.010430 ci_low=0.007852 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2231 K4 mean=0.009990 ci_low=0.008145 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2231 K8 mean=0.010430 ci_low=0.007734 noise_q95=0 gate=true
```

Data and privilege schema:

- `AUTO_PRIVILEGE_SCHEMA.md` records the structured runtime signal: `auto_seed` from full working memory before first search, reduced/full component mask visibility, `step`, and `tool_history` for first-search localization.
- Student inference has no auto privilege; `auto_populate_first_search=false` in the reduced mask.
- Dataset source: Student under `H_-auto_populate_first_search`, restricted to first-search / first-evidence-population relevant decision windows.
- Unique K4 states recovered for training/eval construction: `1180` across `370` query ids.
- Query-disjoint split: `train_unique=860`, `valid_unique=164`, `test_unique=156`.
- Positive train unique states: `532`; first-turn train unique states: `456`.
- Resampled update-budget files were written:
  - `AUTO_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_VALUE_POSITIVE_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_FIRST_TURN_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_VALID_1K.jsonl`: 1000 rows.
  - `AUTO_TEST_1K.jsonl`: 1000 rows.
- `AUTO_TRAIN_8K` and related 8K files are update budgets, not 8K unique states. Resampling is explicit in-row via `opd_row_id`, `snapshot_hash`, and `resampled_duplicate`.
- `lambda_args=0.0` throughout; categorical 8-way route targets only.

Training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- LR: `1e-5`.
- Update budget: `8000` train rows per cell, `1000` validation rows, `epochs=1`.
- Initial 8-cell matrix completed:
  - relevant `route_kl_forward`, seeds 42/43.
  - relevant `route_kl_reverse`, seeds 42/43.
  - relevant `action_ce`, seeds 42/43.
  - value-positive `route_kl_forward`, seeds 42/43.
- Full/remaining and event-localized cells completed:
  - relevant `route_kl_reverse`, seeds 44/45.
  - relevant `action_ce`, seeds 44/45.
  - `FIRST_TURN_ONLY_CE`, seeds 42/43.
- Total completed training cells: `14`. All wrote `result.json`, had finite loss/grad, changed trainable params, and had `invalid_tool=0.0`.

Same-state route-proxy closed-loop result:

- Evaluator: finalizer/proxy over query-disjoint `AUTO_TEST_1K`; this is not official external BrowseComp reward.
- Student inference condition: no auto privilege.

```text
Base Student:                         JS=0.109307 CE=1.230626 agreement=0.244 overall_reward=0.590777 invalid=0
first_turn_only_action_ce_seed42:      JS=0.117034 CE=1.134433 agreement=0.634 overall_reward=0.765977 invalid=0
first_turn_only_action_ce_seed43:      JS=0.070548 CE=0.786220 agreement=0.632 overall_reward=0.790053 invalid=0
relevant_action_ce_seed42:             JS=0.065676 CE=0.765296 agreement=0.700 overall_reward=0.822881 invalid=0
relevant_action_ce_seed43:             JS=0.069365 CE=0.783413 agreement=0.684 overall_reward=0.813937 invalid=0
relevant_action_ce_seed44:             JS=0.072330 CE=0.792340 agreement=0.674 overall_reward=0.808176 invalid=0
relevant_action_ce_seed45:             JS=0.066732 CE=0.765288 agreement=0.674 overall_reward=0.810812 invalid=0
relevant_route_kl_forward_seed42:      JS=0.045184 CE=0.668246 agreement=0.596 overall_reward=0.785680 invalid=0
relevant_route_kl_forward_seed43:      JS=0.044623 CE=0.666239 agreement=0.638 overall_reward=0.804826 invalid=0
relevant_route_kl_reverse_seed42:      JS=0.044919 CE=0.669647 agreement=0.698 overall_reward=0.831637 invalid=0
relevant_route_kl_reverse_seed43:      JS=0.043177 CE=0.663138 agreement=0.698 overall_reward=0.832410 invalid=0
relevant_route_kl_reverse_seed44:      JS=0.042467 CE=0.660550 agreement=0.734 overall_reward=0.848923 invalid=0
relevant_route_kl_reverse_seed45:      JS=0.044644 CE=0.666814 agreement=0.662 overall_reward=0.815604 invalid=0
value_positive_route_kl_forward_seed42 JS=0.069387 CE=0.768542 agreement=0.624 overall_reward=0.787301 invalid=0
value_positive_route_kl_forward_seed43 JS=0.062405 CE=0.739052 agreement=0.636 overall_reward=0.795882 invalid=0
```

Gate result:

```text
best_model:                 relevant_route_kl_reverse_seed44
base_overall_reward:         0.5907768084
best_overall_reward:         0.8489228879
base_trajectory_recall:      0.244
best_trajectory_recall:      0.734
base_final_answer_recall:    0.2385346381
best_final_answer_recall:    0.7318766626
invalid_tool_max:            0.0
main_gate_pass:              true
```

Conclusion:

```text
AUTO_POPULATE_FIRST_SEARCH_PASS_SAME_STATE_ROUTE_PROXY
```

Interpretation for other servers/agents:

- This is the strongest completed H100 BTP component result so far in this simplified record: value confirm passed, training finished, and same-state route-proxy gate passed with large Student-after > Student-base gains.
- The safe claim is route/control-policy improvement under local same-state proxy evaluation, not official BrowseComp Chroma parity and not yet a final external task-reward win.
- Unlike H100-1 verify selective, this run currently has no shuffled-target control recorded. If another agent needs paper-grade causal evidence, run a matched shuffle-target control before overclaiming state-conditioned privilege learning.
- `FIRST_TURN_ONLY_CE` is a meaningful baseline: it improves strongly but does not beat the best relevant reverse-KL cell. This supports event-localized control being useful, while best current recipe is still relevant reverse Route-KL.
- Recommended recipe for H20 clean SFT rerun: reuse this pipeline with `auto_populate_first_search`, `lambda_args=0`, relevant states, reverse Route-KL and action-CE top candidates, plus the first-turn-only baseline and a shuffle-target control.
- Do not mix argument distillation into this auto-populate run; previous argument-side auto-populate signal was known bad/negative, and this successful run explicitly avoids args.
- Downstream baseline comparison should use `Base Student`, best auto Student, and Full Harness reference, always marking student inference as no-privilege.
- `BEST_AUTO_STUDENT.json` currently has `checkpoint: null` due to a finalizer aggregation issue, but the best cell checkpoint exists at the path below.

Best route-proxy checkpoint:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/btp_h100_2_auto_populate/cells/relevant_route_kl_reverse_seed44/checkpoint
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
AUTO_VALUE_CONFIRM/
AUTO_VALUE_GATE.json
AUTO_PRIVILEGE_SCHEMA.md
DATA_AUDIT.md
AUTO_TRAINING_CELLS.csv
FIRST_TURN_ONLY_BASELINE.csv
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
BEST_AUTO_STUDENT.json
H1002_BTP_AUTO_HANDOFF.json
SHA256SUMS
```

Implementation / recovery notes:

- New/modified H100-2 scripts include `scripts/run_btp_h1002_auto_populate.py`, `scripts/build_btp_auto_data.py`, `scripts/train_route_opd.py`, `scripts/eval_route_opd.py`, `scripts/finalize_btp_auto_training.py`, `scripts/launch_btp_auto_8gpu.sh`, `scripts/launch_btp_auto_full_remaining.sh`, and `scape/training/route_opd.py`.
- Value confirm reuses the existing H100-2 live fork/replay machinery from `scripts/run_h1002_verify_value_confirm.py` but writes isolated BTP outputs under `outputs/btp_h100_2_auto_populate/`.
- `AUTO_EFFECT_ACTIVE` uses real renderer/mask state: full `auto_seed` available, reduced `auto_seed` absent, before first search. It is not a keyword filter.
- Finalizer `RUN_MANIFEST.json` records `status=completed`, `exit_code=0`, and completed shards `AUTO_VALUE_CONFIRM`, `gate`, `schema`, `handoff`.
- After completion, no `run_btp_h1002_auto_populate.py`, `train_route_opd.py`, or `launch_btp_auto` processes were running; `nvidia-smi` showed all 8 GPUs idle.

## 2026-08-16 H100-2 Structured Privilege vs Matched Text formal matrix and real closed-loop

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/
```

Real closed-loop output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/
```

Second-component diagnostic output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816/
```

Task scope:

- Followed `/mnt/songzijun/Capability_Evolution/SCAPE/todo/0816-1/README.md` and `H100-2_structured_privilege.md`.
- Component P0: `auto_populate_first_search`.
- Component P1 / second structured-native check: `importance_tagging`.
- Goal was the matched-information comparison: whether harness-native structured privilege beats or matches deterministic textualization under the same semantic fields, same on-policy states, same split, same objective, and no-privilege Student inference.
- This section supersedes any interpretation that only same-state route proxy was completed. The primary H100-2 result below is the real BM25 closed-loop execution in `outputs/h100_2_real_closed_loop_bm25_0816/`.
- GPU-heavy Python used `/opt/vllm-qwen3-1.7b-harness/bin/python` with torch/CUDA; default Python was used only for pytest because the `/opt` torch env does not include pytest.

Structured/textual protocol:

- Frozen state source for AUTO training/proxy matrix:
  `outputs/h100_3_real_influence_shards/auto_populate_first_search/REAL_INFLUENCE_PER_STATE.jsonl`.
- Split is query-disjoint and frozen in:
  `TRAIN_SPLIT_MANIFEST.json`, `VALID_SPLIT_MANIFEST.json`, `TEST_SPLIT_MANIFEST.json`.
- Split sizes:

```text
train: 608 states
valid: 208 states
test:  208 states
```

- Student route-head inference features explicitly exclude privileged fields. They contain only reduced/no-privilege state identifiers/counts: query id hash, step, tool-history length, document count, prior-search count, and state hash feature.
- Structured/textual variants differ only in teacher/control target construction from matched fields; deployed route-head inference does not receive `auto_seed_present`, full mask fields, teacher tool, textual privilege, gold labels, rewards, or future observations.
- Textualizer uses deterministic key/value JSON rendering and round-trips back to the structured record. `AUTO_INFORMATION_EQUIVALENCE_AUDIT.md` records `1024/1024` pass.
- `student_inference_has_privilege=false` in all handoffs and cell summaries.

AUTO formal route-head training matrix:

```text
AUTO_STRUCT_DIRECT seeds 42,43,44,45
AUTO_STRUCT_TYPED seeds 42,43,44,45
AUTO_MATCHED_TEXT seeds 42,43,44,45
AUTO_JSON_TEXT_DIAGNOSTIC seeds 42,43
AUTO_STRUCT_TYPED_DEBOTTLENECK seeds 42,43,44,45
AUTO_STRUCT_EVENT_TUPLE seeds 42,43,44,45
```

- Total AUTO trained cells: `22`.
- Every cell wrote `cells/<variant>_seed<seed>/route_head.pt`, `summary.json`, `STATUS_LIVE.md`, and `DONE`.
- All route-head checkpoints were reloadable; losses/gradients were finite; invalid tool rate was `0.0`.
- First batch used 8-way parallelism across the 8 visible GPUs; redesign cells also used 8-way parallelism.
- Same-state route-head proxy diagnostics are preserved as `AUTO_REPRESENTATION_ROUTE_PROXY.csv`, but are no longer the primary result.

AUTO real closed-loop setting:

- Runner: `scripts/run_h100_2_route_head_closed_loop.py`.
- Output: `outputs/h100_2_real_closed_loop_bm25_0816/`.
- Environment: real BM25 BrowseComp-Plus interaction using `/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25`, `queries.tsv`, and `qrel_evidence.txt`.
- The evaluator loads trained route-head checkpoints, lets the Student choose tools over multiple steps, mutates the BM25 state, and computes final reward from executed state. This is not a same-state route proxy.
- Methods compared:
  - `BASE_REDUCED`
  - `AUTO_MATCHED_TEXT`
  - best structured variant selected from AUTO matrix: `AUTO_STRUCT_TYPED`
- Query count: `128`.
- Max steps: `6`.
- Student inference privilege: `false`.

AUTO real closed-loop result:

```text
best_structured_variant: AUTO_STRUCT_TYPED
Structured - Textual overall_reward delta: 0.0
Structured - Base overall_reward delta:      +0.03
Textual - Base overall_reward delta:         +0.03
paired bootstrap CI for Structured - Textual: [0.0, 0.0]
```

Per-method means:

```text
AUTO_MATCHED_TEXT:
  n=128
  overall_reward=-0.015
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=1.0

AUTO_STRUCT_TYPED:
  n=128
  overall_reward=-0.015
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=1.0

BASE_REDUCED:
  n=128
  overall_reward=-0.045
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=3.0
```

Primary conclusion for H100-2:

```text
Structured ~= Textual > Base
```

Interpretation:

- This satisfies the H100-2 minimum meaningful outcome on the primary real closed-loop evaluator: structured matches textual and both beat base under no-privilege inference.
- It does **not** support a `Structured > Textual` claim.
- Do not cite the route-head proxy negative/positive diagnostics as the primary H100-2 result. The primary result is `real_closed_loop.status=completed_real_closed_loop_bm25` in `H1002_STRUCTURED_PRIVILEGE_HANDOFF.json`.
- The real closed-loop reward remains low/negative in absolute value and has zero final-answer recall. It is useful for relative structured/text/base comparison, but not yet paper-grade external task success.
- Other servers should treat the current method as a parity result, not a structured-advantage result and not a final paper-grade BrowseComp result.

Second component / importance_tagging diagnostic:

- Output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816/`.
- Frozen state source: `outputs/h100_3_real_influence_shards/importance_tagging/REAL_INFLUENCE_PER_STATE.jsonl`.
- Split sizes: `train=608`, `valid=208`, `test=208`.
- Cells:

```text
IMPORTANCE_STRUCT_TYPED seeds 42,43,44,45
IMPORTANCE_MATCHED_TEXT seeds 42,43,44,45
IMPORTANCE_STRUCT_ORDERED_TAGS seeds 42,43,44,45
```

- Total importance cells: `12`, all with reloadable `route_head.pt`.
- This was a formal route-head diagnostic for the second value-positive component; it was not used as the primary real closed-loop comparison.

Importance result:

```text
best_structured_variant: IMPORTANCE_STRUCT_TYPED
Structured - Textual reward-proxy delta: -0.004795988090336323
student_inference_has_privilege=false
```

Interpretation for importance:

- importance_tagging does not currently show structured advantage over matched text.
- Because AUTO real closed-loop is parity rather than failure, do not discard the whole structured-privilege direction solely from the importance diagnostic.
- If another agent continues H100-2, next work should redesign the structured interface or run a stronger component-specific importance real closed-loop evaluator rather than reusing the current structured adapter unchanged.

Final decision written in the handoff:

```text
status: completed_real_closed_loop_parity
method_decision: keep_as_parity_not_structured_advantage
best_structured_variant: AUTO_STRUCT_TYPED
structured_vs_textual_delta: 0.0
student_inference_has_privilege: false
```

Main handoff and decision files:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_STRUCTURED_PRIVILEGE_HANDOFF.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_FINAL_DECISION_AFTER_REDESIGN.md
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_FINAL_DECISION_AFTER_REDESIGN.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/REAL_CLOSED_LOOP_HANDOFF.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json
```

Required artifacts present in the canonical directory:

```text
STRUCTURED_COMPONENT_INVENTORY.md
MATCHED_INFORMATION_PROTOCOL.md
AUTO_INFORMATION_EQUIVALENCE_AUDIT.md
STRUCTURED_INTERFACE_V1.md
STRUCTURED_INTERFACE_V2.md
STRUCTURED_REP_DEBUG.md
AUTO_REPRESENTATION_CELLS.csv
AUTO_REPRESENTATION_CLOSED_LOOP.csv
AUTO_STRUCTURED_VS_TEXTUAL.md
AUTO_REPRESENTATION_BOOTSTRAP.csv
IMPORTANCE_VALUE_CONFIRM/
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
BEST_STRUCTURED_STUDENT.json
H1002_STRUCTURED_PRIVILEGE_HANDOFF.json
REAL_CLOSED_LOOP_HANDOFF.json
H1002_FINAL_DECISION_AFTER_REDESIGN.md
H1002_FINAL_DECISION_AFTER_REDESIGN.json
SHA256SUMS
```

Implementation notes for other servers/agents:

- New scripts in the main checkout:

```text
scripts/run_h100_2_auto_formal_route.py
scripts/run_h100_2_importance_formal_route.py
scripts/run_h100_2_route_head_closed_loop.py
scripts/run_h100_2_structured_privilege_matrix.py
```

- `scripts/run_h100_2_structured_privilege_matrix.py` was an earlier offline evidence/proxy matrix and should not be treated as the final main result.
- `scripts/run_h100_2_auto_formal_route.py` trains the route-head cells and keeps Student inference no-privilege.
- `scripts/run_h100_2_route_head_closed_loop.py` is the real BM25 interactive closed-loop evaluator used for the main comparison.
- `AUTO_REPRESENTATION_ROUTE_PROXY.csv` preserves the old proxy aggregate after the required `AUTO_REPRESENTATION_CLOSED_LOOP.csv` was replaced with real closed-loop results.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted. Git status for the relevant new scripts is untracked unless another agent stages/commits them.

Verification performed:

```text
cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816
sha256sum -c SHA256SUMS
# all OK

cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816
sha256sum -c SHA256SUMS
# all OK

cd /mnt/songzijun/Capability_Evolution/SCAPE
python -m pytest tests/test_component_mask.py tests/test_dual_view.py tests/test_metric_pairing.py tests/test_rollout_influence_contract.py -q
# 6 passed
```

Operational status:

- No H100-2 training/closed-loop processes were left running.
- Final `nvidia-smi` showed all 8 GPUs idle with about `1 MiB` used per GPU.
- `/opt/vllm-qwen3-1.7b-harness/bin/python` has torch/CUDA and was used for training/closed-loop; it does not have pytest, so pytest verification used the default Python environment.

## 2026-08-15 H100-1 Beyond Textual Privilege verify_tool selective/value-conditioned OPD

Status: selective/value-conditioned sweep completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE`; full expansion and real closed-loop gate were not run.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/
```

Task scope:

- Verify Tool / `V8D_VERIFY_TOOL` mainline for Beyond Textual Privilege.
- Goal was to test whether value-conditioned selective OPD can produce the first truly positive Student distillation result from released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- All outputs remain local compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`.
- GPU-heavy Python used `/opt/bishop-harness/bin/python3.11`; no `/mnt` JuiceFS torch environment was used for training/eval.
- 8 visible H100-class GPUs were used for the main 8-cell 2K-equivalent sweep; the shuffle-target control ran as a single-GPU matched-budget control.

Execution status for this todo:

- Completed: privilege schema audit, value mining data build, Stage A/B selective mining, 2K-equivalent 8-cell training sweep, shuffled-target control, handoff material generation.
- Not performed: full 8K/update-budget expansion, real closed-loop `BTP_VERIFY_DEV128` / `BTP_VERIFY_TEST256` benchmark gate, any claim that the run produced a final verified Student win over Base.
- Partial / pending in the broader H100 baseline factory: matched-text OPD was only prepared, while OPSD / OPCD / SEED stayed blocked by missing faithful contracts.
- All downstream agents should treat the route-proxy gains as provisional only and keep the shuffle-target control in mind.

Current todo coverage:

- `H100-1` verify selective OPD: selective/value-conditioned sweep completed; full expansion and real closed-loop gate did **not** run.
- `H100-3` subtractive_curation event-conditioned selective distillation: completed separately and stopped after closed-loop gate failure.
- `H100-4` baseline factory: only partial preparation is done in the broader H100 baseline line; matched-text OPD is prepared, but OPSD / OPCD / SEED remain blocked.
- Do not infer any cross-task final winner from this file alone; read the per-experiment sections below for the exact gating status.

Data and privilege schema:

- Source value rows: H100-2 live fork/replay output `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/h100_2_verify_value_confirm/VERIFY_VALUE_PER_STATE.jsonl`.
- Snapshot backjoin source: H100-2 frozen `manifests/` and `manifest_shards/` under the same output directory.
- Builder script: `scripts/build_btp_verify_selective_data.py` in the H100-1 worktree.
- Recovered strict same-state rows with snapshots: `1164` unique K4 states across `367` query ids.
- `86` value rows could not be backjoined to a frozen snapshot and were excluded; no fake snapshot or duplicate unique state was created.
- `SELECT_POSITIVE`: `609` rows across `263` query ids, rule `A_K4 > replay_noise_q95` with `replay_noise_q95=0.0`.
- `SELECT_STRICT`: also `609` rows; all positives had action disagreement and positive teacher margin.
- 2K-equivalent training used the `609` positive states with explicit resampling to match optimizer update budget. Resampling is marked in-row and must not be reported as 2K unique states.
- `VERIFY_PRIVILEGE_SCHEMA_V2.md` includes only state-time fields: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, and remaining budget. Future labels such as `A_K4`, terminal reward, replay noise, and gold answer recall are explicitly excluded from privilege inputs.

Training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- Learning rate: smoke at `1e-5`, then all formal cells uniformly lowered to `5e-6` because reverse-KL smoke had large but finite gradients. Do not selectively tune LR per objective.
- Update budget: `2000` train rows per cell, `1000` validation rows.
- Formal 8-cell matrix:
  - `SELECT_POSITIVE + route_kl_forward`, seeds 42/43.
  - `SELECT_POSITIVE + route_kl_reverse`, seeds 42/43.
  - `SELECT_POSITIVE + action_ce`, seeds 42/43.
  - `advantage-weighted positive + route_kl_forward`, seeds 42/43.
- Value weight for weighted cells: `w_i=max(A_K4-noise_floor,0)`, batch/global mean normalized in data and clipped to fixed upper bound `4.0` in the loss path.

2K-equivalent route results:

```text
weighted_positive route_kl_forward: mean ΔJS=-0.400175, mean ΔCE=-2.597806, mean post agreement=0.6245, invalid_tool=0
SELECT_POSITIVE route_kl_forward:  mean ΔJS=-0.399374, mean ΔCE=-2.594443, mean post agreement=0.6105, invalid_tool=0
SELECT_POSITIVE route_kl_reverse:  mean ΔJS=-0.390069, mean ΔCE=-2.543890, mean post agreement=0.6410, invalid_tool=0
SELECT_POSITIVE action_ce:         mean ΔJS=-0.277253, mean ΔCE=-1.589558, mean post agreement=0.6400, invalid_tool=0
```

Important stability note:

- All formal cells were finite and wrote checkpoints.
- `positive_route_kl_reverse_seed43` had `max_grad_norm=19561.04`; treat reverse-KL as less stable even though it finished.
- Weighted forward Route-KL had lower gradients (`600-696`) and best average route proxy among non-control cells.

TEST256 same-state proxy evaluation:

- Evaluator: `scripts/eval_route_opd.py` after fixing adapter loading so LoRA checkpoints are loaded via `PeftModel.from_pretrained(base, adapter)` and not double-wrapped.
- Split: `BTP_VERIFY_TEST256_same_state_proxy`, first 256 rows from query-disjoint `VT_TEST_1K.jsonl`.
- This is **not** external BrowseComp task reward. CER / Trajectory Recall / Final Answer Recall are not evaluated in this path.

```text
base_student                         JS=0.056527  CE=0.875336  agreement=0.000000
weighted_route_kl_forward_seed43      JS=0.029468  CE=0.748518  agreement=0.660156
positive_route_kl_forward_seed42      JS=0.031821  CE=0.759385  agreement=0.699219
weighted_route_kl_forward_seed42      JS=0.033711  CE=0.767588  agreement=0.695313
positive_route_kl_forward_seed43      JS=0.034831  CE=0.772555  agreement=0.605469
positive_route_kl_reverse_seed42      JS=0.036729  CE=0.786680  agreement=0.601563
positive_route_kl_reverse_seed43      JS=0.042469  CE=0.816023  agreement=0.679688
positive_action_ce_seed42             JS=0.151659  CE=1.720422  agreement=0.675781
positive_action_ce_seed43             JS=0.153668  CE=1.750814  agreement=0.675781
```

Critical shuffle-target control:

- Control data: same `VT_SELECTIVE_TRAIN_2K` source states and same 2000 update budget.
- Teacher targets were shuffled across states with seed `8151042`.
- Teacher action marginal distribution was preserved exactly: `end_search=1919`, `read_document=81`.
- State-conditioned pairing was destroyed; there were zero fixed points in the permutation.
- Control objective: seed 42 forward 8-way Route-KL.

```text
unshuffled positive_route_kl_forward_seed42 TEST256 proxy: JS=0.031821, CE=0.759385, agreement=0.699219
shuffled_route_kl_forward_seed42 TEST256 proxy:            JS=0.023599, CE=0.722810, agreement=0.648438
```

Conclusion:

```text
NO_FULL_EXPANSION
```

Interpretation for other servers/agents:

- Do **not** cite this H100-1 run as the first positive Student distillation result.
- Route-level proxy improves strongly, but the shuffled-target control matches or exceeds the unshuffled seed42 proxy. Current evidence is compatible with marginal teacher-target imitation, not isolated state-conditioned privilege learning.
- The selected positive training rows are heavily dominated by `end_search`/`read_document`; there is no verify argmax in the selected teacher-target marginal. Therefore the run does not demonstrate verify-specific behavior improvement.
- Do not launch full 8K/update-budget expansion from this result. `FULL_TRAINING.csv` intentionally records `not_started`.
- Do not report `Student_after > Student_base` for real task reward; only same-state route proxy improved. Real `BTP_VERIFY_DEV128` / `BTP_VERIFY_TEST256` closed-loop reward, CER, Trajectory Recall, and Final Answer Recall were not evaluated.
- Do not hand this checkpoint to H100-4 as a final best Student for baseline comparison. `BEST_VERIFY_STUDENT.json` marks the best route-proxy checkpoint but also marks it as not recommended as final Student.
- H20 clean SFT rerun should **not** reuse this recipe as-is. First collect a larger on-policy verify pool with a verify-action-positive subset, repeat weighted/unweighted forward Route-KL, keep the shuffle-target control, and only then run the real closed-loop gate.

Best route-proxy checkpoint, not main-gate winner:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/cells/weighted_route_kl_forward_seed43/checkpoint
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
VERIFY_PRIVILEGE_SCHEMA_V2.md
DATA_AUDIT.md
VALUE_MINING_SUMMARY.md
VALUE_PER_STATE.jsonl
SELECT_POSITIVE.jsonl
SELECT_STRICT.jsonl
SELECTIVITY_STATS.csv
TRAINING_2K_EQUIV.csv
TRAINING_GATE.json
FULL_TRAINING.csv
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
SHUFFLED_TARGET_CONTROL.md
BEST_VERIFY_STUDENT.json
H1001_BTP_VERIFY_HANDOFF.json
SHA256SUMS
```

Implementation / recovery notes:

- New/modified H100-1 scripts: `scripts/build_btp_verify_selective_data.py`, `scripts/launch_btp_verify_selective_8gpu.sh`, `scripts/eval_route_opd.py`, `scripts/run_btp_verify_proxy_eval_8gpu.sh`, `scripts/train_route_opd.py`, `scape/training/route_opd.py`.
- Adapter evaluation bug fixed in H100-1 `scape/training/route_opd.py`: adapter checkpoints must be loaded from base model via PEFT once; double-wrapping adapters invalidates proxy evaluation.
- Advantage-weighted loss path was fixed so `normalized_value_weight` actually multiplies the main objective and is clipped to `4.0`.
- GPU cleanup verified after completion: all 8 GPUs idle.

## 2026-08-15 H100-3 Beyond Textual Privilege subtractive_curation event-conditioned selective distillation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/btp_h100_3_subtractive/
```

Task scope:

- Component: `V8D_SUBTRACTIVE_CURATION` / `subtractive_curation`.
- Goal was to test whether curation-critical event-conditioned selective distillation can produce the third transferable Beyond Textual Privilege component from released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- All outputs are local BM25/HF compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`. Do not describe these as official Harness-1 Chroma parity results.
- GPU-heavy Python used `/opt/scape-h1003-hf-scorer/bin/python`; no `/mnt` JuiceFS torch environment was used.
- 8 H100-class GPUs were used for raw collection, K4 value mining, K8 confirmation, and the first-stage training matrix.

Important implementation files added/modified in the H100-3 worktree:

```text
scripts/run_btp_subtractive.py
scripts/prepare_btp_subtractive_training.py
scripts/eval_btp_subtractive_closed_loop.py
scripts/train_route_opd.py
scape/training/route_opd.py
```

Data/event schema:

- Raw collection target was `SC_RAW_TRAIN_12K`; actual full collection produced `13,280` runtime states from `830` BrowseComp+ query ids, with `16` on-policy reduced-view turns per query.
- Event schema output: `CURATION_EVENT_SCHEMA.md`.
- Event fields are from real local Harness-compatible runtime state only: `snapshot.working_memory.documents`, `curated_ids`, `curated_importance`, `verified_unsupported`, `evidence_graph`, and `teacher_action.arguments.add_ids/remove_ids`.
- No LLM subjective curation labels were used.
- Id supervision is allowed only when `add_ids` exist in current `documents` and `remove_ids` exist in current `curated_ids`.
- Query-disjoint training/eval route splits were generated from `SC_VALUE_POSITIVE.jsonl`: `SC_ROUTE_TRAIN_2K.jsonl`, `SC_VALID_1K.jsonl`, `SC_TEST_1K.jsonl`.

Stage A / value mining:

```text
Raw states:              13,280
Stage A candidates:       1,855
K4 value rows:            1,855
K4 positive:                703
K4 zero:                    333
K4 negative:                819
K4 broad mean value:       -0.0022237197
K8 confirmation rows:        96
K8 negative-control repeat:  32
```

Interpretation of value mining:

- Broad/unfiltered subtractive curation is not supported: the K4 broad mean is slightly negative.
- There is a real selective positive subset (`703` states), so the useful hypothesis is event/value-conditioned, not broad OPD.
- Final paper main set should use `SC_VALUE_POSITIVE` only if a closed-loop gate passes. `SC_TERMINAL_POSITIVE` was written as secondary-only data and was not mixed into the main training set.

First-stage training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- LR: `1e-5`.
- Bounded first-stage ranking pass: `train_states=512`, `eval_states=128`, `epochs=1`.
- The full `train_states=2000`, `eval_states=1000` attempt was too slow with serial route scoring and was stopped; logs are preserved. It should not be reported as the completed training setting.
- Required 8-cell matrix completed after trainer microbatch optimization:
  - `route_kl_forward`, seeds 42/43.
  - `action_ce`, seeds 42/43.
  - `route_kl_arg_ce`, seeds 42/43.
  - `value_weighted_route_kl`, seeds 42/43.

Training result summary from `ROUTE_OPD_REPORT.json` / `TRAINING_CELLS.csv`:

```text
value_weighted_route_kl: mean ΔJS=-0.025046, mean ΔCE=-0.106236, finite=true, params_changed=true
route_kl_forward:        mean ΔJS=-0.017013, mean ΔCE=-0.066911, finite=true, params_changed=true
route_kl_arg_ce:         mean ΔJS=-0.017013, mean ΔCE=-0.066911, finite=true, params_changed=true
action_ce:               mean ΔJS=+0.052417, mean ΔCE=+0.462664, finite=true, params_changed=true
```

Argument-supervision audit:

- `SC_VALUE_POSITIVE` contained zero train rows with valid curate argument supervision under the strict current-state id audit.
- Therefore `route_kl_arg_ce` reduced to route KL on this split and explicitly reports argument CE as not applicable.
- No nonexistent document id was teacher-forced.
- Do not claim this run tested successful pointer/add-remove-id supervision.

Closed-loop gate:

- Evaluator: `scripts/eval_btp_subtractive_closed_loop.py`.
- Split: first `256` rows from query-disjoint `SC_TEST_1K.jsonl`.
- Models evaluated: base student, value-weighted Route-KL seeds 42/43, Route-KL forward seeds 42/43.
- Metrics are same-state proxy metrics derived from runtime state; they are not official external BrowseComp task reward.

Gate result:

```text
best_model: value_weighted_route_kl_seed42
base curated_evidence_recall: 0.0
best curated_evidence_recall: 0.0
base overall_reward:          0.0
best overall_reward:          0.0
base final_answer_recall:     0.0
best final_answer_recall:     0.0
best invalid_tool_rate:       0.0
closed_loop_gate_pass:        false
```

Conclusion:

```text
STOP subtractive_curation
```

Interpretation for other servers/agents:

- Do **not** launch Top-2 × 4 seeds full training for subtractive_curation from this run.
- Do **not** return to broad/unfiltered 8K subtractive curation OPD; the K4 broad mean is negative and the closed-loop gate failed.
- Do **not** cite this as `Student_after > Student_base`; closed-loop proxy showed no improvement over base.
- Do **not** cite this as evidence for working argument/pointer distillation; valid arg-supervision rows were zero under the strict id audit.
- Safe claim: event-conditioned mining found a selective positive subset, and value-weighted Route-KL improved route proxy metrics in the bounded first-stage pass, but the recipe failed the closed-loop gate and must stop.
- Next candidate priority from the todo remains `auto_populate_first_search > importance_tagging`, but only take over `auto_populate_first_search` if H100-2 is not already running it.
- H20 clean SFT rerun should not blindly reuse this recipe. If revisited, first collect states with actual valid `curate` add/remove id targets and a nonzero closed-loop recall signal.

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
CURATION_EVENT_SCHEMA.md
DATA_AUDIT.md
VALUE_PER_STATE.jsonl
SC_VALUE_POSITIVE.jsonl
SC_TERMINAL_POSITIVE.jsonl
SC_VALUE_NEGATIVE_CONTROL.jsonl
TRAINING_CELLS.csv
ARGUMENT_SUPERVISION_AUDIT.md
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
BEST_SUBTRACTIVE_STUDENT.json
H1003_BTP_SUBTRACTIVE_HANDOFF.json
SHA256SUMS
```

Operational/recovery notes:

- Raw collection initially OOMed with an 8-way batched scorer. `scripts/run_btp_subtractive.py` was patched to microbatch continuation scoring; the full 13,280-state collection then completed.
- K4 full value mining completed in six canonical shards covering all 1,855 candidates. Partial K4 shards from an early overlap run exist and should not be double-counted as canonical K4 rows.
- Training initially collided with active K8 memory on GPUs 1/7 and two seed-43 cells OOMed. Those exact PIDs were stopped/relaunched after K8 freed memory; broad `pkill` was rejected and not used.
- `scape/training/route_opd.py` was extended with `value_weighted_route_kl`, `route_kl_arg_ce`, and microbatched `route_logits`. A tiny smoke passed with finite loss/grad and `smoke_pass=true` before relaunching the 8-cell matrix.
- Closed-loop evaluation completed successfully and wrote `CLOSED_LOOP_GATE.json` with `pass=false`.
- Final cleanup verified no `run_btp_subtractive.py`, `train_route_opd.py`, or `eval_btp_subtractive` workers remained; all 8 GPUs were idle.

## 2026-08-14 H100-4 verify_tool Structured-vs-Textual privilege representation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/outputs/h100_4_privilege_representation/
```

Important source data:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_4_verify_confirm/verify_tool_hf_scorer/REAL_INFLUENCE_PER_STATE.jsonl
```

Environment/resource notes:

- Used `/opt/scape-hf-scorer` as the Python/CUDA environment.
- Do not rerun SCAPE GPU workloads from `/mnt` JuiceFS Python environments.
- This machine exposed 4 visible GPUs, not the 8 GPUs assumed by the todo. The 8-cell matrix was run in two GPU waves.
- Existing source data had only 2048 states total, so query-disjoint split could not be `2000/256/256` without duplication. Actual split is `1536/256/256` from 96/16/16 query ids.
- No samples were duplicated to fake the requested 2K train size.

Experiment setting:

- Component fixed to `verify_tool`.
- Structured privilege: structured non-natural-language JSON boolean `verify_available`, derived from Harness runtime `verify_tool` mask / `full_view.verify_available`.
- Textual privilege: deterministic natural-language template rendering exactly the same `verify_available` boolean.
- Information equivalence: pass. No LLM textualizer, no extra reasoning, no claim text, no candidate document ids, no future information.
- Student inference/runtime view: reduced no-privilege view only; no privileged field.
- Objectives run: Route-KL and Action CE.
- Seeds run: 42 and 43 for each structured/textual x objective cell.
- Smoke checks passed for finite loss, finite gradients, reloadable checkpoint, normalized route distribution, invalid tool rate 0.

Primary metrics:

```text
Structured Route-KL common-reference test JS: 0.016458936035633087
Textual Route-KL common-reference test JS:    0.01645179372280836
Route-KL seed wins: ['structured', 'textual']
structured_beats_base: false
structured_beats_text: false
textual_beats_struct: false
```

Conclusion:

```text
REPRESENTATION_PARITY_OR_UNSTABLE_GAP_NO_BASE_GAIN
```

Interpretation for other servers/agents:

- Do not cite this run as evidence that structured privilege beats textual privilege.
- Do not cite this run as evidence that textual privilege robustly beats structured privilege either; the Route-KL seed direction is inconsistent and the mean gap is tiny.
- The safe paper-level claim from this run is that this particular verify_tool representation comparison is inconclusive / parity-like under the available 2048-state source, with no stable base gain.
- 8K expansion was not triggered because the source only contains 2048 states and generating new data was outside this completed H100-4 run.
- Official Chroma parity was not run and remains `false` in handoff.

Key artifacts:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
STRUCTURED_PRIVILEGE_SCHEMA.md
REPRESENTATION_CONTRACT.md
INFORMATION_EQUIVALENCE_AUDIT.md
TEXTUALIZATION_TEMPLATE.md
DATA_AUDIT.md
TRAIN_SPLIT_MANIFEST.json
VALID_SPLIT_MANIFEST.json
TEST_SPLIT_MANIFEST.json
TEACHER_SIGNAL_DIAGNOSTIC.csv
TEACHER_SIGNAL_DIAGNOSTIC.md
NULL_REPRESENTATION_CONTROLS.md
REPRESENTATION_2K_CELLS.csv
REPRESENTATION_2K_REPORT.md
REPRESENTATION_GATE.json
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
STRUCTURED_VS_TEXTUAL.md
H1004_PRIVILEGE_REP_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
pytest tests/test_learnability_metrics_v3.py  # 7 passed
sha256sum -c SHA256SUMS                      # passed
```

## 2026-08-14 H100-3 verify_tool 8-way route OPD objective sweep

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/h100_3_route_opd_verify/
```

Experiment setting:

- Component fixed to `verify_tool`.
- Route space fixed to 8-way canonical tool distribution: `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- Source data: `outputs/h100_3_supervision_coherence/STATE_MANIFESTS/verify_tool_states.json`.
- Split: query-disjoint train/valid/test, deterministic seed `81403`.
- Train/eval budget: 2000 train states and 1000 valid states per cell.
- Objectives: `route_kl_forward`, `route_kl_reverse`, `tool_name_token_kl`, `action_ce`.
- Seeds: `42`, `43` for each objective.
- Shared hyperparameters: `epochs=1`, `lr=1e-5`, `lora_rank=8`, `lora_alpha=16`, `lambda_anchor=0.05`.
- Local compat only: `official_chroma_parity=false`, `local_compat_only=true`.

Validation:

- Metric contract passed on same distribution and perturbed distribution.
- Smoke checks existed for all 4 objectives, but only `route_kl_forward` smoke passed; the other three smoke runs were unstable under the chosen budget.
- Finalization produced `ROUTE_OPD_2K_CELLS.csv`, `ROUTE_OPD_2K_REPORT.md`, `ROUTE_OPD_GATE.json`, `H1003_ROUTE_OPD_HANDOFF.json`, `SHA256SUMS`.

Primary results:

- `route_kl_forward`: mean ΔJS = `0.017984`, smoke pass rate `0.00`.
- `route_kl_reverse`: mean ΔJS = `0.015376`, smoke pass rate `0.00`.
- `tool_name_token_kl`: mean ΔJS = `0.024542`, smoke pass rate `0.00`.
- `action_ce`: mean ΔJS = `0.029073`, smoke pass rate `0.00`.
- Best objective by post-eval JS was `route_kl_reverse`, but `route_js_improved=false` overall.

Conclusion:

```text
OBJECTIVE_MISMATCH_NOT_SUFFICIENT_EXPLANATION
```

Interpretation for other servers/agents:

- This 2K sweep does **not** support the claim that route-level OPD already fixes verify_tool routing.
- None of the four objectives reduced held-out route JS on average; all moved the distribution away from the teacher reference.
- Do not expand this run to 8K from this result alone.
- Do not write this up as evidence for route-KL superiority over action CE or token-name KL.
- Closed-loop Student evaluation was completed later as a 64-state same-state proxy evaluation; `best_objective=base_student`, `route_js_improved=false`, `closed_loop_student_improved=false`.
- The closed-loop proxy outputs are in `outputs/h100_3_route_opd_verify/CLOSED_LOOP_RESULTS.md`, `CLOSED_LOOP_GATE.json`, and `H1003_ROUTE_OPD_HANDOFF.json`.
- Because the source manifest does not include external answer gold, the closed-loop metrics are state-derived proxies, not an external gold benchmark.

## 2026-08-15 H100-4 Beyond Textual Privilege baseline factory

Status: partial completion. Worktree:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/
```

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/outputs/btp_h100_4_baselines/
```

Task scope:

- Compare Base, Full-Harness trajectory SFT, Standard OPD, Matched Text OPD, OPSD, OPCD, and SEED under the Beyond Textual Privilege protocol.
- Keep the same released `pat-jj/harness-1` checkpoint family, shared train/eval data contract, no inference-time privilege, and canonical V3 KL/JS metrics.
- This H100 machine exposed 4 GPUs, so parallel work was run in 4-GPU waves. GPU-heavy Python used `/opt/scape-hf-scorer/bin/python`; no `/mnt` JuiceFS torch environment was used.
- All results remain `LOCAL_COMPAT_ONLY=true` and `official_chroma_parity=false`.

Available source data and controls:

- Reused the frozen H100-3 route data at `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/h100_3_route_opd_verify/VT_ROUTE_TRAIN_8K.jsonl`.
- Source contains 8,000 route states with Student/Teacher route distributions, reduced/full prompts, and same-state provenance.
- Route-control objective audit ran on 4 GPUs with route-KL seeds 42/43 plus action-CE and legacy name-only controls.
- V3 metric tests passed: 10 focused tests passed, including learnability metrics and metric pairing contracts.
- All generated artifacts pass `sha256sum -c SHA256SUMS`; no residual torchrun/vLLM experiment processes remain and all 4 GPUs are idle.

Completed control results:

```text
Base                         completed as route-control reference
Full-Harness SFT/action-CE  completed as route-control objective
Standard OPD/route-KL       completed as route-control objective, seeds 42/43
Objective diagnosis         NO_OBJECTIVE_RESCUE
Recommended interpretation  released Harness-1 is a placement-boundary setting
```

Important qualification: these completed entries are V3 route-policy controls, not a completed 7B LoRA/closed-loop Search baseline factory. They do not support `Student_after > Student_base`, `Ours > OPSD`, `Ours > SEED`, or `Structured > Matched Text`.

H100-1 V2 synchronization barrier:

- H100-1 produced `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/VERIFY_PRIVILEGE_SCHEMA_V2.md` and `SELECT_POSITIVE.jsonl`.
- V2 schema contains nine state-time fields: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, and remaining budget.
- The H100-4 deterministic adapter generated `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl` from 609 H100-1 positive states across 266 query ids.
- Information audit passed `609/609` structured-to-text round trips. No future labels, reward, gold answer, or generated reasoning were included.
- Matched Text OPD status is `prepared_pending_training`; the actual optimizer/teacher training has not yet been launched.

Blocked or pending methods:

```text
Matched Text OPD              prepared_pending_training; V2 data audit passed
OPSD-style                    blocked: no faithful verified textual Search trajectory contract
OPCD-style                    blocked: no frozen historical successful-search textual experience contract
SEED-style distillation-only  blocked: no skill analyzer/rescoring implementation and no stable RL pipeline
```

SCOPE registry checks:

- `SCOPE/scripts/iclr/preflight.sh` was blocked because `artifacts/datasets/round2_audit_100q/query_manifest.json` is absent.
- `SCOPE/scripts/iclr/run_baseline_dryrun.sh` was blocked under system Python because `yaml`, `torch`, `transformers`, and `vllm` are unavailable there. The approved `/opt` environment imports the core packages, but cannot repair the missing query manifest.
- SCOPE external SEED/OPID/SDAR adapters are dry-run metadata adapters, not executable full training implementations; they must not be reported as completed baselines.

Required artifacts and handoff:

```text
outputs/btp_h100_4_baselines/RUN_MANIFEST.json
outputs/btp_h100_4_baselines/STATUS_LIVE.md
outputs/btp_h100_4_baselines/BASELINE_PROTOCOL.md
outputs/btp_h100_4_baselines/FAIRNESS_AUDIT.md
outputs/btp_h100_4_baselines/MATCHED_INFORMATION_AUDIT.md
outputs/btp_h100_4_baselines/OPSD_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/OPCD_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/SEED_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/TRAINING_BUDGETS.csv
outputs/btp_h100_4_baselines/BASELINE_RESULTS.csv
outputs/btp_h100_4_baselines/BASELINE_RESULTS.md
outputs/btp_h100_4_baselines/MAIN_COMPARISON_TABLE.csv
outputs/btp_h100_4_baselines/MAIN_COMPARISON_TABLE.md
outputs/btp_h100_4_baselines/PAIRED_BOOTSTRAP.csv
outputs/btp_h100_4_baselines/COMPUTE_COST.csv
outputs/btp_h100_4_baselines/H1004_BTP_BASELINE_HANDOFF.json
outputs/btp_h100_4_baselines/SHA256SUMS
```

Recovery instructions for the next server/agent:

1. Do not rerun the old boolean-only representation experiment as the matched-text result.
2. Start from `matched_v2/matched_v2_pairs.jsonl` and preserve the 609/609 information-equivalence audit.
3. Integrate the V2 structured fields into the existing HF teacher/student optimizer path, then run matched-text seeds 42 and 43 with the same occupancy/update budget as the structured method.
4. Obtain and freeze the final train/valid/test query manifests before any cross-method comparison.
5. Implement or import faithful OPSD/OPCD/SEED contracts before placing those methods in the main table; otherwise keep them explicitly blocked.
6. Recompute `BASELINE_RESULTS`, paired bootstrap statistics, compute costs, handoff JSON, and `SHA256SUMS` after every new wave.

Key implementation files:

```text
scripts/run_btp_h1004_baselines.py
scripts/build_btp_matched_text_v2.py
scape/training/hf_tool_opd.py
outputs/btp_h100_4_baselines/objective_controls/
outputs/btp_h100_4_baselines/matched_v2/
```

## 2026-08-16 H100-4 Novelty / Matched Text / OPHSD / importance_tagging update

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Task scope:

- Followed `/mnt/songzijun/Capability_Evolution/SCAPE/todo/0816-1/README.md` and `H100-4_novelty_baselines.md`.
- Goals were novelty collision audit, Matched Text V2 baseline recovery, faithful OPHSD-style feasibility audit, and second-component value mining for `importance_tagging`.
- GPU-heavy work used `/opt/scape-hf-scorer/bin/python`; no `/mnt` JuiceFS Python/torch environment was used.
- `CLAUDE.md` experience was read from the H100 worktree roots and the environment rule was followed.
- Current machine had 4 visible H100-class GPUs and all were idle after completion.

Novelty setting and conclusion:

- Required novelty matrix was generated in `NOVELTY_MATRIX_20260816.md` and `NOVELTY_RED_LINES.md`.
- Explicit red lines for paper/agents:
  - Do not claim first OPD internalization of a harness.
  - Do not claim first action-only privileged-information distillation.
  - Do not claim first non-text privileged information.
  - Do not present selective/state-matched OPD alone as the contribution.
  - Distinguish Harness-1 component placement/value from skill-program distillation.
- The safe claim is parity plus component placement/value distinction, not structured superiority.

Matched Text V2 input recovery and formal AUTO sync:

- The main checkout originally lacked `matched_v2_pairs.jsonl` and the V2 builder script.
- Recovered from sibling worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE` and synced into the main checkout:

```text
outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl
outputs/btp_h100_4_baselines/matched_v2/V2_SPLIT_MANIFEST.json
outputs/btp_h100_4_baselines/matched_v2/MATCHED_INFORMATION_AUDIT.md
scripts/build_btp_matched_text_v2.py
```

- V2 matched-information audit: `609/609` deterministic structured-to-text round trip.
- Pair rows present: `609`.
- V2 fields are state-time fields only: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, remaining budget.
- Do not use the older boolean-only representation smoke as the Matched Text main result.

- H100-2 AUTO formal outputs were then synchronized in because the todo explicitly said to prefer AUTO once frozen:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/
```

- Matched Text formal AUTO branch:
  - seeds 42,43,44,45
  - split 608/208/208
  - route objective `route_kl`
  - `AUTO_MATCHED_TEXT` completed and `route_distribution_normalized=true`
  - `student_inference_has_privilege=false`

Matched Text formal result:

```text
seed 42: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0363339968 agreement=0.8317307829856873 checkpoint_reloadable=True
seed 43: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0435201116 agreement=0.8365384936332703 checkpoint_reloadable=True
seed 44: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0406943820 agreement=0.8317307829856873 checkpoint_reloadable=True
seed 45: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0422902964 agreement=0.8365384936332703 checkpoint_reloadable=True
mean post_test_JS: 0.0407096967
mean agreement:    0.8341345788
best seed:         42
```

Real closed-loop matched-text result from H100-2 BM25 route-head runner:

```text
status: completed_real_closed_loop_bm25
n: 128
overall_reward: -0.015
curated_evidence_recall: 0.0
trajectory_recall: 0.005164930555555555
final_answer_recall: 0.0
tool_calls: 1.0
student_inference_has_privilege: false
```

Matched Text conclusion:

```text
COMPLETED_FORMAL_AUTO_SYNC_REAL_CLOSED_LOOP
```

Interpretation for other servers/agents:

- Matched Text is completed as formal AUTO sync plus real BM25 closed-loop sync.
- Structured and textual are tied in real closed loop; no structured advantage claim is allowed.
- Keep `609/609` information equivalence as the data barrier.

OPHSD-style route-level faithful adaptation:

- A route-level faithful adaptation was implemented and run with the same H100-2 AUTO formal split, using whole-harness terminal-context provenance for teacher routing while keeping student inference no-privilege.
- Seeds: `42`, `43`, `44`, `45`.
- Split: `608/208/208`.
- Route objective: `route_kl`.
- Component-local signal used: `false`.
- Student inference privilege: `false`.

OPHSD-style result:

```text
seed 42: post_test_JS=0.0416205414 agreement=0.8557692766189575 checkpoint_reloadable=True
seed 43: post_test_JS=0.0407332666 agreement=0.8942307829856873 checkpoint_reloadable=True
seed 44: post_test_JS=0.0415914729 agreement=0.884615421295166 checkpoint_reloadable=True
seed 45: post_test_JS=0.0408603474 agreement=0.8701923489570618 checkpoint_reloadable=True
mean post_test_JS: 0.0412014071
mean agreement:    0.8762014071
best seed:         43
```

OPHSD conclusion:

```text
COMPLETED_ROUTE_LEVEL_FAITHFUL_ADAPTATION
```

Interpretation for other servers/agents:

- This is a faithful route-level OPHSD-style baseline, not a claim of a stronger 7B end-to-end closed-loop win.
- It does satisfy the no-privilege inference requirement and preserves whole-harness terminal-context provenance.

importance_tagging second-component value mining:

- Component audited: `importance_tagging`.
- Source evidence: H100-4 `REAL_INF_CONFIRM128` same-state full/reduced influence confirmation.
- Output files:

```text
IMPORTANCE_VALUE_PER_STATE.jsonl
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
```

- Result:

```text
status: VALUE_POSITIVE
n_states: 512
event_support: 512
mean_I_name_normalized: 0.022636132108367416
mean_I_args_raw:        0.0048208512909089524
gate: REAL_INFLUENCE_POSITIVE
K4_value_mining: approximated_from_real_influence_confirm; corrective fork K4 not rerun in this checkout
K8_confirmation: not_run
handoff_allowed: true
```

Interpretation for other servers/agents:

- `importance_tagging` is positive enough to hand off as a second candidate component for downstream distillation consideration.
- Do not claim completed importance distillation or no-privilege closed-loop Student improvement from this H100-4 update alone.

Main comparison table status:

```text
Base Student:                  completed_real_closed_loop_bm25_auto_sync
Full Harness:                  not_rerun_exact_0816_h1004
Matched Text OPD:              completed_formal_auto_sync_real_closed_loop
OPHSD-style:                   completed_route_level_faithful_adaptation
Our Structured Component OPD:  completed_real_closed_loop_bm25_auto_sync
main_claim_allowed:            true, but only parity/no structured-advantage claim
```

Conclusion:

```text
completed_with_auto_matched_sync_and_ophsd_route_adaptation
```

Required artifacts written / verified:

```text
NOVELTY_MATRIX_20260816.md
NOVELTY_RED_LINES.md
MATCHED_TEXT_PROTOCOL.md
MATCHED_TEXT_TRAINING_CELLS.csv
MATCHED_TEXT_CLOSED_LOOP.csv
MATCHED_TEXT_HANDOFF.json
OPHSD_SEARCH_ADAPTATION.md
OPHSD_TRAINING_CELLS.csv
OPHSD_CLOSED_LOOP.csv
OPHSD_HANDOFF.json
IMPORTANCE_VALUE_PER_STATE.jsonl
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
MAIN_COMPARISON_TABLE.csv
MAIN_COMPARISON_TABLE.md
BASELINE_GAP.md
H1004_BTP_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines
sha256sum -c SHA256SUMS
# all checked files OK

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# all 4 GPUs idle after completion
```

Operational notes:

- No `run_route_representation_cell.py`, `run_privilege_representation.py`, `run_ophsd_route_head_cell.py`, `generate_h1004`, or `finalize_h1004` processes were left running.
- Relevant newly synced/added main-checkout scripts are currently untracked by git:

```text
scripts/build_btp_matched_text_v2.py
scripts/run_btp_h1004_baselines.py
scripts/run_privilege_representation.py
scripts/run_route_representation_cell.py
scripts/run_ophsd_route_head_cell.py
scripts/finalize_h1004_0816_matched_proxy.py
scripts/finalize_h1004_0816_complete.py
```

- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-4 actual-model end-to-end baseline status update

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; this update supersedes the earlier H100-4 route-level-only interpretation for any actual-model/main-table claim.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Source of actual-model evidence:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade/
```

Task scope:

- Read the 0816-2 requirements and `CLAUDE.md` environment rules.
- Reconcile H100-4 route-level baseline artifacts with the latest H100-1 paper-grade actual LoRA closed-loop outputs.
- Produce an H100-4 handoff that distinguishes actual full-model / actual LoRA evidence from auxiliary route-level evidence.
- Do not promote route-head or route-level Matched Text / OPHSD results into actual-model main-table results.
- GPU-heavy workloads were not launched in this update; `/opt/scape-hf-scorer/bin/python` was used for the finalizer and checks. This follows the SCAPE rule that torch/vLLM/GPU workloads must not run from `/mnt` environments.

Frozen actual-model evaluator contract:

```text
base checkpoint: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
evaluator: local BM25 BrowseComp-compatible closed-loop from initial state
test split: test256
n_queries: 256
student inference privilege: false
query_disjoint: true
LOCAL_COMPAT_ONLY: true
official_chroma_parity: false
```

Actual-model results copied into H100-4:

```text
Base Student:
  actual_model_weights: true, full base model inference
  reward:              0.367756167853536
  trajectory_recall:   0.15273141571969703
  final_answer_recall: 0.13791542658730158
  turns:               8.46484375

Ours AUTO Component OPD:
  actual_model_weights: true, actual LoRA / merged model
  reward:              0.08242097957001901
  trajectory_recall:   0.10448218231421356
  final_answer_recall: 0.09053199404761905
  turns:               6.80859375

Shuffle control:
  actual_model_weights: true, actual LoRA / merged control model
  reward:              0.3626905445160198
  trajectory_recall:   0.1448993281024531
  final_answer_recall: 0.13947792658730157
  turns:               8.37109375

First-turn-only control:
  actual_model_weights: true, actual LoRA / merged control model
  reward:              0.38749683209932373
  trajectory_recall:   0.14608233563311693
  final_answer_recall: 0.14324776785714288
  turns:               8.1484375
```

Actual-model gate result:

```text
real_closed_loop_pass:      false
student_beats_base:         false
unshuffled_beats_shuffle:   false
recommended_for_main_table: false
AUTO - Base reward delta:   -0.285335188283517
AUTO - Shuffle reward delta:-0.28026956494600075
invalid_tool_ok:            true
```

Conclusion:

```text
AUTO_ACTUAL_LORA_REAL_CLOSED_LOOP_FAILED_GATE
```

Interpretation for other servers/agents:

- Do **not** use the earlier route-level `Structured/Matched Text > Base by +0.03` result as an actual-model or paper main-table win.
- The controlling actual-model evidence says AUTO actual LoRA does not beat Base and does not beat the shuffled control on the local BM25 real closed-loop test256 run.
- Same-state route-proxy gains remain useful diagnostics only; they are not sufficient for the paper-grade claim chain `Harness component -> Student weights -> no-privilege real task win`.
- Current H100-4 main table should mark `recommended_for_main_table=false` for AUTO as a positive result.
- If the project continues this branch, the next valid action is substantive redesign / new data-contract work, not more route-head reporting.

Full Harness status:

```text
Full Harness exact same-contract reference: missing_required_gap
```

- No completed exact H100-1 paper-grade same-contract Full Harness run was found.
- The available official launcher requires an external vLLM/server path and was not completed under the paper-grade contract.
- Keep Full Harness reward/evidence/tool-cost fields as `NA`; do not impute from proxy or old runs.

Matched Text / OPHSD actual LoRA status:

```text
Matched Text OPD: route_level_only_not_actual_lora; actual LoRA blocked
OPHSD-style:      route_level_only_not_actual_lora; actual LoRA blocked
```

Blocking details:

- `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl` is an information-equivalence / textualization audit file. It has `prompt_student`, `structured_privilege`, `textual_privilege`, and round-trip status, but it lacks the `prompt_reduced`, `P_teacher_route`, `P_ref_route`, and `route_actions` contract required by `scripts/train_route_opd.py`.
- OPHSD artifacts under `outputs/btp_h100_4_baselines/ophsd/cells/OPHSD_ROUTE_CONTEXT_seed*/` are `route_head.pt` route-level baselines, not PEFT/LoRA actual Student model checkpoints.
- Therefore Matched Text and OPHSD must remain `NA_actual_lora` in the actual-model table until a proper same-state actual-LoRA training dataset and evaluator contract are implemented.

Novelty / red-line status:

- `NOVELTY_MATRIX_20260816_LATE.md` and `NOVELTY_RED_LINES_LATE.md` remain the active guard documents.
- Do not claim first OPD harness internalization, first non-text privileged information, selective OPD itself as novelty, or structured intervention / skill-program internalization as exclusive.
- The unresolved experiment needed for the intended contribution is still information-matched structured-vs-textual actual Student OPD under the same no-privilege real closed-loop evaluator, plus faithful OPHSD and Full Harness references.

Required artifacts updated / verified:

```text
END2END_BASELINE_PROTOCOL.md
FULL_HARNESS_REAL_CLOSED_LOOP.csv
MATCHED_TEXT_LORA_TRAINING.csv
MATCHED_TEXT_REAL_CLOSED_LOOP.csv
OPHSD_LORA_TRAINING.csv
OPHSD_REAL_CLOSED_LOOP.csv
STANDARD_OPSD_STATUS.md
END2END_MAIN_TABLE.csv
END2END_MAIN_TABLE.md
END2END_PAIRED_BOOTSTRAP.csv
END2END_COMPUTE_COST.csv
NOVELTY_MATRIX_20260816_LATE.md
NOVELTY_RED_LINES_LATE.md
BASELINE_GAP_LATE.md
H1004_END2END_BASELINE_HANDOFF.json
SHA256SUMS
```

Additional source files copied for provenance:

```text
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.md
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/AUTO_TRAINING_CELLS.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/PAIRED_BOOTSTRAP_AUTO.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/H1001_AUTO_PAPERGRADE_HANDOFF.json
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/BEST_AUTO_PAPERGRADE_STUDENT.json
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/AUTO_PAPERGRADE_SPLIT_MANIFEST.json
```

Finalizer script added:

```text
scripts/finalize_h1004_actual_status_0817.py
```

Verification performed:

```text
/opt/scape-hf-scorer/bin/python scripts/finalize_h1004_actual_status_0817.py
# status: actual_status_finalized

test -s outputs/btp_h100_4_baselines/SHA256SUMS
# SHA256SUMS present, 64 lines

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
# 4 GPUs idle, 1 MiB used each, 0% utilization

ps ... | rg 'train_route_opd|run_btp_auto_lora|rollout_harness|vllm|torchrun'
# no SCAPE experiment processes found
```

Operational notes:

- No GPU jobs were left running.
- No cleanup was required; GPUs were idle before and after the update.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-2 importance_tagging proper K4/K8 formal gate

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; downstream importance LoRA/real closed-loop is blocked by the formal gate result.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_formal_0817/
```

Source fork/replay directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_recovered/
```

Task scope:

- Followed the 0816-2 H100-2 requirement to replace approximate `importance_tagging` influence with proper same-`xi_t` fork/replay.
- Contract: Teacher/full first branch has `importance_tagging` ON; Student/reduced first branch has it OFF; both continuations use reduced policy; no full-harness takeover.
- Formal shards: `512 states x 2 seeds x K4/K8 = 2048 rows`.
- GPU-heavy fork/replay used `/opt/vllm-qwen3-1.7b-harness/bin/python`; the finalizer also used the `/opt` environment. No `/mnt` Python/torch environment was used.

Formal proper fork result:

```text
status: proper_k4_k8_gate_failed
seed8423 K4 mean T-S = -0.010283, ci95 [-0.013769, -0.006797], pos/neg/zero = 168/304/40
seed8423 K8 mean T-S = -0.016494, ci95 [-0.021989, -0.010999], pos/neg/zero = 173/302/37
seed8424 K4 mean T-S = -0.008408, ci95 [-0.011749, -0.005067], pos/neg/zero = 176/291/45
seed8424 K8 mean T-S = -0.013740, ci95 [-0.018943, -0.008537], pos/neg/zero = 189/291/32
proper_k4_positive: false
k8_direction_consistent_positive: false
gate_passed: false
```

Mechanism summary:

```text
first-action student marginal: end_search=1308, read_document=740
first-action teacher marginal: read_document=1308, end_search=740
positive states: 706
negative states: 1188
zero states: 154
mean teacher-minus-student tool cost delta: +0.815430
```

Conclusion:

```text
IMPORTANCE_TAGGING_PROPER_FORK_GATE_FAILED
```

Interpretation for other servers/agents:

- Do not start `importance_tagging` actual LoRA OPD, real closed-loop evaluation, paired bootstrap, or causal control from this gate result.
- The earlier approximate `REAL_INFLUENCE_POSITIVE` signal is superseded for paper-grade causal claims by this proper fork result.
- The full branch changes the first action, but usually by making the branch spend more read/tool cost without improving the formal objective under reduced continuation.
- Under the 0816-2 rule, `importance_tagging` is not a valid second positive component unless a later contract audit finds a concrete bug or a new component-aligned target is justified from fresh evidence.

Required artifacts written / verified:

```text
IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl
IMPORTANCE_K4_K8_GATE.json
IMPORTANCE_MECHANISM_CASES.jsonl
IMPORTANCE_MECHANISM_ANALYSIS.md
IMPORTANCE_TARGET_CONTRACT.md
IMPORTANCE_DATA_AUDIT.md
IMPORTANCE_DATA_AUDIT.json
IMPORTANCE_LORA_TRAINING_CELLS.csv
IMPORTANCE_REAL_CLOSED_LOOP.csv
IMPORTANCE_REAL_CLOSED_LOOP.md
IMPORTANCE_CAUSAL_CONTROL.csv
IMPORTANCE_PAIRED_BOOTSTRAP.csv
BEST_IMPORTANCE_STUDENT.json
H1002_IMPORTANCE_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_importance_proper_formal_0817.py
# status: proper_k4_k8_gate_failed

cd outputs/0816_2_importance_proper_formal_0817 && sha256sum -c SHA256SUMS
# all checked files OK
```

Operational notes:

- Initial cleanup of stale PID `38804` and duplicate PID `233427` was blocked by the permission policy because they were not created in this session. After explicit user authorization, both processes were stopped; PID `38804` required `kill -9` after ignoring SIGTERM.
- Final cleanup check: no `run_h100_2_live_fork_replay` / `importance_tagging` residual processes remained, and all 8 GPUs reported `1 MiB` used with `0%` utilization.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 0816-2 next-round execution status: AUTO and importance gates

Status: completed gate reconciliation in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; no GPU-heavy training was launched because both controlling gates block their downstream LoRA expansion.

Canonical new output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_final/
```

Task scope:

- Read `/mnt/songzijun/CLAUDE.md` and the 0816-2 next-round specs.
- Confirm H100-1 AUTO actual-LoRA paper-grade real closed-loop result remains the controlling evidence for the AUTO branch.
- Aggregate the recovered formal `importance_tagging` proper fork shards into final gate artifacts.
- Update cross-agent status so downstream servers do not launch blocked LoRA training from failed gates.
- Environment rule followed: used `/opt/vllm-qwen3-1.7b-harness/bin/python` for aggregation; no `/mnt` Python/torch environment was used.

Completed: AUTO actual LoRA real closed-loop gate

```text
source: /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade/
Base Student reward:              0.367756167853536
AUTO actual LoRA reward:          0.08242097957001901
Shuffle control reward:           0.3626905445160198
First-turn-only control reward:   0.38749683209932373
student_beats_base:               false
unshuffled_beats_shuffle:         false
real_closed_loop_pass:            false
recommended_for_main_table:       false
```

Conclusion:

```text
AUTO_ACTUAL_LORA_REAL_CLOSED_LOOP_FAILED_GATE
```

Interpretation:

- Do not continue expanding the old AUTO route-KL recipe as a positive paper-grade branch.
- Same-state route-proxy gains remain diagnostics only.
- The next valid AUTO action is substantive redesign/data-contract work, not more route-head reporting.

Completed: importance_tagging formal proper K4/K8 fork gate

Formal recovered source shards:

```text
outputs/0816_2_importance_proper_fork_formal_recovered/K4_seed8423/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K4_seed8424/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K8_seed8423/shards/importance_tagging_K8.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K8_seed8424/shards/importance_tagging_K8.jsonl
```

Formal proper fork result:

```text
K4 seed8423: n=512 mean_T_minus_S=-0.010283203125 median=-0.015 pos=168 neg=304 zero=40
K4 seed8424: n=512 mean_T_minus_S=-0.008408203125 median=-0.015 pos=176 neg=291 zero=45
K8 seed8423: n=512 mean_T_minus_S=-0.016494140625 median=-0.015 pos=173 neg=302 zero=37
K8 seed8424: n=512 mean_T_minus_S=-0.013740234375 median=-0.015 pos=189 neg=291 zero=32
full_harness_takeover_count: 0 for all four shards
```

Gate result:

```text
status: proper_fork_formal_gate_failed
proper_K4_positive: false
K8_direction_consistent_positive: false
gate_passed: false
decision: discard_importance_tagging_as_positive_component; do_not_start_importance_lora_opd
```

Interpretation:

- This supersedes the older approximate `REAL_INFLUENCE_POSITIVE` and 64-state smoke gate.
- Do not launch `importance_tagging` actual-LoRA OPD from this result.
- Per 0816-2 decision rules, switch to another high-level structured control candidate or perform a substantive contract/mechanism audit; do not try to rescue this by loss sweeping.

Required artifacts written / verified:

```text
outputs/0816_2_importance_proper_fork_formal_final/RUN_MANIFEST.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_SUMMARY.csv
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_K4_K8_GATE.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_DATA_AUDIT.md
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_MECHANISM_ANALYSIS.md
outputs/0816_2_importance_proper_fork_formal_final/H1002_IMPORTANCE_HANDOFF.json
outputs/0816_2_importance_proper_fork_formal_final/SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_importance_proper_fork_0817.py
# status: proper_fork_formal_gate_failed

cd outputs/0816_2_importance_proper_fork_formal_final && sha256sum -c SHA256SUMS
# all OK

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# all 8 GPUs idle, 1 MiB used each, 0% utilization
```

Current 0816-2 status buckets:

```text
已完成:
- AUTO actual-LoRA real closed-loop + shuffle/first-turn controls: completed, failed gate.
- importance_tagging proper K4/K8 formal fork: completed, failed gate.
- H100-4 actual-model reconciliation: completed; Matched Text/OPHSD remain route-level only, not actual LoRA.
- Full Harness exact same-contract reference: resolved blocked by missing same-contract runner binding, not by GPU/env.
- Matched Text actual-LoRA baseline: resolved blocked by missing prompt/teacher/response training contract.
- OPHSD actual-LoRA baseline: resolved blocked by missing faithful prompt/response actual-LoRA contract.
- Structured actual Student V1/V2 redesign: resolved blocked after audit; current data supports route-head only, not valid actual-LoRA V1/V2.

正在进行:
- None. No SCAPE experiment process is running at this update.

未开始:
- None under the current 0816-2 artifact set. New work requires new data/runner-contract implementation, not simply launching existing experiments.
```

Remaining-resolution artifacts:

```text
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.md
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.json
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.csv
outputs/0816_2_remaining_resolution_final/RUN_MANIFEST.json
outputs/0816_2_remaining_resolution_final/SHA256SUMS
```

Additional 0816-2 execution after stop-hook escalation:

```text
Full Harness same-contract reference:
  output: outputs/0816_2_full_harness_same_contract_test256/
  status: completed_real_closed_loop_reference
  n_queries: 128 (source manifest only exposed 128 usable query ids)
  runtime flags: V8D_AUTO_POPULATE_FIRST_SEARCH=1, V8D_IMPORTANCE_TAGGING=1, V8D_VERIFY_TOOL=1, V8D_TOKEN_BUDGET_MARKER=1
  internal method label: BASE_REDUCED from reused evaluator
  overall_reward: 0.024405273437500003
  trajectory_recall: 0.006510416666666667
  final_answer_recall: 0.0078125
  error_rate: 0.0

Matched Text actual-LoRA bridge:
  train output: outputs/0816_2_actual_lora_bridge/matched_text/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/matched_text/seed42/checkpoint
  smoke16 reward: 0.14961805555555555
  mean_reward_delta_vs_base: 0.0

Structured actual Student bridge:
  train output: outputs/0816_2_actual_lora_bridge/structured/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/structured/seed42/checkpoint
  smoke16 reward: 0.14961805555555555
  Structured - Matched Text smoke delta: 0.0

OPHSD actual-LoRA bridge:
  train output: outputs/0816_2_actual_lora_bridge/ophsd/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/ophsd/seed42/checkpoint
  smoke16 reward: 0.13705555555555554
  mean_reward_delta_vs_base: -0.0125625
```

Caveats for other servers/agents:

- The Full Harness reference used the actual-LoRA evaluator path with full V8D runtime flags enabled before import. The evaluator's internal method label remains `BASE_REDUCED`; interpret by output directory and runtime contract, not by the label alone.
- The Matched Text / Structured / OPHSD bridge runs are real PEFT/LoRA adapters and real closed-loop smoke results, but their training rows are generated from route-distribution argmax to canonical tool-call text. They are dev/smoke evidence, not a replacement for paper-grade recollected prompt/response teacher rows.
- The bridge tool-span audit reports `parsable_rate=0.0` under the existing `scape_tool_mask_v1`, so these bridge adapters should not be promoted as final actual-LoRA baselines without repairing the response rendering/mask contract.

Final 0816-2 status artifacts:

```text
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.md
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.json
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.csv
outputs/0816_2_final_experiment_status/RUN_MANIFEST.json
outputs/0816_2_final_experiment_status/SHA256SUMS
```

Operational notes:

- Finalizer scripts added: `scripts/finalize_importance_proper_fork_0817.py`, `scripts/finalize_0816_2_remaining_resolutions.py`, `scripts/run_0816_2_actual_lora_bridge.py`, `scripts/finalize_0816_2_final_experiment_status.py`.
- `/opt/bishop-harness/bin/python` has torch/transformers/peft/vllm/pyserini and was used for GPU/LoRA work; no `/mnt` Python/torch environment was used.
- No SCAPE GPU workers were left running; all 8 GPUs were idle after completion.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 0816-2 final status: all requested experiments resolved

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`. This supersedes the earlier continuation-status note that listed an isolated K8 rerun as running. That rerun and the obsolete duplicate/blocked fork processes have been stopped after user authorization, and the canonical formal gate uses the already complete recovered 2048-row proper-fork output.

Canonical final status directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_final_status_0817/
```

Completed / failed gates:

```text
H100-1 AUTO actual LoRA real closed-loop + shuffle:
  status: completed_failed_gate
  decision: do_not_claim_student_win; redesign_required
  result: student_beats_base=false; unshuffled_beats_shuffle=false; real_closed_loop_pass=false

H100-2 importance_tagging proper K4/K8:
  status: completed_failed_gate
  decision: do_not_start_importance_lora_opd
  result: k4_positive=false; k8_positive=false; gate_passed=false
```

Completed / blocked by required gate or faithful-contract audit:

```text
H100-2 importance actual LoRA + real closed-loop + causal control:
  status: completed_blocked_by_failed_gate
  reason: proper fork gate failed; launching LoRA would violate the 0816-2 Go/Discard rule

H100-3 Structured actual Student V1/V2:
  status: completed_blocked_or_not_supported_by_existing_artifacts
  reason: existing evidence is route-head parity with structured_vs_textual_delta=0.0, not actual LoRA; new actual-LoRA V1/V2 contract is required

H100-4 Full Harness same-contract reference:
  status: completed_missing_required_runner_gap
  reason: faithful exact same-contract Full Harness runner/server binding is missing; result must remain NA and must not be imputed

H100-4 Matched Text actual LoRA:
  status: completed_blocked_contract_missing
  reason: matched_v2_pairs lacks prompt_reduced, P_teacher_route, P_ref_route, route_actions required by the actual LoRA trainer

H100-4 OPHSD actual LoRA:
  status: completed_blocked_contract_missing
  reason: existing OPHSD artifacts are route_head.pt cells; no faithful whole-harness actual-LoRA training/evaluator contract exists
```

Current buckets:

```text
已完成:
- AUTO actual-LoRA real closed-loop + shuffle/first-turn controls: completed, failed gate.
- importance_tagging proper K4/K8 formal fork: completed, failed gate.
- importance actual-LoRA/real-closed-loop/causal-control branch: completed as blocked by failed gate.
- Structured actual Student V1/V2 branch: completed as blocked/not supported by existing artifacts; route-head parity only.
- Full Harness exact same-contract reference: completed as missing faithful runner gap; NA, no imputation.
- Matched Text actual-LoRA baseline: completed as blocked by missing training contract.
- OPHSD actual-LoRA baseline: completed as blocked by missing faithful actual-LoRA contract.
- Residual process cleanup: completed clean.

正在进行:
- None.

未开始:
- None under the current 0816-2 artifact set. Any future progress requires new data/runner-contract implementation or substantive redesign, not simply launching remaining queued jobs.
```

Required artifacts written / verified:

```text
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.csv
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.json
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.md
outputs/0816_2_final_status_0817/PROCESS_STATUS.txt
outputs/0816_2_final_status_0817/GPU_STATUS.txt
outputs/0816_2_final_status_0817/SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_0816_2_experiment_status_0817.py
# status: 0816_2_final_status_written, residual_processes=0

cd outputs/0816_2_final_status_0817 && sha256sum -c SHA256SUMS
# all checked files OK

ps ... | rg 'run_h100_2_live_fork_replay|importance_tagging|0817_importance|train_route_opd|run_btp_auto_lora|closed_loop|torchrun|vllm'
# no output

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
# GPU 0-7 all 1 MiB used, 0% utilization
```

Operational notes:

- Environment rule followed: finalizers used `/opt/vllm-qwen3-1.7b-harness/bin/python`; no `/mnt` Python/torch environment was used for GPU/torch work.
- Residual process cleanup required explicit user authorization because the stale PIDs were not created in this session. After authorization, duplicate K8 exited on SIGTERM and stale K4 required `kill -9`.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-2 importance formal gate provenance correction

Status: completed. This note supersedes earlier `importance_tagging` gate provenance/numeric summaries that referenced `outputs/0816_2_importance_proper_fork_formal_recovered/` as canonical.

Canonical latest output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_final/
```

Latest source shards:

```text
outputs/0816_2_importance_proper_fork_formal_stream/K4_seed8423/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K4_seed8424/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K8_seed8423/shards/importance_tagging_K8.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K8_seed8424/shards/importance_tagging_K8.jsonl
```

Recovery performed:

- `K8_seed8424` formal stream shard was append-resumed from 410 rows to 512/512 rows with `scripts/resume_importance_k8_seed8424_append_0817.py` using `/opt/scape-h1003-hf-scorer-cu128/bin/python`.
- A slower isolated full rerun under `outputs/0817_importance_k8_seed8424_full_rerun/` was stopped after the formal shard reached 512/512 and is not part of the canonical gate.
- `scripts/finalize_importance_proper_fork_0817.py` was parameterized to reject incomplete shards by default and then used to write the canonical formal gate.

Latest canonical result:

```text
status: proper_fork_formal_gate_failed
K4 seed8423: n=512 mean_T_minus_S=-0.010898438 median=-0.015 pos=157 neg=313 zero=42
K4 seed8424: n=512 mean_T_minus_S=-0.008935547 median=-0.015 pos=169 neg=297 zero=46
K8 seed8423: n=512 mean_T_minus_S=-0.017021484 median=-0.015 pos=159 neg=309 zero=44
K8 seed8424: n=512 mean_T_minus_S=-0.014091797 median=-0.015 pos=179 neg=297 zero=36
full_harness_takeover_count: 0 for all four shards
proper_K4_positive: false
K8_direction_consistent_positive: false
gate_passed: false
decision: discard_importance_tagging_as_positive_component; do_not_start_importance_lora_opd
```

Artifacts:

```text
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_K4_K8_GATE.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_MECHANISM_ANALYSIS.md
outputs/0816_2_importance_proper_fork_formal_final/H1002_IMPORTANCE_HANDOFF.json
outputs/0816_2_importance_proper_fork_formal_final/LATEST_PROVENANCE_CORRECTION.md
```

Operational status:

- No SCAPE recovery or training processes remain.
- All 8 GPUs were idle after finalization.
