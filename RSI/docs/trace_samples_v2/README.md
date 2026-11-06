# E1 trace samples v2: pre- vs post-harness-fix, 9B vs 27B (webshop + alfworld)

Exported 2026-11-06. Covers the four evolution campaigns rerun after the
harness measurement-chain fixes (see commit 8790243):

1. **admissible_commands rendering** — the engine's legal-command payload
   (`Observation.structured`) is now appended to every rendered observation
   (was silently dropped).
2. **env_docs populated** — `configs/env_docs/alfworld_v1.md` and
   `webshop_v1.md` were PLACEHOLDER stubs.
3. **environment_doc PATH read** — `runtime.environment_doc` was a file PATH
   rendered as a literal filename string; the doc content never reached the
   model in E0/E1.

## Headline numbers

| domain | model | pre-fix champion | post-fix champion | baseline (gen0 best) |
|---|---|---|---|---|
| alfworld | 9B (zhfrd) | 0.250 (8hk1b, r4) | **0.750** | 0.583 |
| alfworld | 27B (gth6z) | N/A (panel skipped) | **0.833** | 0.833 |
| webshop | 9B (afe1d r10) | 0.450 (bojde r9) | **0.563** | 0.306 |
| webshop | 27B (cdyeb r11) | 0.589 (spzne r5) | **0.547** | 0.514 |

Full per-candidate score tables: `docs/e1_alfworld_evolution_results_9b_27b.csv`,
`docs/e1_webshop_rerun_results_9b.csv`, `docs/e1_webshop_rerun_results_27b.csv`.

## Files

| file | what it is |
|---|---|
| `alfworld_traces_post_27b.json` | **24 task traces** (command-level) from the 27B alfworld evolution panels (gth6z). Champion bank 83ae9fdf (`completion_gate`), macro 0.833. |
| `webshop_traces_post_27b.json` | **24 task traces** (action-level: `search[..]` / `click[..]` + purchase record) from the 27B webshop evolution panels (cdyeb). Champion-bank gen1 0.547. |
| `webshop_traces_pre_9b.json` | **12 task traces (obs-level, steps with obs head/tail + action text)** — paired final eval baseline arm (empty bank), macro 5.56%. The pre-fix behavior reference for webshop. |
| `alfworld_pre_9b_scores_only.json` | Scores/selection history only (14 evaluated + rejected proposals) for 8hk1b. **Per-step traces not recoverable** (see caveat). |
| `alfworld_traces_post_9b_v2.json` | **24 task traces (command-level)** from the trace-preservation rerun (toqpp, same fixed-harness protocol as zhfrd; dedicated OUT root — no overwrite). Champion 755b48b8 macro 0.417; panel variance vs zhfrd's 0.75 noted in meta. |
| `alfworld_post_9b_scores_only.json` | Scores/selection history for zhfrd (14 evaluated, champion 622625de 0.75) — the headline 9B alfworld run. Traces overwritten by the later 27B rerun; the rerun above preserves step-level traces of the same protocol. |
| `webshop_post_9b_scores_only.json` | Scores/selection history for afe1d (6 evaluated, champion b0ad1c7d 0.563). Traces overwritten by the later 27B rerun. |

## Trace-availability caveat (honest scope)

`final_artifact.json` files are **last-writer-wins per task_id**: when a later
campaign re-evaluates the same frozen tasks, the earlier command trajectories
are overwritten. Surviving step-level traces:

- webshop **pre** (9B): survived because the paired-eval job wrote to a
  separate OUT root (`docs/trace_samples/traces_baseline_minimal.json`, the
  original v1 export).
- alfworld/webshop **post 27B**: the final-state snapshot (24 tasks each).
- alfworld pre / post-9B, webshop post-9B: **scores only** — the step-level
  behavior was overwritten by later reruns of the same task ids. The score
  histories (every candidate, every generation, selection decisions) are
  complete and are included here instead.

## Trace format

- command-level (`*_traces_post_27b.json`): `tasks[task_id].commands` — the
  exact actor-side action sequence; the domain scorer replays these against
  the live engine to produce the recorded panel value.
- obs-level (`webshop_traces_pre_9b.json`): `tasks[task_id].steps[]` —
  per-step `{kind, action, obs_head, obs_tail, obs_len}` (bounded export).
- scores-only (`*_scores_only.json`): `generation_history[]` — every proposer
  attempt with generation, candidate id, and macro score (accepted or
  rejected), plus the final champion.
