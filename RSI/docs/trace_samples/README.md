# E1 webshop-9B trace samples: pre-evolution vs post-evolution

Paired final eval on the frozen `harness_select` 12-task split (official human
goals, asin-grouped, manifest `4a46047d`, seed 17). Same config, same seeds,
7 GPU panel workers for both arms. Source job:
`t-20261104203246-juytd` (rsi-e1-paireval-webshop-9b-8gpu-20261104-r1).

## Files

| file | what it is |
|---|---|
| `traces_baseline_minimal.json` | Arm A — H0 baseline (empty bank, no evolved components). macro **5.56%**, 1/12 tasks nonzero, 68 steps total. |
| `traces_champion_bank.json` | Arm B — champion bank `accbce6a` (goal_completion_guard + action_recovery_guard). macro **45.28%**, 9/12 nonzero, 143 steps. |
| `SUMMARY.json` | per-arm macro/tasks/steps. |
| `evolve_result_webshop_9b_r9.json` | Full evolution search record (11 candidates, 6 generations, selection trace) that produced the champion. |
| `champion_bank_src/*.py` | The two evolved component sources (model-proposed, unmodified). |

## Paired per-task delta (baseline → champion)

```
task      baseline  champion   delta
g00821       0.000     0.500  +0.500
g00870       0.000     0.667  +0.667
g02742       0.000     0.100  +0.100
g04025       0.000     0.667  +0.667
g05013       0.667     0.667  +0.000
g05234       0.000     0.667  +0.667
g07359       0.000     0.000  +0.000
g07507       0.000     0.000  +0.000
g07863       0.000     0.667  +0.667
g08140       0.000     0.500  +0.500
g09442       0.000     1.000  +1.000
g09904       0.000     0.000  +0.000
                      8 win / 0 loss / 4 tie, macro +39.72pp
```

## What the champion components do

- **goal_completion_guard** (`GoalCompletionGuard`): vetoes `None` action
  proposals ("unknown action type None" was the dominant baseline failure
  mode — see `steps[kind=error]` in the baseline traces), and injects
  "do not finalize unless explicitly completed" guidance when the last
  observation errored.
- **action_recovery_guard** (`ActionRecoveryGuard`): recovers from malformed
  tool calls and veto/repair loops so the episode keeps making progress
  instead of burning its action budget.

## Notes for HarnessOPD analysis

- Steps format: `{order, kind, action?, obs_head, obs_tail, obs_len, obs_ok?}`.
  `kind` ∈ `episode_boundary | tool_call | error | ...`; `action` is the raw
  JSON action the model proposed; observations are head/tail truncated.
- `work_train` (12 tasks) was the evolution's own pool; `harness_select`
  (these 12) was consumed as the selection panel during evolve — so this is
  an **independent re-measure on the same frozen split**, not a fully
  held-out final test. The final_test role is declared in config
  (`final_test_access: after_campaign_freeze`) but no third split exists yet.
- Champion macro reproduced across two independent measurements:
  45.0% (evolve panel, gen2) vs 45.28% (this eval).
- Baseline failure signature worth mining for OPD pairs: repeated
  `error / invalid action format: unknown action type None` until the action
  budget wall — exactly what `action_recovery_guard` targets.
