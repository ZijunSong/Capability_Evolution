# Diagnosis summary format (per generation)

Diagnosis is refreshed every generation from real traces.  Categories and
mechanism hints are produced by `rsi.evolution.diagnosis`; this document is the
human/model-facing interpretation key.

| failure kind | likely mechanism | candidate directions |
|---|---|---|
| model_invalid_action | unparsable/illegal actions | proposal validator hook, format guidance in before_model |
| tool_error | bad arguments / wrong tool | argument checking, retry controller, error-recovery component |
| max_actions / budget_wall | too long before finishing | context compression, progress summary, termination heuristic |
| wrong_result | executed but wrong artifact | verification-before-finish, candidate validation |
| finish_blocked | gating component too strict | tune or prune the gate |

Rules for the diagnosis consumer:

- Success samples are reviewed too: mechanisms must not break learned abilities.
- Regression task ids are audit references only — never encode them in code.
- Structural trigger counts are NOT training-admission evidence; only real
  OPD probe statistics are (`rsi.opd.probe`).
