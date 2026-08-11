# LEGACY_SCOPE_NOTES

Historical lessons from `/data/ppnm/Capability_Evolution/SCOPE` that inform SCAPE **without code dependence**.

## Positive historical example

- `duplicate_evidence` local KEEP/SKIP behavior was learnable in closed loop under SCOPE’s typed label setup (Round7–8 / Round14 Dup calibration).
- Interpretation for SCAPE: *simple local harness behaviors can sometimes be internalized* — but that pipeline is capability-specific and is **not** the SCAPE method.

## Hard negatives

- Full **rollback** closed loop did not establish retirement (R13 / R14-lite Gate B fail: recover recall ~0.31–0.33).
- Do not reopen multi-round “checkpoint selector rescue” for rollback under SCAPE GPU budget.

## Metric warnings

- Canonical final-answer accuracy on BrowseComp+ can be a **floor / mismatch** relative to harness reward and retrieval metrics.
- SCAPE must report curated/trajectory recall, harness reward, and cost — not only exact final answer.

## Allowed tool migrations (already reimplemented under `scape/`)

- paired W/L/T + bootstrap CI
- stable hash splits / manifest freezing patterns
- resume / SHA256SUMS / STATUS_LIVE / result-record appenders
- GPU queue launcher skeletons

## Forbidden method carry-over

`P_m`, KEEP/SKIP schemas, rollback classifiers, O7, Information-Safe Gate, old ModuleRetirementGate.
