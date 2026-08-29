# SCAPE Component OPD

`scape_component_opd` is an EasyOPD method that **wraps Harness-1** rather than reimplementing the search runtime.

- Harness-1 (`SCAPE/external/harness-1`) is source of truth for WorkingMemory, tools, and V8D components.
- EasyOPD/verl owns distributed training.
- `Harness1Bridge` forks the same Student state with the target component ON to read Teacher side effects.
- Projection is **skip-to-anchor only**: after projection a Teacher event is either
  - `align` — a Student-native tool call used as the OPD label, or
  - `skip` / ε — Harness-only event, keep scanning for the next realizable Student action.

Privileged context (Evidence Graph, sentence compress, token-budget marker, rerank instruction) is never distilled as Teacher tokens. Graph capability is learned from the Student-realizable downstream `curate` (or other native tool), after a realizability check.

Student inference privilege must remain false. `verify` is not a Student tool; it is skipped.
