# FRAMEWORK_SELECTION_AUDIT

## Decision

Default selection: **EasyOPD + verl**.

## Why this choice fits SCAPE

- EasyOPD separates method-local supervision logic from verl execution.
- SCAPE needs per-component realizability, projection, state fork/restore, and teacher-sidecar boundaries.
- The current migration should not rewrite verl core or duplicate rollout infrastructure.

## Compared frameworks

- **EasyOPD**: best fit as the primary method-oriented wrapper.
- **verl native OPD**: useful backend, but too low-level for SCAPE component-local contracts.
- **KDFlow**: useful reference for KL numerics and efficiency, not the primary architecture.
- **SOD**: useful reference for step-wise OPD, but not enough as a full SCAPE component framework.
- **OpenRLHF**: broader RLHF stack, but not the target abstraction for SCAPE component supervision.
- **TRL/GKD**: good distillation reference, but not sufficient for SCAPE state/component contracts.

## Extension point chosen

Add a new EasyOPD method package:

- `easyopd/methods/scape_component_opd/`

and keep SCAPE runtime as the source of truth.
