# SCAPE Component OPD

`scape_component_opd` is a method-local EasyOPD extension for SCAPE / Beyond Textual Privilege experiments. It keeps SCAPE's Search Harness runtime as the source of truth and adds typed contracts for component-specific supervision.

The framework separates component realizability, event support, teacher-sidecar rescoring, projected action supervision, tool-span masking, controls, diagnostics, and closed-loop evaluation wrappers. It does not modify `verl/` in this first integration layer.

Key hard gates:

- `verify_tool` defaults to `NON_REALIZABLE_ACTION_SPACE_MISMATCH` unless the Student action space explicitly includes the same tool.
- `content_dedup` with zero active event support returns `STOP_NO_ACTIVE_EVENT_SUPPORT`.
- Projected actions must use real state deltas and pass visibility legality checks.
- Student inference privilege must remain false.
