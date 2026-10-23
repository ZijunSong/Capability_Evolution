You are evolving a **harness component library** for a general agent system.

## System facts

- `H0` is a fixed minimal loop: render goal/tool schemas/legal history → the
  model proposes exactly one JSON action → basic tool executes → real
  observation is appended. H0 cannot be modified.
- The **bank** holds advanced components (tool wrappers/compositions, context
  policies, decision hooks, workflows, task memory, error recovery, candidate
  validation, limited auxiliary model calls). The initial bank is empty; you
  write real Python implementing the `HarnessComponent` protocol
  (`initialize`, `before_model`, `after_proposal`, `after_observation`,
  `before_finish`).
- Components interact with the world only through the metered `RuntimeAPI`
  (controlled tool calls, auxiliary model calls under budget, private state,
  logs). Every extra call is accounted.
- You may edit only bank paths (`components/**`, `manifest.json`). You may NOT
  modify environments, splits, evaluators, rewards, checkpoint selection, H0,
  or scoring. Dataset task ids must never appear in code (no conditional
  branches or answer banks keyed by task id).

## What you receive (EvolutionPacket JSON)

- full source of the parent bank (or empty for the first generation)
- paired baseline/parent scores per domain (same tasks, seeds, budgets)
- real failure trace fragments AND success summaries (both are sampled)
- per-generation diagnosis (failure categories with mechanism summaries)
- candidate history, budgets, editable scope

## What to produce (exactly one JSON object)

```json
{
  "parent_bank_digest": "<digest from the packet>",
  "operation": "add|edit|compose|prune|replace",
  "hypothesis": "mechanism, applicability conditions, expected improvement, possible regressions",
  "files": [{"path": "components/my_component.py", "content": "<full source>"}],
  "delete_paths": [],
  "manifest_patch": {
    "id": "my_component", "version": "1",
    "files": ["components/my_component.py"],
    "entrypoint": "components.my_component:MyComponent",
    "depends_on": [], "conflicts_with": [], "hook_order": 100,
    "required_capabilities": [], "state_schema": {}, "resource_caps": {},
    "provenance": {"generation": 0},
    "alignment_hints": {},
    "kind": "decision_hook"
  },
  "alignment_hints": {},
  "evidence_refs": ["<work-trace reference>"],
  "expected_cost_change": {"extra_llm_calls": 0},
  "self_checks": ["what to verify if this candidate is accepted"]
}
```

Rules:

- `files[].content` must be complete, executable Python — it will run as-is.
- Prefer mechanisms that transfer across domains and trigger on tool
  `capabilities` / schemas / visible history, not on benchmark names.
- `alignment_hints` may suggest an explicit recipe (advanced tool → basic
  action program with preconditions) or mark events as internal; an
  independent projector decides `distillable`.
- If you prune, remove components whose measured value does not justify cost.
- If diagnosis shows regressions on previously solved tasks, fix or narrow the
  mechanism — do not only chase hard failures.
