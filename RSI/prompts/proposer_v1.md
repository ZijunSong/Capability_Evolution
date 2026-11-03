You are evolving a **harness component library** for a general agent system.

## System facts

- `H0` is a fixed minimal loop: render goal/tool schemas/legal history → the
  model proposes exactly one JSON action → basic tool executes → real
  observation is appended. H0 cannot be modified.
- The **bank** holds advanced components (tool wrappers/compositions, context
  policies, decision hooks, workflows, task memory, error recovery, candidate
  validation, limited auxiliary model calls). The initial bank is empty; you
  write real Python implementing the `HarnessComponent` protocol.
- Components interact with the world only through the metered `RuntimeAPI`
  (controlled tool calls, auxiliary model calls under budget, private state,
  logs). Every extra call is accounted.
- You may edit only bank paths (`components/**`, `manifest.json`). You may NOT
  modify environments, splits, evaluators, rewards, checkpoint selection, H0,
  or scoring. Dataset task ids must never appear in code (no conditional
  branches or answer banks keyed by task id).

## EXACT importable API surface (nothing else exists)

From `rsi.harness.component` ONLY — there are no submodules (never write
`rsi.harness.component.dataclasses` or invent types like `ActionLog`,
`ToolResult`, `Step`, `ToolResponse`):

```python
from rsi.harness.component import (
    EpisodeContext,   # fields: episode_id, task, tool_catalog, private_state,
                      #         rng_seed, extra_llm_call_budget, config (budgets dict)
    PublicView,       # fields: task, tool_catalog, history, remaining_action_budget, extra
    RuntimeAPI,       # methods: call_tool(ToolCall)->Observation, call_model(str, max_new_tokens)->str,
                      #          get_state()->Mapping, set_state(Mapping), budget()->Mapping, log(msg, **fields)
    ViewPatch,        # (append_instructions, history_filter, extra, changed)
    ProposalEffect,   # (action, veto, veto_reason, notes)
    StatePatch,       # (set, notes)
    FinishEffect,     # (block, block_reason, action)
)
from rsi.contracts.action import ToolCall, Finish   # action types
from rsi.contracts.events import Event              # after_observation event
```

Event and Observation are DATACLASSES (attribute access — never treat them as
dicts; `event.get(...)` is a runtime error):

```
Event fields: event_id, order, component_id, event_kind, pre_state_digest,
post_state_digest, action (ToolCall|Finish|None), observation (Observation|None),
visibility, effect_type, information_provenance (tuple), cost, data (Mapping)
Observation fields: content (str), ok (bool), error_kind (str|None),
provenance (object with .source_tool/.source_ref), structured (Mapping)
```
So `event.observation.content`, `event.action.name`, `dict(event.data)` are
valid; `event["observation"]` / `event.get("x")` are NOT.

EpisodeContext also exposes `ctx.config` with `{"max_actions", "max_tool_calls",
"max_extra_llm_calls", "max_wall_ms", "rng_seed"}`.  Imports of any other
`rsi.*` submodule are allowed ONLY for type annotations of these names.

## HARD REQUIREMENTS (violations = candidate rejected)

1. **Every component class MUST define ALL FIVE hooks with EXACTLY these
   parameter lists** (self is implicit; do not omit any parameter; optional
   extra parameters are allowed only with defaults):

```python
from rsi.harness.component import (
    EpisodeContext, FinishEffect, ProposalEffect, PublicView, RuntimeAPI,
    StatePatch, ViewPatch,
)

class MyComponent:
    def initialize(self, ctx: EpisodeContext) -> None:
        ...

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        ...

    def after_proposal(self, action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        ...

    def after_observation(self, event, api: RuntimeAPI) -> StatePatch:
        ...

    def before_finish(self, action, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        ...
```

   Wrong: `before_model(self, view)` — rejected. Right: `before_model(self, view, api)`.
   Return the small effect dataclasses (`ViewPatch()`, `ProposalEffect(action=None)`,
   `StatePatch()`, `FinishEffect()`) when there is nothing to do. Import them
   from `rsi.harness.component`.

2. **`manifest_patch.entrypoint` class name must EXACTLY match the class
   defined in the source file.** If the file defines `class FinishGuard`,
   the entrypoint must be `components.<module>:FinishGuard`.

3. Hooks must never raise on ordinary inputs — wrap risky logic in try/except and
   type-check before use: `action.arguments` values are arbitrary JSON
   (str/int/float/...), `event.observation` may be None, `view.history` may be
   empty. A hook that raises is disabled for the episode and counts against the
   candidate's reliability score.

## COMPLETE working example (copy this structure)

```python
from rsi.harness.component import (
    EpisodeContext, FinishEffect, ProposalEffect, PublicView, RuntimeAPI,
    StatePatch, ViewPatch,
)
from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.events import Event


class ExampleGuard:
    def initialize(self, ctx: EpisodeContext) -> None:
        # ctx.get / ctx["config"] also work; attribute access is canonical.
        self.max_actions = int(ctx.config.get("max_actions", 20))
        self.seen_errors = 0

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        if self.seen_errors >= 2:
            return ViewPatch(
                append_instructions="Two errors seen; prefer simple verified actions.",
                changed=True,
            )
        return ViewPatch(changed=False)

    def after_proposal(self, action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        if isinstance(action, ToolCall) and action.name not in {t.name for t in view.tool_catalog}:
            return ProposalEffect(veto=True, veto_reason="unknown tool")
        return ProposalEffect(action=None)

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch:
        # Event/Observation are dataclasses: attribute access.
        obs = event.observation
        if obs is not None and not obs.ok:
            self.seen_errors += 1
            return StatePatch(set={"last_error": str(obs.content)[:200]})
        return StatePatch()

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        state = api.get_state()
        if state.get("last_error") and action.answer is None:
            return FinishEffect(block=True, block_reason="unresolved error and no answer")
        return FinishEffect(block=False)
```

Self-check before answering:
1. All five hooks present with EXACTLY the example signatures?
2. entrypoint class name == the class in the file?
3. Only imports from `rsi.harness.component`, `rsi.contracts.*`?
4. Every hook returns an effect object (ViewPatch/ProposalEffect/StatePatch/FinishEffect)?
5. No dataset task ids, no benchmark names, no invented types/attrs?

## GROW the bank — complexity and capability are the goal

You are building toward a complex, capable harness (think: verification,
repair loops, state tracking, plan-then-act, budgeted auxiliary calls), not a
single throwaway hook.  Rules for growth:

- **Your patch extends the CURRENT parent bank** (the champion of the search
  so far).  Its source is in the packet.  You may:
  - `add` new components that COOPERATE with existing ones (declare
    `depends_on`, order with `hook_order`),
  - `edit` an existing component's file to improve it,
  - `compose` several new components in ONE patch (multi-file + multi-entry
    manifest_patch.components list),
  - `prune` a component whose measured cost is not justified.
- **Multi-component patch example** (manifest_patch with the `components`
  list form):

```json
{
  "manifest_patch": {
    "components": [
      {"id": "error_tracker", "version": "1", "files": ["components/error_tracker.py"],
       "entrypoint": "components.error_tracker:ErrorTracker", "hook_order": 10,
       "kind": "decision_hook", "depends_on": [], "conflicts_with": [],
       "required_capabilities": [], "state_schema": {}, "resource_caps": {},
       "provenance": {"generation": 1}, "alignment_hints": {}},
      {"id": "action_verifier", "version": "1", "files": ["components/action_verifier.py"],
       "entrypoint": "components.action_verifier:ActionVerifier", "hook_order": 30,
       "kind": "candidate_validation", "depends_on": ["error_tracker"], "conflicts_with": [],
       "required_capabilities": [], "state_schema": {}, "resource_caps": {},
       "provenance": {"generation": 1}, "alignment_hints": {}}
    ]
  }
}
```

- **Prefer score-improving mechanisms over damage control.**  The diagnosis
  tells you why tasks FAIL; write mechanisms that make them PASS:
  - wrong results (`task_error`/`wrong_result`) -> verification before finish
    (re-read/compare via `api.call_tool`), or a `api.call_model` self-check
    (each aux call costs budget — `ctx.config["max_extra_llm_calls"]` is the
    cap, use it where it pays),
  - premature `finish` -> gate finish on explicit completion checks,
  - `max_actions` -> context compaction, better action planning in
    `before_model` guidance, or early stop on hopeless loops,
  - tool misuse -> argument validation/repair in `after_proposal`.
- Measured gain on the panels decides; complexity without gain is pruned by
  cost.  But a mechanism that converts failures into passes is the goal.

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
