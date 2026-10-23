"""A-route: explicit action mapping via recipes.

A recipe is a *basic action program with preconditions* — not a fixed string
list of assumed results.  Example (task book §8.2): an advanced
``inspect_then_fix`` (read file, check, patch) with a Student that has not read
the file yet yields a first legal target of basic ``read_file``; the later
patch content is supervised only after the read result is in the Student's
visible history — never pasting Teacher-future paths/line numbers/ids into the
current ``write_file``.

Default policy: at most ONE executable next action per Student decision.
Multi-action recipes must be executed step-by-step in an isolated
Student-compatible replay, each step emitting its own prefix/target pair,
recorded as extra recipe-replay data (never the whole macro as several
on-policy samples of one decision).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue
from rsi.contracts.trace import RealizabilityReport


@dataclass(frozen=True)
class RecipeStep:
    """One basic action in a recipe program.

    ``arguments`` may contain ``{placeholder}`` slots filled from *already
    visible* facts (student history / teacher public events) at projection time.
    ``requires_fact_refs`` lists placeholders that must be resolvable before
    this step may be supervised.
    """

    tool: str
    arguments: Mapping[str, str]
    requires_fact_refs: tuple[str, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    advanced_tool: str
    steps: tuple[RecipeStep, ...]
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)

    @staticmethod
    def from_hints(hints: Mapping[str, JSONValue]) -> "Recipe | None":
        """Build a Recipe from a component's declarative ``alignment_hints``."""
        data = hints.get("recipe") if isinstance(hints, Mapping) else None
        if not isinstance(data, Mapping):
            return None
        steps = []
        for s in data.get("steps", []):
            steps.append(
                RecipeStep(
                    tool=str(s["tool"]),
                    arguments={str(k): str(v) for k, v in dict(s.get("arguments", {})).items()},
                    requires_fact_refs=tuple(str(r) for r in s.get("requires_fact_refs", [])),
                    name=str(s.get("name", "")),
                )
            )
        return Recipe(
            recipe_id=str(data.get("recipe_id", "recipe")),
            advanced_tool=str(data.get("advanced_tool", "")),
            steps=tuple(steps),
            provenance=dict(data.get("provenance") or {}),
        )


_SLOT = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def resolve_arguments(
    step: RecipeStep,
    visible_facts: Mapping[str, str],
) -> tuple[dict[str, str] | None, tuple[str, ...]]:
    """Fill argument slots from visible facts only.

    Returns (resolved_args, unresolved_refs).  Unresolvable slots mean the step
    cannot be supervised yet (missing information) — the alignment either
    supervises the acquire action or skips.
    """
    out: dict[str, str] = {}
    unresolved: list[str] = []
    for key, template in step.arguments.items():
        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in visible_facts:
                return visible_facts[name]
            unresolved.append(name)
            return ""

        resolved = _SLOT.sub(sub, template)
        if unresolved and resolved == "":
            continue
        out[key] = resolved
    if unresolved:
        return None, tuple(unresolved)
    return out, ()


def project_next_action(
    recipe: Recipe,
    *,
    step_index: int,
    visible_facts: Mapping[str, str],
    max_targets_per_decision: int = 1,
) -> tuple[Action | None, int, tuple[str, ...]]:
    """Project the next Student-native action from the recipe program.

    Returns (action | None, next_step_index, unresolved_refs).  With the
    default ``max_targets_per_decision=1`` exactly one action is emitted per
    call; subsequent steps wait for the corresponding observations to enter the
    Student history (which advances the step index at the next decision).
    """
    if step_index >= len(recipe.steps):
        return None, step_index, ()
    step = recipe.steps[step_index]
    resolved, unresolved = resolve_arguments(step, visible_facts)
    if resolved is None:
        return None, step_index, unresolved
    action: Action = ToolCall(name=step.tool, arguments=resolved)
    n = max(1, int(max_targets_per_decision))
    return action, step_index + min(n, len(recipe.steps) - step_index), ()
