"""View rendering and deterministic context policy.

``deterministic_window_v1``: render the fixed header (task, tool schemas,
protocol) plus as much of the legal history as fits the character budget,
dropping *oldest* events first.  The raw event log is never modified; only the
rendered window shrinks.  No LLM summarization may stand in for H0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rsi.contracts.action import encode_action
from rsi.contracts.common import digest_of_text
from rsi.contracts.events import Event
from rsi.contracts.observation import ToolSpec
from rsi.contracts.task import PublicTask
from rsi.harness.component import PublicView

CONTEXT_POLICY_ID = "deterministic_window_v1"
RENDERER_VERSION = "rsi_render_v1"

ACTION_PROTOCOL = (
    "Respond with exactly one JSON object per turn, either "
    '{"type": "tool_call", "name": "<tool>", "arguments": {...}} or '
    '{"type": "finish", "answer": "<final answer or null>", "artifact_refs": []}.'
)


def render_tool_catalog(catalog: tuple[ToolSpec, ...]) -> str:
    lines = []
    for t in catalog:
        lines.append(f"- {t.name}: {t.description} params={json.dumps(dict(t.parameters_schema), sort_keys=True)}")
    return "\n".join(lines)


def render_event(event: Event) -> str:
    if event.action is not None and event.observation is not None:
        return f"[{event.order}] action: {encode_action(event.action)}\n[{event.order}] observation: {event.observation.content}"
    if event.action is not None:
        return f"[{event.order}] action: {encode_action(event.action)}"
    if event.observation is not None:
        return f"[{event.order}] observation: {event.observation.content}"
    return f"[{event.order}] {event.event_kind.value}: {json.dumps(dict(event.data), sort_keys=True)}"


def render_header(task: PublicTask, catalog: tuple[ToolSpec, ...], extra_instructions: str | None = None) -> str:
    parts = [
        f"Task ({task.domain}): {task.instruction}",
        "Tools:",
        render_tool_catalog(catalog),
        ACTION_PROTOCOL,
    ]
    if extra_instructions:
        parts.append("Additional guidance:\n" + extra_instructions)
    return "\n".join(parts)


def render_view(view: PublicView, *, max_chars: int = 8000) -> str:
    """Deterministic rendering with oldest-first history dropping."""
    extra = dict(view.extra)
    appended = extra.get("appended_instructions")
    header = render_header(view.task, view.tool_catalog, appended if isinstance(appended, str) else None)
    budget = max_chars - len(header) - 64
    if budget < 0:
        budget = 0
    kept: list[str] = []
    used = 0
    for event in sorted(view.history, key=lambda e: e.order, reverse=True):
        text = render_event(event)
        if used + len(text) + 2 > budget:
            break
        kept.append(text)
        used += len(text) + 2
    kept.reverse()
    body = "\n".join(kept)
    prompt = f"{header}\n\nHistory:\n{body}\n\nNext action:"
    return prompt


def view_digest(view: PublicView) -> str:
    return digest_of_text(render_view(view))


def prompt_hash() -> str:
    return digest_of_text(RENDERER_VERSION + "|" + ACTION_PROTOCOL + "|" + CONTEXT_POLICY_ID)


@dataclass(frozen=True)
class ContextPolicy:
    policy_id: str = CONTEXT_POLICY_ID
    max_chars: int = 8000

    def render(self, view: PublicView) -> str:
        return render_view(view, max_chars=self.max_chars)
