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
from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.events import Event
from rsi.contracts.observation import ToolSpec
from rsi.contracts.task import PublicTask
from rsi.harness.component import PublicView
from rsi.runtime.profiles import H0Profile

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


def render_header(
    task: PublicTask,
    catalog: tuple[ToolSpec, ...],
    extra_instructions: str | None = None,
    profile: H0Profile | None = None,
) -> str:
    """Fixed ordering (guide §5.2): profile → environment doc → task → tools
    → protocol → bank guidance.  The bank may ADD guidance after the fixed
    base but never reorders or overwrites it."""
    parts: list[str] = []
    if profile is not None:
        parts.append(profile.instruction_text)
        if profile.environment_doc_text:
            parts.append("Environment interface:\n" + profile.environment_doc_text)
    parts.append(f"Task ({task.domain}): {task.instruction}")
    parts.append("Tools:")
    parts.append(render_tool_catalog(catalog))
    parts.append(ACTION_PROTOCOL)
    if extra_instructions:
        parts.append("Additional guidance:\n" + extra_instructions)
    return "\n".join(parts)


def truncate_head_tail(text: str, *, max_chars: int) -> str:
    """Deterministic head-tail truncation for an oversized observation
    (guide §5.4): the newest feedback must stay visible; a silent empty
    history is a measurement bug.  Marks the elision with the original and
    omitted lengths."""
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    # 60% head / 40% tail keeps both the action taken and its result tail.
    head = int(max_chars * 0.6)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n[... TRUNCATED {omitted} chars of {len(text)} total (head-tail transport truncation) ...]\n"
        + text[len(text) - tail:]
    )


def render_view(
    view: PublicView,
    *,
    max_chars: int = 8000,
    profile: H0Profile | None = None,
) -> str:
    """Deterministic rendering with oldest-first history dropping AND a
    guaranteed newest-observation window (guide §5.4): the latest event is
    always shown, head-tail truncated if it alone exceeds the budget; the
    remaining budget fills older history newest-first.  No LLM summarization;
    the raw event log is untouched."""
    extra = dict(view.extra)
    appended = extra.get("appended_instructions")
    header = render_header(
        view.task,
        view.tool_catalog,
        appended if isinstance(appended, str) else None,
        profile=profile,
    )
    budget = max_chars - len(header) - 64
    if budget < 0:
        # guide §5.4 item 1: header over budget is an explicit error, not a
        # silent zero-history render.
        raise ProtocolViolation(
            f"render header ({len(header)} chars) exceeds the context window "
            f"({max_chars} chars) — reduce the environment doc/tool catalog or "
            "raise runtime.max_chars; refusing to render an empty history"
        )
    ordered = sorted(view.history, key=lambda e: e.order, reverse=True)
    body_parts: list[str] = []
    used = 0
    for idx, event in enumerate(ordered):
        text = render_event(event)
        if idx == 0 and len(text) + 2 > budget:
            # newest event alone exceeds the budget: show it truncated, no
            # older history.
            text = truncate_head_tail(text, max_chars=budget)
            body_parts.append(text)
            used = len(text) + 2
            break
        if used + len(text) + 2 > budget:
            break
        body_parts.append(text)
        used += len(text) + 2
    body_parts.reverse()
    body = "\n".join(body_parts)
    prompt = f"{header}\n\nHistory:\n{body}\n\nNext action:"
    return prompt


def view_digest(view: PublicView, *, policy: ContextPolicy | None = None) -> str:
    """Digest the ACTUAL rendered prompt (guide §6.1): the old default-8000
    re-render disagreed with a policy configured at a different window."""
    if policy is not None:
        return digest_of_text(policy.render(view))
    return digest_of_text(render_view(view))


def prompt_hash(policy: ContextPolicy | None = None) -> str:
    """Protocol identity over renderer + action schema + window + profile
    text (guide §6.1): changing the profile or the window changes the hash,
    so cached baselines never silently cross protocol versions."""
    parts = [RENDERER_VERSION, ACTION_PROTOCOL, CONTEXT_POLICY_ID]
    if policy is not None:
        parts.append(f"max_chars={policy.max_chars}")
        if policy.profile is not None:
            parts.append(f"profile_digest={policy.profile.digest()}")
    return digest_of_text("|".join(parts))


@dataclass(frozen=True)
class ContextPolicy:
    """Renderer + window + profile in one identity (guide §5.2/§6.1): the
    profile is part of the protocol identity, so a policy without one is a
    legacy object kept only for old tests; new campaign paths must build it
    through runtime.factory, which always attaches the config's profile."""

    policy_id: str = CONTEXT_POLICY_ID
    max_chars: int = 8000
    profile: H0Profile | None = None

    def render(self, view: PublicView) -> str:
        return render_view(view, max_chars=self.max_chars, profile=self.profile)
