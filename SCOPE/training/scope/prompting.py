"""Prompt formatting for action-level SDI."""

from __future__ import annotations

from typing import Any

from harness.capability.adapters import render_capability_action
from harness.capability.action_space import CapabilityAction
from training.scope.compact_target import CompactDupTarget, compact_target_from_sample, render_compact_target


def format_sdi_prompt(student_state_text: str) -> str:
    ctx = (student_state_text or "").strip()
    return f"{ctx}\n\nNext action (tool JSON):\n"


def format_compact_prompt(student_state_text: str) -> str:
    ctx = (student_state_text or "").strip()
    return f"{ctx}\n\nDuplicate decision (JSON operation):\n"


def format_operation_prompt(
    student_state_text: str,
    *,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Restricted verbalizer prompt for operation_ce (Round 3+).

  Includes per-candidate context so bilateral labels at the same turn remain
  distinguishable in the student-visible prompt.
    """
    ctx = (student_state_text or "").strip()
    parts = [ctx]
    if candidate_id:
        parts.append(f"\nCandidate evidence under review: {candidate_id}")
    if curated_document_ids:
        shown = list(curated_document_ids)[:32]
        suffix = "..." if len(curated_document_ids) > len(shown) else ""
        parts.append(
            "Currently curated evidence IDs: "
            + ", ".join(str(x) for x in shown)
            + suffix
        )
    parts.append("\nEvidence admission operation:")
    return "\n".join(parts)


def format_operation_prompt_from_sample(sample: dict[str, Any]) -> str:
    """Build operation prompt from a training sample (state + candidate context)."""
    state_text = str(
        sample.get("student_state_text")
        or (sample.get("decision_state") or {}).get("rendered_context")
        or ""
    )
    compact = compact_target_from_sample(sample)
    ds = sample.get("decision_state") or {}
    curated = ds.get("curated_document_ids") or ds.get("curated_evidence_ids") or []
    return format_operation_prompt(
        state_text,
        candidate_id=compact.candidate_id if compact else None,
        curated_document_ids=curated,
    )


def format_sdi_example(
    student_state_text: str,
    target_action: dict,
) -> tuple[str, str]:
    meta_fmt = None
    if isinstance(target_action, dict) and "operation" in target_action:
        compact = CompactDupTarget.from_dict(target_action)
        prompt = format_compact_prompt(student_state_text)
        return prompt, render_compact_target(compact)
    prompt = format_sdi_prompt(student_state_text)
    action = CapabilityAction.from_dict(target_action)
    target = render_capability_action(action)
    return prompt, target
