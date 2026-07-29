"""Prompt formatting for action-level SDI."""

from __future__ import annotations

from harness.capability.adapters import render_capability_action
from harness.capability.action_space import CapabilityAction
from training.scope.compact_target import CompactDupTarget, render_compact_target


def format_sdi_prompt(student_state_text: str) -> str:
    ctx = (student_state_text or "").strip()
    return f"{ctx}\n\nNext action (tool JSON):\n"


def format_compact_prompt(student_state_text: str) -> str:
    ctx = (student_state_text or "").strip()
    return f"{ctx}\n\nDuplicate decision (JSON operation):\n"


def format_operation_prompt(student_state_text: str) -> str:
    """Restricted verbalizer prompt for operation_ce (Round 3)."""
    ctx = (student_state_text or "").strip()
    return f"{ctx}\n\nEvidence admission operation:\n"


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
