"""Evidence-window extraction for the proposer packet (guide §4.3 / PR-04).

Replaces "first 8 steps, re-truncated to 500 chars" with CLASSIFIED failure
windows: for each failed task, pick the evidence that matches its failure
mechanism and keep the exact tool call + its feedback together.  The window
carriers event refs so the proposer's hypotheses can cite real evidence
(``failure_evidence_refs``), and the analysis runner can reload full
traces by ref instead of trusting truncated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue
from rsi.evolution.outcomes import Outcome, classify_outcome, invalid_measurement_blocked


@dataclass(frozen=True)
class FailureWindow:
    task_id: str
    failure_type: str  # invalid_action | api_error | no_output | no_progress | budget
    evidence_text: str
    event_ids: tuple[str, ...] = ()
    n_decisions: int = 0

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "task_id": self.task_id,
            "failure_type": self.failure_type,
            "evidence_text": self.evidence_text,
            "event_ids": list(self.event_ids),
            "n_decisions": self.n_decisions,
        }


def _classify_step(step: Mapping[str, Any]) -> str | None:
    """Classify one step's failure mechanism from its bounded telemetry."""
    act = str(step.get("action") or "")
    obs = str(step.get("obs_head") or "")
    obs_err = step.get("obs_ok", True) is False
    kind = str(step.get("kind") or "")
    # explicit error kinds (flag or event kind)
    if kind == "error" or obs_err:
        if "invalid action format" in obs or "model_invalid_action" in obs:
            return "invalid_action"
        return "api_error"
    # vendor-embedded failure strings inside a nominally-ok observation
    # (AppWorld returns "Execution failed. Traceback: …" WITHOUT raising —
    # the adapter now flags ok=False, but panel evidence may predate that)
    if "Execution failed" in obs or "EXEC_ERROR" in obs or "Traceback" in obs:
        return "api_error"
    if act and not obs.strip():
        return "no_output"
    return None


def failure_windows_from_steps(
    *,
    task_id: str,
    steps: Sequence[Mapping[str, Any]],
    terminated_reason: str,
    value: float,
    success: bool,
    measurement_status: str | None = None,
    max_windows: int = 3,
    max_chars: int = 600,
) -> tuple[FailureWindow, ...]:
    """Extract classified failure windows from a task's bounded step list.

    Only PARTIAL_FAILURE / FAILURE tasks contribute (never successes, never
    INVALID measurements).  Each window keeps one error step PLUS the step
    right before it (the action that led to the failure) — a window that
    shows only the error cannot distinguish a bad action from a bad env.
    """
    meta = {"value": value, "success": success, "measurement_status": measurement_status}
    if invalid_measurement_blocked(meta):
        return ()
    try:
        outcome = classify_outcome(meta)
    except ValueError:
        return ()
    if outcome is Outcome.SUCCESS:
        return ()

    windows: list[FailureWindow] = []
    seen_types: set[str] = set()
    for i, step in enumerate(steps):
        ftype = _classify_step(step)
        if ftype is None or ftype in seen_types:
            continue
        seen_types.add(ftype)
        prev = steps[i - 1] if i > 0 else {}
        prev_text = ""
        if prev:
            prev_text = (
                f"prev[{prev.get('order', '?')}]: {str(prev.get('action') or '')[:200]}"
                f" -> {str(prev.get('obs_head') or '')[:160]}\n"
            )
        err_text = (
            f"err[{step.get('order', '?')}]: {str(step.get('action') or '')[:200]}"
            f" -> {str(step.get('obs_head') or '')[:240]}"
        )
        text = (
            f"terminated={terminated_reason} value={value} failure_type={ftype}\n"
            f"{prev_text}{err_text}"
        )[:max_chars]
        eid = str(step.get("event_id") or step.get("order") or "")
        windows.append(
            FailureWindow(
                task_id=task_id,
                failure_type=ftype,
                evidence_text=text,
                event_ids=(eid,) if eid else (),
                n_decisions=len(steps),
            )
        )
        if len(windows) >= max_windows:
            break
    # budget/no-progress termination without a per-step error still deserves
    # a window (guide table: budget wall is UNKNOWN_NEEDS_TRACE, not a hint)
    if not windows and terminated_reason in ("budget_wall", "max_actions"):
        text = f"terminated={terminated_reason} value={value} n_steps={len(steps)}"[:max_chars]
        windows.append(
            FailureWindow(task_id=task_id, failure_type="budget", evidence_text=text,
                          n_decisions=len(steps))
        )
    return tuple(windows)


# Proposer proposal-required fields (guide §4.3).  A hypothesis that only
# says "be more careful" cannot cite evidence — the packet declares the
# required schema and the validator (evolution/validator.py) can enforce it.
REQUIRED_PROPOSAL_FIELDS = (
    "failure_evidence_refs",
    "hypothesis",
    "expected_trigger",
    "expected_change",
    "possible_regression",
    "extra_budget",
)

# Generic "advice" phrases that fail the evidence contract (guide §4.3: a
# proposal whose content is only "更仔细/必要时反思" is not a mechanism).
GENERIC_HYPOTHESIS_PHRASES = (
    "be more careful",
    "更仔细",
    "pay attention",
    "as needed",
    "必要时反思",
    "reflect when needed",
    "think harder",
    "try again",
    "improve the process",
)


def hypothesis_quality_problems(patch_hypothesis: str, evidence_refs: Sequence[str]) -> tuple[str, ...]:
    """Return problems for a proposal that violates the PR-04 evidence
    contract.  Empty tuple = acceptable.

    Rules (guide §4.3): the hypothesis must NAME a mechanism (non-trivial
    text) and cite failure evidence refs; purely generic-advice phrasing is
    rejected.
    """
    problems: list[str] = []
    hyp = str(patch_hypothesis or "").strip()
    if len(hyp) < 24:
        problems.append("hypothesis too short to name a mechanism")
    lowered = hyp.lower()
    if any(p.lower() in lowered for p in GENERIC_HYPOTHESIS_PHRASES) and len(hyp) < 80:
        problems.append("hypothesis is generic advice without a mechanism")
    if not evidence_refs:
        problems.append("no failure_evidence_refs cited (guide §4.3)")
    return tuple(problems)
