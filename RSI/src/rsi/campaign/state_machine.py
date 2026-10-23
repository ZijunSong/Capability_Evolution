"""Campaign state machine (task book §10.1).

    PREFLIGHT -> BASELINE -> EVOLVE -> CONFIRM_HARNESS -> PROBE_OPD
    -> TRAIN -> SELECT_CHECKPOINT -> CONFIRM_MODEL -> COMMIT_PROMOTION
    -> BASELINE(next round) ... -> FREEZE_CAMPAIGN -> FINAL_EVAL

Every stage can end in an explicit failure/insufficient-evidence state; the
machine is not "loop N times then declare success".
"""

from __future__ import annotations

from typing import Any

MAIN_STATES = (
    "PREFLIGHT",
    "BASELINE",
    "EVOLVE",
    "CONFIRM_HARNESS",
    "PROBE_OPD",
    "TRAIN",
    "SELECT_CHECKPOINT",
    "CONFIRM_MODEL",
    "COMMIT_PROMOTION",
    "FREEZE_CAMPAIGN",
    "FINAL_EVAL",
    "DONE",
)

STOP_STATES = (
    "NO_CONFIRMED_HARNESS_GAIN",
    "INSUFFICIENT_OPD_SUPPORT",
    "MODEL_NOT_PROMOTED",
    "NO_ADMISSIBLE_TEACHER",
    "CANDIDATES_REPEATEDLY_INVALID",
    "BUDGET_EXHAUSTED",
    "PREFLIGHT_FAILED",
    "ERROR",
)

TRANSITIONS = {
    "PREFLIGHT": ("BASELINE", "PREFLIGHT_FAILED", "ERROR"),
    "BASELINE": ("EVOLVE", "BUDGET_EXHAUSTED", "ERROR"),
    "EVOLVE": ("CONFIRM_HARNESS", "NO_ADMISSIBLE_TEACHER", "CANDIDATES_REPEATEDLY_INVALID", "BUDGET_EXHAUSTED", "ERROR"),
    "CONFIRM_HARNESS": ("PROBE_OPD", "NO_CONFIRMED_HARNESS_GAIN", "ERROR"),
    "PROBE_OPD": ("TRAIN", "INSUFFICIENT_OPD_SUPPORT", "ERROR"),
    "TRAIN": ("SELECT_CHECKPOINT", "BUDGET_EXHAUSTED", "ERROR"),
    "SELECT_CHECKPOINT": ("CONFIRM_MODEL", "ERROR"),
    "CONFIRM_MODEL": ("COMMIT_PROMOTION", "MODEL_NOT_PROMOTED", "ERROR"),
    "COMMIT_PROMOTION": ("BASELINE", "FREEZE_CAMPAIGN", "ERROR"),
    "FREEZE_CAMPAIGN": ("FINAL_EVAL", "ERROR"),
    "FINAL_EVAL": ("DONE", "ERROR"),
}


class StateMachine:
    def __init__(self, state: str = "PREFLIGHT", history: list[dict[str, Any]] | None = None):
        if state not in MAIN_STATES + STOP_STATES:
            raise ValueError(f"unknown state {state!r}")
        self.state = state
        self.history: list[dict[str, Any]] = list(history or [])

    def transition(self, to: str, *, reason: str = "", evidence: dict[str, Any] | None = None) -> None:
        if self.state in STOP_STATES or self.state == "DONE":
            raise ValueError(f"campaign already terminal in {self.state!r}")
        allowed = TRANSITIONS.get(self.state, ())
        if to not in allowed:
            raise ValueError(f"illegal transition {self.state!r} -> {to!r}")
        self.history.append({"from": self.state, "to": to, "reason": reason, "evidence": dict(evidence or {})})
        self.state = to

    @property
    def is_terminal(self) -> bool:
        return self.state in STOP_STATES or self.state == "DONE"

    def to_json(self) -> dict[str, Any]:
        return {"state": self.state, "history": self.history}
