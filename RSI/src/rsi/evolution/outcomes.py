"""Unified task-outcome semantics (guide §4.1 / PR-03).

ONE classification shared by evolution/service.analysis_runner,
evolution/search._fragments_from_per_task and evolution/diagnosis.

The old three sites each guessed failure from ``value`` alone:

- analysis_runner + _fragments_from_per_task: ``value < 1e-9`` → failure.  A
  task with partial credit 0.4 and ``success=False`` was fed to the proposer
  as a SUCCESS case.
- diagnosis: ``value < 1.0`` → failure.  Same record classified differently
  by a sibling module.

The unified rule requires an EXPLICIT strict-success label:

- measurement must be VALID (EVAL_INVALID is never an outcome — it blocks);
- ``success=True`` → SUCCESS;
- ``success=False`` with ``value > 0`` → PARTIAL_FAILURE (usable for reward
  signal and progress telemetry, NEVER as a success case);
- ``success=False`` with ``value == 0`` → FAILURE.

``component_errors`` stay a separate reliability field: a task that happened
to succeed while a component errored is still a task success with a
component-reliability problem — both facts are preserved, neither overwrites
the other.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from rsi.contracts.common import JSONValue


class Outcome(str, Enum):
    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    FAILURE = "failure"
    INVALID = "invalid"


# Back-compat alias for callers that want the raw label strings.
OUTCOME_VALUES = tuple(o.value for o in Outcome)


def classify_outcome(meta: Mapping[str, Any]) -> Outcome:
    """Classify one per-task evaluation record.

    ``meta`` accepts either a raw ScoreRecord-style mapping (``details`` holds
    ``measurement_status``) or a flattened per-task dict.  Missing strict
    ``success`` raises — silently defaulting to success/failure is exactly the
    bug this module removes.
    """
    status = _measurement_status(meta)
    if status is not None and str(status).upper() != "VALID":
        return Outcome.INVALID
    success = meta.get("success")
    if not isinstance(success, bool):
        raise ValueError(
            "missing strict success label (bool `success`) — refusing to "
            "guess the outcome from value alone"
        )
    if success:
        return Outcome.SUCCESS
    try:
        value = float(meta.get("value", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return Outcome.PARTIAL_FAILURE if value > 0.0 else Outcome.FAILURE


def _measurement_status(meta: Mapping[str, Any]) -> str | None:
    status = meta.get("measurement_status")
    if status is not None:
        return str(status)
    details = meta.get("details")
    if isinstance(details, Mapping):
        status = details.get("measurement_status")
        if status is not None:
            return str(status)
    return None


def is_failure_outcome(outcome: Outcome) -> bool:
    """Failure cases for the packet: partial failures AND full failures (an
    INVALID measurement is neither — it is evidence of a broken evaluation,
    reported separately and excluded from method comparisons)."""
    return outcome in (Outcome.PARTIAL_FAILURE, Outcome.FAILURE)


def outcome_counts(per_task: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = {o.value: 0 for o in Outcome}
    invalid_ids: list[str] = []
    for tid, meta in per_task.items():
        try:
            counts[classify_outcome(meta).value] += 1
        except ValueError:
            counts[Outcome.INVALID.value] += 1
            invalid_ids.append(tid)
    counts["_invalid_task_ids"] = invalid_ids  # type: ignore[assignment]
    return counts


def metric_value(value: Any, *, metric: str, domain: str | None = None, metric_name: str | None = None) -> float:
    """Explicit metric routing (guide §4.2 / PR-03).

    Selection/confirm previously read ``ScoreRecord.value`` blindly — for
    AppWorld that is the PARTIAL rate even when the config declared
    ``scoring.primary=appworld_strict_success``, so the confirm gate was not
    measuring what its config claimed.

    - ``strict_success``  → the boolean strict-success label as 0/1;
    - ``partial_test_pass_rate`` → asserts the record actually is that metric;
    - ``webshop_reward``  → asserts the record is from webshop (its value IS
      the reward);
    anything else raises instead of silently returning ``value``.
    """
    if metric == "strict_success":
        s = value.get("success") if isinstance(value, Mapping) else None
        if isinstance(value, Mapping) and not isinstance(s, bool):
            raise ValueError("strict_success requires a bool `success` label")
        return 1.0 if s else 0.0
    if metric == "partial_test_pass_rate":
        name = (metric_name or (value.get("metric_name") if isinstance(value, Mapping) else None) or "")
        if name != "appworld_partial_test_pass_rate":
            raise ValueError(
                f"partial_test_pass_rate requested but record metric is {name!r}"
            )
        return float(value.get("value", 0.0)) if isinstance(value, Mapping) else float(value)
    if metric == "webshop_reward":
        dom = domain or (value.get("domain") if isinstance(value, Mapping) else None)
        if dom != "webshop":
            raise ValueError(f"webshop_reward requested but record domain is {dom!r}")
        return float(value.get("value", 0.0)) if isinstance(value, Mapping) else float(value)
    raise ValueError(f"unknown metric {metric!r}")


def invalid_measurement_blocked(meta: Mapping[str, Any]) -> bool:
    """True when the record must NOT enter candidate comparison or training
    reward (EVAL_INVALID: missing output DB, scorer exception, ...)."""
    status = _measurement_status(meta)
    return status is not None and str(status).upper() in ("EVAL_INVALID", "INVALID")
