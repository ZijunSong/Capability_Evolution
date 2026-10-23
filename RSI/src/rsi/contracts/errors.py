"""Error taxonomy.

The four runtime failure classes required by the task book
(candidate_invalid / component_runtime_error / environment_error /
model_invalid_action) plus infra and protocol classes used for retry policy
and reporting.  Every runtime failure path must map to one of these.
"""

from __future__ import annotations


class RSIError(Exception):
    """Base class for all RSI errors."""

    kind: str = "rsi_error"


class CandidateInvalid(RSIError):
    """The candidate patch/manifest is structurally or semantically invalid."""

    kind = "candidate_invalid"


class ComponentRuntimeError(RSIError):
    """A harness component raised while executing."""

    kind = "component_runtime_error"


class EnvironmentError_(RSIError):
    """The environment adapter failed for an environmental reason.

    Named with a trailing underscore to avoid shadowing the builtin.
    """

    kind = "environment_error"


class ModelInvalidAction(RSIError):
    """The model produced an unparsable or schema-illegal action."""

    kind = "model_invalid_action"


class InfraError(RSIError):
    """Network / process / platform failure that pre-registered retry policy may retry."""

    kind = "infra_error"


class ProtocolViolation(RSIError):
    """An immutable boundary was violated (hidden grader access, split leakage...)."""

    kind = "protocol_violation"


class BudgetExceeded(RSIError):
    """A hard budget bound was reached."""

    kind = "budget_exceeded"


class NotImplementedCapability(RSIError):
    """A required capability is unavailable; callers must fail loudly, not fallback."""

    kind = "not_implemented"


def classify(exc: BaseException) -> str:
    """Map an exception to its taxonomy kind string."""
    if isinstance(exc, RSIError):
        return exc.kind
    return "unclassified"
