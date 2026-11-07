"""Teacher readiness artifacts and the formal-training gate (guide §5.1/PR-05).

A bank is NOT a teacher merely because an evolve run produced it.  Formal OPD
training requires a ``teacher_readiness`` artifact binding the confirmed ΔH
evidence to the EXACT identities used at training time:

- bank_digest, policy_digest (base + adapter), runtime_contract_digest,
  environment_digest, metric_contract_digest, confirm_manifest_digest;
- the confirm result reference and its paired delta/interval;
- an alignment report reference (same-state teacher proposal coverage).

Any identity change invalidates the artifact (status falls back to
UNMEASURED), so stale confirmations can never authorize new training runs.
``require_teacher_readiness: true`` in the training config makes
``run_train`` refuse to start without a READY artifact covering the current
identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.common import JSONValue, digest_of

READINESS_SCHEMA = "teacher_readiness_v1"

# Ordered lifecycle states (guide §5.1).  Only the two READY_* states
# authorize formal training.
READY_STATES = ("READY_FOR_PILOT", "READY_FOR_FORMAL")
TERMINAL_NEGATIVE = ("NO_CONFIRMED_GAIN", "ALIGNMENT_UNSUPPORTED", "RUNTIME_INVALID")

VALID_STATES = (
    "UNMEASURED",
    "RUNTIME_INVALID",
    "NO_CONFIRMED_GAIN",
    "PARTIAL_SIGNAL_ONLY",
    "GAIN_CONFIRMED_ALIGNMENT_PENDING",
    "ALIGNMENT_UNSUPPORTED",
    "READY_FOR_PILOT",
    "READY_FOR_FORMAL",
)


@dataclass(frozen=True)
class TeacherReadiness:
    domain: str
    status: str
    code_commit: str
    bank_digest: str
    policy_digest: str
    runtime_contract_digest: str
    environment_digest: str
    metric_contract_digest: str
    confirm_manifest_digest: str
    confirm_result_ref: str
    paired_delta: float | None = None
    paired_interval: tuple[float, float] | None = None
    alignment_report_ref: str | None = None
    validator_version: str = "environment_registry_v1"
    extra: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATES:
            raise ValueError(
                f"unknown readiness status {self.status!r} (valid: {VALID_STATES})"
            )

    @property
    def ready(self) -> bool:
        return self.status in READY_STATES

    def identity_digest(self) -> str:
        """Digest over the identities the artifact is bound to."""
        return digest_of(
            {
                "domain": self.domain,
                "bank_digest": self.bank_digest,
                "policy_digest": self.policy_digest,
                "runtime_contract_digest": self.runtime_contract_digest,
                "environment_digest": self.environment_digest,
                "metric_contract_digest": self.metric_contract_digest,
                "confirm_manifest_digest": self.confirm_manifest_digest,
            }
        )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "schema_version": READINESS_SCHEMA,
            "domain": self.domain,
            "status": self.status,
            "code_commit": self.code_commit,
            "bank_digest": self.bank_digest,
            "policy_digest": self.policy_digest,
            "runtime_contract_digest": self.runtime_contract_digest,
            "environment_digest": self.environment_digest,
            "metric_contract_digest": self.metric_contract_digest,
            "confirm_manifest_digest": self.confirm_manifest_digest,
            "confirm_result_ref": self.confirm_result_ref,
            "paired_delta": self.paired_delta,
            "paired_interval": list(self.paired_interval) if self.paired_interval else None,
            "alignment_report_ref": self.alignment_report_ref,
            "validator_version": self.validator_version,
            "identity_digest": self.identity_digest(),
            "extra": dict(self.extra),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
        return p


def load_readiness(path: str | Path) -> TeacherReadiness:
    raw = json.loads(Path(path).read_text("utf-8"))
    if raw.get("schema_version") != READINESS_SCHEMA:
        raise ValueError(
            f"readiness schema {raw.get('schema_version')!r} != {READINESS_SCHEMA!r}"
        )
    interval = raw.get("paired_interval")
    return TeacherReadiness(
        domain=str(raw["domain"]),
        status=str(raw["status"]),
        code_commit=str(raw.get("code_commit", "")),
        bank_digest=str(raw.get("bank_digest", "")),
        policy_digest=str(raw.get("policy_digest", "")),
        runtime_contract_digest=str(raw.get("runtime_contract_digest", "")),
        environment_digest=str(raw.get("environment_digest", "")),
        metric_contract_digest=str(raw.get("metric_contract_digest", "")),
        confirm_manifest_digest=str(raw.get("confirm_manifest_digest", "")),
        confirm_result_ref=str(raw.get("confirm_result_ref", "")),
        paired_delta=(float(raw["paired_delta"]) if raw.get("paired_delta") is not None else None),
        paired_interval=(tuple(float(x) for x in interval) if interval else None),
        alignment_report_ref=raw.get("alignment_report_ref"),
        validator_version=str(raw.get("validator_version", "environment_registry_v1")),
        extra=dict(raw.get("extra") or {}),
    )


def check_readiness_for_training(
    *,
    readiness: TeacherReadiness | None,
    bank_digest: str,
    policy_digest: str,
    runtime_contract_digest: str,
    environment_digest: str,
) -> list[str]:
    """Return the list of BLOCKING deficiencies for a formal training run.

    Empty list = training may start.  Identity mismatches invalidate the
    artifact even when its status was READY — a stale confirmation must never
    authorize a run under different identities (guide §5.1).
    """
    problems: list[str] = []
    if readiness is None:
        return ["no teacher_readiness artifact found (require_teacher_readiness=true)"]
    if readiness.status not in READY_STATES:
        problems.append(
            f"readiness status {readiness.status!r} is not a READY state; formal "
            "OPD training requires READY_FOR_PILOT or READY_FOR_FORMAL"
        )
    for field_name, actual, bound in (
        ("bank_digest", bank_digest, readiness.bank_digest),
        ("policy_digest", policy_digest, readiness.policy_digest),
        ("runtime_contract_digest", runtime_contract_digest, readiness.runtime_contract_digest),
        ("environment_digest", environment_digest, readiness.environment_digest),
    ):
        if not bound:
            problems.append(f"readiness binds an EMPTY {field_name} — artifact incomplete")
        elif actual != bound:
            problems.append(
                f"{field_name} changed since the readiness artifact was issued "
                f"({actual[:12]} != bound {bound[:12]}) — artifact invalidated"
            )
    if readiness.paired_delta is None and readiness.status in READY_STATES:
        problems.append("READY artifact carries no paired_delta — confirmation evidence missing")
    return problems
