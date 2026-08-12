"""Adapter registry for Round14 capabilities."""

from __future__ import annotations

from training.scope_round14.adapters.base import CapabilityAdapter
from training.scope_round14.adapters.c0_duplicate_evidence import DuplicateEvidenceAdapter
from training.scope_round14.adapters.c1_stop_decision import StopDecisionAdapter
from training.scope_round14.adapters.c2_verification_routing import VerificationRoutingAdapter
from training.scope_round14.adapters.c3_evidence_admission import EvidenceAdmissionAdapter
from training.scope_round14.adapters.c4_context_budget_routing import ContextBudgetRoutingAdapter
from training.scope_round14.adapters.c5_external_verification_routing import (
  ExternalVerificationRoutingAdapter,
)
from training.scope_round14.adapters.c6_rollback_lite import RollbackLiteAdapter

_ADAPTERS: dict[str, type[CapabilityAdapter]] = {
  "duplicate_evidence": DuplicateEvidenceAdapter,
  "stop_decision": StopDecisionAdapter,
  "verification_routing": VerificationRoutingAdapter,
  "evidence_admission": EvidenceAdmissionAdapter,
  "context_budget_routing": ContextBudgetRoutingAdapter,
  "external_verification_routing": ExternalVerificationRoutingAdapter,
  "rollback_lite": RollbackLiteAdapter,
}


def get_adapter(capability: str) -> CapabilityAdapter:
  key = capability.strip().lower()
  if key not in _ADAPTERS:
    raise KeyError(f"unknown capability adapter: {capability}")
  return _ADAPTERS[key]()


def list_adapters() -> list[str]:
  return sorted(_ADAPTERS.keys())
