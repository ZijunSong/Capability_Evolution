"""Typed shadow harness modules for SCOPE."""

from harness.shadow.action_realizer import ActionRealizer, CandidateAction, realize
from harness.shadow.base import ShadowModule
from harness.shadow.budget_shadow import BudgetShadow
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.shadow.registry import ShadowRegistry, build_default_registry
from harness.shadow.verification_shadow import VerificationShadow

__all__ = [
    "ActionRealizer",
    "BudgetShadow",
    "CandidateAction",
    "EvidenceShadow",
    "ShadowModule",
    "ShadowRegistry",
    "VerificationShadow",
    "build_default_registry",
    "realize",
]
