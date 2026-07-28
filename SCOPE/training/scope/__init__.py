"""SCOPE v3 package: verified decision routing + action-level SDI."""

from training.scope.schema import (
    BranchType,
    DecisionSupervisionSampleV3,
    GateFlags,
    Route,
    VerificationFlags,
    WeightTerms,
)
from training.scope.routing import RoutingResult, route_decision
from training.scope.losses import SDILossConfig, compute_sdi_loss, action_span_labels
from training.scope.weighting import WeightingConfig, compute_weight_terms
from training.scope.capability_stats import CapabilityStatsAggregator

__all__ = [
    "BranchType",
    "CapabilityStatsAggregator",
    "DecisionSupervisionSampleV3",
    "GateFlags",
    "Route",
    "RoutingResult",
    "SDILossConfig",
    "VerificationFlags",
    "WeightTerms",
    "WeightingConfig",
    "action_span_labels",
    "compute_sdi_loss",
    "compute_weight_terms",
    "route_decision",
]
