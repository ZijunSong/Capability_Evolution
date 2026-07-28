"""SCOPE dual-mode OPD v2 package."""

from training.opd_v2.transitions import OPDTransitionV2
from training.opd_v2.router import GuidanceDecision, GuidanceRouter

__all__ = ["GuidanceDecision", "GuidanceRouter", "OPDTransitionV2"]
