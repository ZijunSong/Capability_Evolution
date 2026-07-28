"""Distillability probe stubs (P8) — non-blocking for Round 1 training.

Formal probe_distillability.py lives under training/; this module holds
shared enums / safe-by-construction purity markers used by gates & stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from harness.capability.capability_id import CapabilityId


class ProceduralPurity(str, Enum):
    """P_c: whether supervision is information-safe by construction."""

    SAFE_BY_CONSTRUCTION = "safe_by_construction"
    NEEDS_PROBE = "needs_probe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


# Round-1 defaults from Go25 audit conclusions
ROUND1_PURITY: dict[CapabilityId, ProceduralPurity] = {
    CapabilityId.DUPLICATE_EVIDENCE: ProceduralPurity.SAFE_BY_CONSTRUCTION,
    CapabilityId.PREMATURE_STOP: ProceduralPurity.SAFE_BY_CONSTRUCTION,
    CapabilityId.IRRELEVANT_EVIDENCE: ProceduralPurity.UNSAFE,  # circular GT
    CapabilityId.INVALID_CITATION: ProceduralPurity.NEEDS_PROBE,
}


@dataclass(frozen=True)
class DistillabilityScore:
    capability_id: CapabilityId
    procedural_purity: ProceduralPurity
    purity_score: float  # 1.0 = safe
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "capability_id": self.capability_id.value,
            "procedural_purity": self.procedural_purity.value,
            "purity_score": self.purity_score,
            "notes": self.notes,
        }


def round1_purity(capability: CapabilityId) -> DistillabilityScore:
    purity = ROUND1_PURITY.get(capability, ProceduralPurity.UNKNOWN)
    score = {
        ProceduralPurity.SAFE_BY_CONSTRUCTION: 1.0,
        ProceduralPurity.NEEDS_PROBE: 0.5,
        ProceduralPurity.UNSAFE: 0.0,
        ProceduralPurity.UNKNOWN: 0.0,
    }[purity]
    notes = ""
    if capability == CapabilityId.IRRELEVANT_EVIDENCE:
        notes = "circular labeling vs shadow; disabled for Round 1"
    return DistillabilityScore(
        capability_id=capability,
        procedural_purity=purity,
        purity_score=score,
        notes=notes,
    )
