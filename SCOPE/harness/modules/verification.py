"""M2: Verification module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

VERIFICATION_NODE_IDS = ("V1", "V2", "V3", "V4", "V5")

_NODE_OPTION_MAP = {
    "V1": "expose_verify_tool",
    "V2": "store_records",
    "V3": "render_records",
    "V4": "verification_aware_curation",
    "V5": "store_records",
}


@dataclass
class VerificationRecord:
    turn_id: int
    claim: str
    doc_ids: list[str]
    judgments: dict[str, bool]
    rationales: dict[str, str]
    source_observation_ids: list[str] | None = None

    def compact_text(self) -> str:
        parts = [f"claim: {self.claim}"]
        for doc_id in self.doc_ids:
            verdict = "yes" if self.judgments.get(doc_id) else "no"
            parts.append(f"  {doc_id}: {verdict}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "claim": self.claim,
            "doc_ids": list(self.doc_ids),
            "judgments": dict(self.judgments),
            "rationales": dict(self.rationales),
            "source_observation_ids": list(self.source_observation_ids or []),
        }

    @classmethod
    def from_verify_output(
        cls,
        turn_id: int,
        claim: str,
        doc_ids: list[str],
        raw_output: str,
    ) -> VerificationRecord:
        judgments: dict[str, bool] = {}
        rationales: dict[str, str] = {}
        current_doc: str | None = None
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("# DOCUMENT ID:"):
                current_doc = line.split(":", 1)[1].strip()
            elif line.startswith("verdict:") and current_doc:
                judgments[current_doc] = "yes" in line.lower()
            elif line.startswith("rationale:") and current_doc:
                rationales[current_doc] = line.split(":", 1)[1].strip()
        for doc_id in doc_ids:
            judgments.setdefault(doc_id, False)
            rationales.setdefault(doc_id, "")
        return cls(
            turn_id=turn_id,
            claim=claim,
            doc_ids=list(doc_ids),
            judgments=judgments,
            rationales=rationales,
        )


class _VerificationNode(HarnessNode):
    def __init__(self, node_id: str, *, enabled: bool) -> None:
        super().__init__(enabled=enabled)
        self.node_id = node_id
        self.module_id = "verification"

    def run(self, payload: Any, context: ExecutionContext) -> NodeResult:
        wm = context.working_memory
        if self.node_id == "V3" and wm is not None:
            records = getattr(wm, "verification_records", [])
            compact = "\n".join(r.compact_text() for r in records[-5:])
            return NodeResult(output=compact, changed_state=bool(records))
        return NodeResult(output=payload, changed_state=self.node_id in {"V2", "V4"})

    def fallback(self, payload: Any, context: ExecutionContext) -> NodeResult:
        return NodeResult(
            output="" if self.node_id == "V3" else payload,
            metadata={"fallback_used": True, "verify_unavailable": self.node_id == "V1"},
        )


def build_verification_module(config: ModuleConfig) -> HarnessModule:
    nodes: list[HarnessNode] = []
    for node_id in VERIFICATION_NODE_IDS:
        opt_key = _NODE_OPTION_MAP[node_id]
        enabled = config.enabled and config.options.get(opt_key, True)
        nodes.append(_VerificationNode(node_id, enabled=bool(enabled)))
    return HarnessModule(module_id="verification", nodes=nodes, config=config)
