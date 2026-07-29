"""ActionRealizer: map capability operation → executable runtime action.

Round 2 architecture:
    DecisionState → capability operation (KEEP_EVIDENCE / SKIP_DUPLICATE)
        → ActionRealizer → runtime executable action

Uses only student-visible state — no hidden teacher information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.artifacts.schema import PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import CapabilityId
from harness.capability.state import DecisionState
from harness.capability.dup_operation import DupOperation


@dataclass(frozen=True)
class CandidateAction:
    action: CapabilityAction
    source: str  # "operation_realized" | "artifact_recommended" | "deterministic"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "source": self.source,
            "notes": self.notes,
        }


_OP_ALIASES: dict[str, CapabilityActionType] = {
    "skip_curate": CapabilityActionType.CURATE_DOCUMENT,
    "replace_evidence": CapabilityActionType.CURATE_DOCUMENT,
    "continue_search": CapabilityActionType.CONTINUE_SEARCH,
    "verify": CapabilityActionType.VERIFY_CLAIM,
    "open_source": CapabilityActionType.OPEN_DOCUMENT,
    "stop_and_answer": CapabilityActionType.STOP_AND_ANSWER,
    "search": CapabilityActionType.SEARCH,
    "rewrite_query": CapabilityActionType.REWRITE_QUERY,
    "curate_document": CapabilityActionType.CURATE_DOCUMENT,
    "update_evidence": CapabilityActionType.UPDATE_EVIDENCE,
    "verify_claim": CapabilityActionType.VERIFY_CLAIM,
}


class ActionRealizer:
    """Deterministic operation → runtime action realizer for Round 2."""

    def realize_operation(
        self,
        state: DecisionState,
        operation: DupOperation | str,
        *,
        candidate_id: str | None = None,
        student_action: CapabilityAction | None = None,
    ) -> CandidateAction:
        """Map compact Dup operation to executable runtime action."""
        op = DupOperation(str(operation).upper()) if not isinstance(operation, DupOperation) else operation
        if op == DupOperation.SKIP_DUPLICATE:
            return self._skip_duplicate(state, candidate_id, student_action)
        return self._keep_evidence(state, candidate_id, student_action)

    def _candidate_add_ids(
        self,
        state: DecisionState,
        candidate_id: str | None,
        student_action: CapabilityAction | None,
    ) -> list[str]:
        if candidate_id and candidate_id in set(state.pool_document_ids):
            return [candidate_id]
        if student_action and student_action.action_type == CapabilityActionType.CURATE_DOCUMENT:
            adds = student_action.arguments.get("add_ids") or []
            return [str(d) for d in adds if str(d) in set(state.pool_document_ids)]
        return []

    def _skip_duplicate(
        self,
        state: DecisionState,
        candidate_id: str | None,
        student_action: CapabilityAction | None,
    ) -> CandidateAction:
        remove_ids: list[str] = []
        add_ids: list[str] = []
        if student_action and student_action.action_type == CapabilityActionType.CURATE_DOCUMENT:
            raw_adds = student_action.arguments.get("add_ids") or []
            add_ids = [
                str(d) for d in raw_adds
                if str(d) != str(candidate_id or "")
                and str(d) in set(state.pool_document_ids)
            ]
            if candidate_id and str(candidate_id) in set(state.curated_document_ids):
                remove_ids = [str(candidate_id)]
        return CandidateAction(
            action=CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={"add_ids": add_ids, "remove_ids": remove_ids},
            ),
            source="operation_realized",
            notes="SKIP_DUPLICATE",
        )

    def _keep_evidence(
        self,
        state: DecisionState,
        candidate_id: str | None,
        student_action: CapabilityAction | None,
    ) -> CandidateAction:
        add_ids = self._candidate_add_ids(state, candidate_id, student_action)
        if not add_ids and student_action:
            return CandidateAction(
                action=student_action,
                source="operation_realized",
                notes="KEEP_EVIDENCE_student",
            )
        return CandidateAction(
            action=CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={"add_ids": add_ids, "remove_ids": []},
            ),
            source="operation_realized",
            notes="KEEP_EVIDENCE",
        )

    def realize(
        self,
        decision_state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> CandidateAction | None:
        # Endorse: student action is the target
        if artifact.mode.value == "endorse":
            return CandidateAction(
                action=artifact.student_action,
                source="artifact_recommended",
                notes="endorse_student_action",
            )

        cap = artifact.resolved_capability()
        if cap == CapabilityId.DUPLICATE_EVIDENCE:
            return self._realize_duplicate(decision_state, artifact)
        if cap == CapabilityId.PREMATURE_STOP:
            return self._realize_premature_stop(decision_state, artifact)

        # Fallback: use recommended_action if present
        if artifact.recommended_action is not None:
            return CandidateAction(
                action=artifact.recommended_action,
                source="artifact_recommended",
            )
        return self._from_operation(artifact)

    def _realize_duplicate(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> CandidateAction:
        # Prefer skip / do-not-curate: empty add_ids, optionally remove duplicate target
        target = artifact.target
        remove_ids: list[str] = []
        add_ids = artifact.operation_args.get("add_ids")
        if isinstance(add_ids, list):
            # Keep non-duplicate adds if provided
            curated = set(state.curated_document_ids)
            add_ids = [d for d in add_ids if str(d) in set(state.pool_document_ids)]
        else:
            add_ids = []

        if target and str(target) in set(state.curated_document_ids) | set(state.pool_document_ids):
            # If student was about to curate the duplicate, skip it
            student_adds = artifact.student_action.arguments.get("add_ids") or []
            if target in [str(x) for x in student_adds]:
                add_ids = [d for d in student_adds if str(d) != str(target)]
            elif str(target) in set(state.curated_document_ids):
                remove_ids = [str(target)]

        if artifact.recommended_action is not None and (
            artifact.recommended_action.action_type == CapabilityActionType.CURATE_DOCUMENT
        ):
            return CandidateAction(
                action=artifact.recommended_action,
                source="artifact_recommended",
                notes="duplicate_skip_curate",
            )

        return CandidateAction(
            action=CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={"add_ids": add_ids, "remove_ids": remove_ids},
            ),
            source="deterministic",
            notes="duplicate_skip_curate",
        )

    def _realize_premature_stop(
        self,
        state: DecisionState,
        artifact: PrivilegedArtifact,
    ) -> CandidateAction:
        # Prefer existing recommended_action when fully specified
        if artifact.recommended_action is not None:
            rec = artifact.recommended_action
            if rec.action_type in {
                CapabilityActionType.CONTINUE_SEARCH,
                CapabilityActionType.SEARCH,
                CapabilityActionType.REWRITE_QUERY,
                CapabilityActionType.VERIFY_CLAIM,
            }:
                # If search without query but has query_intent → realize intent
                if rec.action_type in {
                    CapabilityActionType.CONTINUE_SEARCH,
                    CapabilityActionType.SEARCH,
                    CapabilityActionType.REWRITE_QUERY,
                }:
                    q = rec.arguments.get("query") or rec.arguments.get("queries")
                    intent = (
                        artifact.operation_args.get("query_intent")
                        or rec.arguments.get("query_intent")
                    )
                    if not q and intent:
                        query = self._intent_to_query(state, str(intent))
                        return CandidateAction(
                            action=CapabilityAction(
                                action_type=CapabilityActionType.SEARCH,
                                arguments={"query": query, "query_intent": str(intent)},
                            ),
                            source="intent_realized",
                            notes=f"intent={intent}",
                        )
                return CandidateAction(action=rec, source="artifact_recommended")

        intent = artifact.operation_args.get("query_intent") or artifact.diagnosis
        if intent:
            query = self._intent_to_query(state, str(intent))
            return CandidateAction(
                action=CapabilityAction(
                    action_type=CapabilityActionType.SEARCH,
                    arguments={"query": query, "query_intent": str(intent)},
                ),
                source="intent_realized",
                notes=f"intent={intent}",
            )

        # Default: continue search with goal-conditioned query
        query = self._intent_to_query(state, "fill_missing_claim")
        return CandidateAction(
            action=CapabilityAction(
                action_type=CapabilityActionType.CONTINUE_SEARCH,
                arguments={"query": query},
            ),
            source="deterministic",
            notes="default_continue_search",
        )

    def _intent_to_query(self, state: DecisionState, intent: str) -> str:
        """Deterministic query from intent + visible state (no LLM)."""
        goal = (state.goal or state.query or "").strip()
        unsupported = list(state.unsupported_claims)
        if not unsupported and state.evidence_claims:
            unsupported = [
                c.claim_id
                for c in state.evidence_claims
                if str(c.status).lower()
                in {"unsupported", "weak", "unverified", "partial", "unknown", ""}
            ]
        intent_l = intent.lower().replace(" ", "_")
        if "conflict" in intent_l or intent_l == "unresolved_conflict":
            return f"{goal} resolve conflicting evidence".strip()
        if "primary" in intent_l or "source" in intent_l:
            return f"{goal} primary source".strip()
        if "coverage" in intent_l or "insufficient" in intent_l or "fill_missing" in intent_l:
            if unsupported:
                return f"{goal} evidence for {unsupported[0]}".strip()
            return f"{goal} additional supporting evidence".strip()
        if "ground" in intent_l or "answer_not" in intent_l:
            return f"{goal} verify answer grounding".strip()
        return f"{goal} {intent}".strip()

    def _from_operation(self, artifact: PrivilegedArtifact) -> CandidateAction | None:
        op = artifact.recommended_operation
        if not op:
            return None
        try:
            at = CapabilityActionType(op)
        except ValueError:
            at = _OP_ALIASES.get(op)
            if at is None:
                return None
        return CandidateAction(
            action=CapabilityAction(
                action_type=at,
                arguments=dict(artifact.operation_args),
                target_claim_id=artifact.target_claim_id,
            ),
            source="deterministic",
        )


def realize(
    decision_state: DecisionState,
    artifact: PrivilegedArtifact,
) -> CandidateAction | None:
    return ActionRealizer().realize(decision_state, artifact)
