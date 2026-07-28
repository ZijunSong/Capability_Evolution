"""Visibility guard: prevent future / invisible privileged leakage."""

from __future__ import annotations

from dataclasses import dataclass

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState


@dataclass(frozen=True)
class VisibilityCheck:
    valid: bool
    violations: tuple[str, ...]


def _action_document_refs(action: CapabilityAction | None) -> set[str]:
    if action is None:
        return set()
    args = action.arguments or {}
    refs: set[str] = set()
    for key in ("doc_ids", "add_ids", "remove_ids", "document_ids"):
        vals = args.get(key)
        if isinstance(vals, list):
            refs.update(str(x) for x in vals if x)
        elif isinstance(vals, str) and vals:
            refs.add(vals)
    for key in ("doc_id", "id"):
        if args.get(key):
            refs.add(str(args[key]))
    return refs


def check_artifact_visibility(
    state: DecisionState,
    artifact: PrivilegedArtifact,
) -> VisibilityCheck:
    violations: list[str] = []

    if artifact.episode_id != state.episode_id:
        violations.append("episode_id_mismatch")
    if artifact.turn_id != state.turn_id:
        violations.append("turn_id_mismatch")
    # task_id is on state; artifact may store it in metadata
    task_id = artifact.metadata.get("task_id")
    if task_id is not None and task_id != state.task_id:
        violations.append("task_id_mismatch")

    visible_obs = set(state.observation_ids) | set(getattr(state, "observed_ids", ()))
    for eid in artifact.evidence_ids:
        if eid not in visible_obs:
            violations.append(f"evidence_not_visible:{eid}")

    visible_docs = set(state.visible_document_ids) | set(state.pool_document_ids)
    for did in artifact.document_ids:
        if did not in visible_docs:
            violations.append(f"document_not_visible:{did}")

    for did in _action_document_refs(artifact.recommended_action):
        if did not in visible_docs:
            violations.append(f"recommended_doc_not_visible:{did}")

    # Future turn leakage: evidence created after current turn
    # (DecisionState only exposes current observation_ids; extra check via metadata)
    future_ids = artifact.metadata.get("future_observation_ids") or []
    if future_ids:
        violations.append("future_observation_present")

    # Recommended query may be new text, but must not embed hidden answer markers
    if artifact.recommended_action is not None:
        q = str(artifact.recommended_action.arguments.get("query", ""))
        if "__HIDDEN__" in q or "FUTURE_OBS" in q:
            violations.append("recommended_query_hidden_fact")

    return VisibilityCheck(valid=len(violations) == 0, violations=tuple(violations))


def mask_artifact_if_invalid(
    state: DecisionState,
    artifact: PrivilegedArtifact,
) -> tuple[PrivilegedArtifact, VisibilityCheck]:
    check = check_artifact_visibility(state, artifact)
    if check.valid:
        return artifact, check
    from dataclasses import replace

    masked = replace(
        artifact,
        mode=GuidanceMode.IGNORE,
        recommended_action=None,
        metadata={
            **artifact.metadata,
            "visibility_violations": list(check.violations),
            "masked": True,
        },
    )
    return masked, check
