"""Build SCOPE v3 supervision datasets from online DecisionState audit events."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.artifacts.schema import GuidanceMode, PrivilegedArtifact
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import (
    ROUND1_ENABLED_CAPABILITIES,
    REASON_CODE_TO_CAPABILITY,
    is_round1_trainable,
    parse_capability_id,
)
from harness.capability.state import DecisionState, VerificationRecordState
from training.scope.routing import route_decision
from training.scope.schema import DecisionSupervisionSampleV3, Route

_DOC_ID_RE = re.compile(r"\b(\d{3,})\b")


@dataclass
class DatasetBuildConfig:
    enabled_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(c.value for c in ROUND1_ENABLED_CAPABILITIES)
    )
    valid_fraction: float = 0.1
    seed: int = 42
    drop_ignore: bool = False
    targeted_probe_train_mask: int = 0  # probes default off for training


@dataclass
class DatasetManifest:
    n_samples: int
    n_train: int
    n_valid: int
    n_queries: int
    route_counts: dict[str, int]
    capability_counts: dict[str, int]
    stop_endorse: int = 0
    stop_correct: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_queries": self.n_queries,
            "route_counts": dict(self.route_counts),
            "capability_counts": dict(self.capability_counts),
            "stop_endorse": self.stop_endorse,
            "stop_correct": self.stop_correct,
            "provenance": dict(self.provenance),
        }


def _collect_doc_ids(event: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    args = event.get("action_arguments") or {}
    for key in ("add_ids", "remove_ids", "doc_ids", "document_ids"):
        vals = args.get(key)
        if isinstance(vals, list):
            ids.extend(str(x) for x in vals)
    rec = event.get("recommended_action")
    if isinstance(rec, dict):
        rargs = rec.get("arguments") or {}
        for key in ("add_ids", "remove_ids", "doc_ids", "document_ids"):
            vals = rargs.get(key)
            if isinstance(vals, list):
                ids.extend(str(x) for x in vals)
            elif isinstance(vals, str) and vals:
                ids.append(vals)
        if rargs.get("doc_id"):
            ids.append(str(rargs["doc_id"]))
    for r in event.get("verification_records") or []:
        if isinstance(r, dict):
            ids.extend(str(x) for x in (r.get("document_ids") or []))
    if not ids:
        ctx = event.get("rendered_context") or ""
        ids.extend(_DOC_ID_RE.findall(ctx)[:40])
    seen: set[str] = set()
    out: list[str] = []
    for d in ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _parse_student_action(event: dict[str, Any]) -> CapabilityAction:
    if isinstance(event.get("student_action_struct"), dict):
        return CapabilityAction.from_dict(event["student_action_struct"])
    raw = event.get("student_action")
    if isinstance(raw, dict):
        return CapabilityAction.from_dict(raw)
    try:
        at = CapabilityActionType(str(raw))
    except ValueError:
        at = CapabilityActionType.UNKNOWN
    return CapabilityAction(
        action_type=at,
        arguments=dict(event.get("action_arguments") or {}),
    )


def _reconstruct_from_flat_event(
    event: dict[str, Any],
) -> tuple[DecisionState, PrivilegedArtifact, CapabilityAction] | None:
    """Rebuild state/artifact from legacy flat audit events (Go25-style)."""
    reason = str(event.get("reason_code") or "")
    cap = parse_capability_id(event.get("capability_id"))
    if cap.value == "unknown":
        mapped = REASON_CODE_TO_CAPABILITY.get(reason)
        student_type = str(event.get("student_action") or "")
        if mapped is None and student_type in {"stop_and_answer", "answer", "abstain"}:
            from harness.capability.capability_id import CapabilityId

            mapped = CapabilityId.PREMATURE_STOP
        if mapped is None:
            return None
        cap = mapped

    student = _parse_student_action(event)
    rec_raw = event.get("recommended_action")
    recommended = CapabilityAction.from_dict(rec_raw) if isinstance(rec_raw, dict) else None

    doc_ids = _collect_doc_ids(event)
    curated_n = int(event.get("curated") or 0)
    curated = (
        tuple(doc_ids[:curated_n])
        if curated_n
        else tuple(doc_ids[: max(1, len(doc_ids) // 2)]) if doc_ids else ()
    )
    pool = tuple(doc_ids) if doc_ids else curated

    verify_recs = tuple(
        VerificationRecordState(
            turn_id=int(r.get("turn_id", 0)),
            claim=str(r.get("claim", "")),
            document_ids=tuple(r.get("document_ids") or []),
            judgments={str(k): bool(v) for k, v in dict(r.get("judgments") or {}).items()},
        )
        for r in (event.get("verification_records") or [])
        if isinstance(r, dict)
    )

    qid = str(event.get("query_id") or event.get("task_id") or "unknown")
    turn = int(event.get("turn_id", 0))
    state = DecisionState(
        episode_id=str(event.get("episode_id") or f"ep_{qid}"),
        task_id=qid,
        turn_id=turn,
        event_id=str(event.get("event_id") or f"{qid}:{turn}"),
        query=str(event.get("query") or ""),
        goal=str(event.get("query") or ""),
        rendered_context=str(event.get("rendered_context") or ""),
        action_history=(),
        observation_ids=("obs_legacy_0",),
        visible_document_ids=pool,
        pool_document_ids=pool,
        curated_document_ids=curated,
        evidence_claims=(),
        verification_records=verify_recs,
        remaining_turns=max(0, 35 - turn),
        remaining_search_calls=None,
        token_budget_used=0,
        token_budget_total=32768,
        last_action_type=student.action_type.value,
        last_action_arguments=dict(student.arguments),
        repeated_query_score=0.0,
        wm_snapshot_hash="legacy",
    )

    try:
        mode = GuidanceMode(str(event.get("mode", "ignore")))
    except ValueError:
        mode = GuidanceMode.IGNORE

    artifact = PrivilegedArtifact.build(
        episode_id=state.episode_id,
        turn_id=state.turn_id,
        module_id=str(event.get("module_id") or "unknown"),
        mode=mode,
        reason_code=reason or "EVIDENCE_UPDATE_VALID",
        student_action=student,
        recommended_action=recommended,
        evidence_ids=state.observation_ids,
        document_ids=curated[:10] or pool[:10],
        confidence=0.8,
        metadata={
            "task_id": state.task_id,
            "legacy_flat_event": True,
            "local_label": event.get("local_label"),
            "local_capabilities": event.get("local_capabilities"),
        },
        capability_id=cap.value,
        teacher_source="legacy_audit_event",
        runtime_fields_used=("remaining_turns",) if cap.value == "premature_stop" else (),
    )
    return state, artifact, student


def _event_to_sample(
    event: dict[str, Any], cfg: DatasetBuildConfig
) -> DecisionSupervisionSampleV3 | None:
    """Convert an audit event dict into a V3 sample when possible."""
    if event.get("schema_version") == "scope.supervision.v3":
        return DecisionSupervisionSampleV3.from_dict(event)

    state_raw = event.get("decision_state") or event.get("state")
    artifact_raw = event.get("artifact")

    if state_raw and artifact_raw:
        state = DecisionState.from_dict(state_raw)
        artifact = PrivilegedArtifact.from_dict(artifact_raw)
        if isinstance(event.get("student_action_struct"), dict):
            student = CapabilityAction.from_dict(event["student_action_struct"])
        elif isinstance(event.get("student_action"), dict):
            student = CapabilityAction.from_dict(event["student_action"])
        else:
            student = artifact.student_action
    else:
        rebuilt = _reconstruct_from_flat_event(event)
        if rebuilt is None:
            return None
        state, artifact, student = rebuilt

    cap = artifact.resolved_capability()
    if cap.value not in cfg.enabled_capabilities and not is_round1_trainable(cap):
        return None

    result = route_decision(
        state,
        artifact,
        student,
        event_id=str(event.get("event_id", "")),
        student_state_text=str(event.get("rendered_context") or state.rendered_context),
        enforce_round1_capability_filter=True,
    )
    sample = result.sample

    if event.get("targeted_probe") or event.get("source") == "targeted_probe":
        from dataclasses import replace

        sample = replace(
            sample,
            train_mask=cfg.targeted_probe_train_mask,
            metadata={**sample.metadata, "targeted_probe": True},
        )
    return sample


def split_by_query(
    samples: list[DecisionSupervisionSampleV3],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[list[DecisionSupervisionSampleV3], list[DecisionSupervisionSampleV3]]:
    """Query-level split (not event-level)."""
    by_query: dict[str, list[DecisionSupervisionSampleV3]] = defaultdict(list)
    for s in samples:
        qid = s.task_id or s.episode_id
        by_query[qid].append(s)
    keys = sorted(by_query.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_valid = max(1, int(round(len(keys) * valid_fraction))) if keys else 0
    if len(keys) <= 1:
        n_valid = 0
    valid_keys = set(keys[:n_valid])
    train, valid = [], []
    for k, items in by_query.items():
        (valid if k in valid_keys else train).extend(items)
    return train, valid


def build_dataset_from_events(
    events: Iterable[dict[str, Any]],
    cfg: DatasetBuildConfig | None = None,
    *,
    provenance: dict[str, Any] | None = None,
) -> tuple[list[DecisionSupervisionSampleV3], list[DecisionSupervisionSampleV3], DatasetManifest]:
    cfg = cfg or DatasetBuildConfig()
    samples: list[DecisionSupervisionSampleV3] = []
    for ev in events:
        s = _event_to_sample(ev, cfg)
        if s is None:
            continue
        if cfg.drop_ignore and s.route == Route.IGNORE:
            continue
        samples.append(s)

    train, valid = split_by_query(
        samples, valid_fraction=cfg.valid_fraction, seed=cfg.seed
    )

    route_counts: dict[str, int] = defaultdict(int)
    cap_counts: dict[str, int] = defaultdict(int)
    stop_endorse = stop_correct = 0
    for s in samples:
        route_counts[s.route.value] += 1
        cap_counts[s.capability_id] += 1
        if s.capability_id == "premature_stop":
            if s.route == Route.ENDORSE:
                stop_endorse += 1
            elif s.route == Route.CORRECT:
                stop_correct += 1

    queries = {s.task_id or s.episode_id for s in samples}
    manifest = DatasetManifest(
        n_samples=len(samples),
        n_train=len(train),
        n_valid=len(valid),
        n_queries=len(queries),
        route_counts=dict(route_counts),
        capability_counts=dict(cap_counts),
        stop_endorse=stop_endorse,
        stop_correct=stop_correct,
        provenance=dict(provenance or {}),
    )
    return train, valid, manifest


def write_split_jsonl(
    path: Path,
    samples: list[DecisionSupervisionSampleV3],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events
