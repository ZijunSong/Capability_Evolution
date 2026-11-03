"""Probe C real-state data: schema, strict loader, grouping, provenance audit.

Formal Probe C forbids the legacy compat paths: no FakeSearchEnv states, no
query_id back-fill, no observation copied into multiple documents, no gold doc
store back-fill, no Teacher/Student branch mixing. Every rejection carries an
explicit reason; nothing silently degrades to legacy defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from trim.probes.probe_c_contract import (
    SPLIT_AUDIT,
    SPLIT_NAMES,
    SPLIT_SELECT,
    SPLIT_TASK_TEST,
    SPLIT_TRAIN,
    STATE_SOURCE_REAL,
    HarnessPairSpec,
)
from trim.state.snapshot import stable_hash

PROBE_C_STATE_SCHEMA_VERSION = "probe_c_state_v1"


class StrictLoadError(ValueError):
    """Raised when a state row violates the formal Probe-C data contract."""

    def __init__(self, reason: str, *, row_index: int | None = None, detail: Any = None):
        self.reason = str(reason)
        self.row_index = row_index
        self.detail = detail
        prefix = f"[{self.reason}]"
        if row_index is not None:
            prefix += f" row={row_index}"
        super().__init__(f"{prefix} {detail if detail is not None else ''}".strip())


@dataclass
class ProbeCState:
    """One pre-action H0 decision state. No post-action or scorer feedback."""

    schema_version: str
    source: str
    model_id: str
    model_revision: str
    policy_checksum: str
    adapter_checksum: str
    query_id: str
    query_text: str
    scenario_group_id: str
    split: str
    rollout_seed: int
    turn_id: int
    student_snapshot: dict[str, Any]
    env_content_hash: str
    snapshot_hash: str
    env_restore_version: str
    raw_messages: list[dict[str, Any]]
    effective_student_prompt_ids: list[int]
    effective_tool_schema: dict[str, Any]
    student_action_ids: list[int]
    sampled_old_logprobs: list[float]
    original_action_parse_result: dict[str, Any]
    student_mask: dict[str, bool]
    teacher_mask: dict[str, bool]
    component_revision: str
    information_provenance: list[dict[str, Any]]
    tokenizer_hash: str
    chat_template_hash: str
    enable_thinking: bool
    truncation_policy_hash: str
    row_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extra"] = dict(self.extra)
        return payload

    @property
    def state_id(self) -> str:
        return self.row_id or stable_hash(
            {
                "query_id": self.query_id,
                "turn_id": self.turn_id,
                "rollout_seed": self.rollout_seed,
                "snapshot_hash": self.snapshot_hash,
            }
        )[:16]

    @property
    def group_key(self) -> str:
        return self.scenario_group_id or self.query_id


# ---------------------------------------------------------------------------
# env-content hash: environment identity, not full-object identity
# ---------------------------------------------------------------------------


def env_content_hash(snapshot: Mapping[str, Any] | Any) -> str:
    """Hash of environment content *excluding* configuration labels.

    Distinguishes "same environment content" (this hash) from "same full
    snapshot object" (``EnvironmentSnapshot.content_hash`` which also covers
    masks/metadata).
    """
    if hasattr(snapshot, "working_memory"):
        payload = {
            "query_id": getattr(snapshot, "query_id", ""),
            "query_text": getattr(snapshot, "query_text", ""),
            "step": getattr(snapshot, "step", 0),
            "working_memory": getattr(snapshot, "working_memory", {}),
            "tool_history": list(getattr(snapshot, "tool_history", []) or []),
            "observations": list(getattr(snapshot, "observations", []) or []),
        }
    else:
        snap = dict(snapshot.get("student_snapshot") or snapshot)
        payload = {
            "query_id": snap.get("query_id", ""),
            "query_text": snap.get("query_text", ""),
            "step": snap.get("step", 0),
            "working_memory": snap.get("working_memory") or {},
            "tool_history": list(snap.get("tool_history") or []),
            "observations": list(snap.get("observations") or []),
        }
    return stable_hash(payload)


# ---------------------------------------------------------------------------
# strict loader
# ---------------------------------------------------------------------------

_PLACEHOLDER_TEXTS = {"", "n/a", "none", "null", "unknown"}


def _reject(reason: str, row_index: int | None = None, detail: Any = None) -> None:
    raise StrictLoadError(reason, row_index=row_index, detail=detail)


def _require(condition: bool, reason: str, row_index: int | None = None, detail: Any = None) -> None:
    if not condition:
        _reject(reason, row_index, detail)


def validate_state_row(
    row: Mapping[str, Any],
    *,
    row_index: int | None = None,
    pair: HarnessPairSpec | None = None,
    expected_source: str = STATE_SOURCE_REAL,
) -> ProbeCState:
    """Validate one raw row into a ProbeCState or raise StrictLoadError."""
    if not isinstance(row, Mapping):
        _reject("ROW_NOT_MAPPING", row_index, type(row).__name__)

    source = str(row.get("source") or "")
    _require(
        source == expected_source,
        "FAKE_OR_WRONG_SOURCE",
        row_index,
        f"source={source!r} required={expected_source!r}",
    )
    schema = str(row.get("schema_version") or "")
    _require(bool(schema), "MISSING_SCHEMA_VERSION", row_index)
    if schema != PROBE_C_STATE_SCHEMA_VERSION:
        _require(
            str(row.get("schema_version") or "") == PROBE_C_STATE_SCHEMA_VERSION,
            "SCHEMA_VERSION_MISMATCH",
            row_index,
            f"got {schema!r} want {PROBE_C_STATE_SCHEMA_VERSION!r}",
        )

    query_text = str(row.get("query_text") or "").strip()
    _require(
        query_text.lower() not in _PLACEHOLDER_TEXTS,
        "MISSING_QUERY_TEXT",
        row_index,
        "query_id back-fill is forbidden",
    )
    query_id = str(row.get("query_id") or "").strip()
    _require(bool(query_id), "MISSING_QUERY_ID", row_index)

    raw_messages = row.get("raw_messages")
    _require(
        isinstance(raw_messages, list) and len(raw_messages) > 0,
        "MISSING_VISIBLE_HISTORY",
        row_index,
        "raw_messages must contain the actual visible history",
    )
    prompt_ids = row.get("effective_student_prompt_ids")
    _require(
        isinstance(prompt_ids, list) and len(prompt_ids) > 0,
        "MISSING_EFFECTIVE_PROMPT_IDS",
        row_index,
        "effective_student_prompt_ids prove what the model actually saw",
    )
    tool_schema = row.get("effective_tool_schema")
    _require(
        isinstance(tool_schema, Mapping) and len(tool_schema) > 0,
        "MISSING_TOOL_SCHEMA",
        row_index,
        "effective_tool_schema cannot be restored as empty",
    )
    _require(
        bool(str(row.get("truncation_policy_hash") or "")),
        "MISSING_TRUNCATION_POLICY",
        row_index,
    )
    if bool(row.get("truncation_lost_target")):
        _reject(
            "TRUNCATION_LOST_TARGET_OBJECT",
            row_index,
            "prompt truncation dropped an object the target needs",
        )

    split = str(row.get("split") or "")
    _require(split in SPLIT_NAMES, "UNKNOWN_SPLIT", row_index, split)

    student_snapshot = row.get("student_snapshot")
    _require(
        isinstance(student_snapshot, Mapping) and len(student_snapshot) > 0,
        "MISSING_STUDENT_SNAPSHOT",
        row_index,
    )

    # Teacher / Student branch mixing is a measurement error, not a soft warning.
    prov = row.get("information_provenance")
    _require(
        isinstance(prov, list),
        "MISSING_INFORMATION_PROVENANCE",
        row_index,
    )
    for entry in prov or []:
        if not isinstance(entry, Mapping):
            continue
        owner = str(entry.get("owner") or entry.get("source") or "")
        kind = str(entry.get("teacher_kind") or entry.get("source_type") or "")
        if owner.startswith("teacher") or kind.startswith("teacher_branch"):
            _reject(
                "TEACHER_STUDENT_BRANCH_MIX",
                row_index,
                f"student state carries teacher-branch provenance {entry}",
            )
        if str(entry.get("source") or "") in {"gold_doc_store", "gold_documents"}:
            _reject("GOLD_DOC_STORE_BACKFILL", row_index, dict(entry))
    snap_meta = student_snapshot.get("metadata") or {}
    if isinstance(snap_meta, Mapping):
        if str(snap_meta.get("owner") or "") not in {"", "student_reduced"}:
            _reject(
                "TEACHER_STUDENT_BRANCH_MIX",
                row_index,
                f"snapshot owner={snap_meta.get('owner')!r}",
            )

    # Anti-backfill: one *observation* copied into many "documents" (the
    # frozen_state_loader fabrication signature). Genuine corpus duplicates
    # (SEC boilerplate shared by distinct doc ids) are legal and only counted.
    docs = (student_snapshot.get("working_memory") or {}).get("documents") or []
    obs_texts = {
        str(m.get("content") or "").strip()
        for m in raw_messages
        if isinstance(m, Mapping)
        and str(m.get("role") or "") == "tool"
        and len(str(m.get("content") or "").strip()) >= 40
    }
    seen_texts: dict[str, str] = {}
    n_dup_groups = 0
    for doc in docs:
        if not isinstance(doc, Mapping):
            continue
        text = str(doc.get("text") or "").strip()
        did = str(doc.get("id") or "")
        if len(text) < 40:
            continue
        prev = seen_texts.get(text)
        if prev is not None and prev != did:
            copied_obs = any(
                text == obs or (obs in text) or (text in obs) for obs in obs_texts
            )
            if copied_obs:
                _reject(
                    "OBSERVATION_COPIED_INTO_MULTIPLE_DOCS",
                    row_index,
                    f"observation text shared by docs {prev!r} and {did!r}",
                )
            n_dup_groups += 1
        seen_texts[text] = did
    if n_dup_groups:
        row_extra_dup = n_dup_groups
    else:
        row_extra_dup = 0

    for key in (
        "model_id",
        "model_revision",
        "policy_checksum",
        "tokenizer_hash",
        "chat_template_hash",
        "component_revision",
    ):
        _require(
            str(row.get(key) or "").strip().lower() not in _PLACEHOLDER_TEXTS,
            "MISSING_IDENTITY_FIELD",
            row_index,
            key,
        )

    student_mask = row.get("student_mask")
    teacher_mask = row.get("teacher_mask")
    _require(
        isinstance(student_mask, Mapping) and isinstance(teacher_mask, Mapping),
        "MISSING_MASKS",
        row_index,
    )
    if pair is not None:
        if dict(student_mask) != dict(pair.student_mask):
            _reject(
                "MASK_PAIR_MISMATCH",
                row_index,
                {"want_student": pair.student_mask, "got": dict(student_mask)},
            )
        if dict(teacher_mask) != dict(pair.teacher_mask):
            _reject(
                "MASK_PAIR_MISMATCH",
                row_index,
                {"want_teacher": pair.teacher_mask, "got": dict(teacher_mask)},
            )

    state = ProbeCState(
        schema_version=schema,
        source=source,
        model_id=str(row["model_id"]),
        model_revision=str(row["model_revision"]),
        policy_checksum=str(row["policy_checksum"]),
        adapter_checksum=str(row.get("adapter_checksum") or ""),
        query_id=query_id,
        query_text=query_text,
        scenario_group_id=str(row.get("scenario_group_id") or query_id),
        split=split,
        rollout_seed=int(row.get("rollout_seed") or 0),
        turn_id=int(row.get("turn_id") or 0),
        student_snapshot=dict(student_snapshot),
        env_content_hash=str(row.get("env_content_hash") or env_content_hash(student_snapshot)),
        snapshot_hash=str(row.get("snapshot_hash") or stable_hash(student_snapshot)),
        env_restore_version=str(row.get("env_restore_version") or ""),
        raw_messages=list(raw_messages),
        effective_student_prompt_ids=[int(t) for t in prompt_ids],
        effective_tool_schema=dict(tool_schema),
        student_action_ids=[int(t) for t in (row.get("student_action_ids") or [])],
        sampled_old_logprobs=[float(x) for x in (row.get("sampled_old_logprobs") or [])],
        original_action_parse_result=dict(row.get("original_action_parse_result") or {}),
        student_mask=dict(student_mask),
        teacher_mask=dict(teacher_mask),
        component_revision=str(row["component_revision"]),
        information_provenance=[dict(e) for e in prov if isinstance(e, Mapping)],
        tokenizer_hash=str(row["tokenizer_hash"]),
        chat_template_hash=str(row["chat_template_hash"]),
        enable_thinking=bool(row.get("enable_thinking", False)),
        truncation_policy_hash=str(row["truncation_policy_hash"]),
        row_id=str(row.get("row_id") or ""),
        extra={
            **{k: v for k, v in row.items() if k not in _KNOWN_KEYS},
            "duplicate_doc_text_groups": row_extra_dup,
        },
    )
    # Hash consistency: env_content_hash must describe the same environment.
    recomputed = env_content_hash(student_snapshot)
    if state.env_content_hash and state.env_content_hash != recomputed:
        _reject(
            "ENV_CONTENT_HASH_MISMATCH",
            row_index,
            {"declared": state.env_content_hash, "recomputed": recomputed},
        )
    return state


_KNOWN_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "model_id",
        "model_revision",
        "policy_checksum",
        "adapter_checksum",
        "query_id",
        "query_text",
        "scenario_group_id",
        "split",
        "rollout_seed",
        "turn_id",
        "student_snapshot",
        "env_content_hash",
        "snapshot_hash",
        "env_restore_version",
        "raw_messages",
        "effective_student_prompt_ids",
        "effective_tool_schema",
        "student_action_ids",
        "sampled_old_logprobs",
        "original_action_parse_result",
        "student_mask",
        "teacher_mask",
        "component_revision",
        "information_provenance",
        "tokenizer_hash",
        "chat_template_hash",
        "enable_thinking",
        "truncation_policy_hash",
        "truncation_lost_target",
        "row_id",
    }
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_strict_real_h0_states(
    path: str | Path | Sequence[Mapping[str, Any]],
    *,
    pair: HarnessPairSpec | None = None,
    expected_source: str = STATE_SOURCE_REAL,
) -> list[ProbeCState]:
    """Strict loader. Any violation raises; there is no legacy fallback."""
    if isinstance(path, (str, Path)):
        rows: Sequence[Mapping[str, Any]] = list(iter_jsonl(Path(path)))
    else:
        rows = list(path)
    states = [
        validate_state_row(row, row_index=i, pair=pair, expected_source=expected_source)
        for i, row in enumerate(rows)
    ]
    assert_group_disjoint(states)
    return states


# ---------------------------------------------------------------------------
# split independence
# ---------------------------------------------------------------------------


def assert_group_disjoint(states: Sequence[ProbeCState]) -> None:
    """Same query / scenario group must not cross splits.

    Same-question different turns, re-sampled trajectories and near-duplicate
    questions share a scenario_group_id; grouping happens before splitting.
    """
    by_group: dict[str, set[str]] = {}
    by_query: dict[str, set[str]] = {}
    for st in states:
        by_group.setdefault(st.group_key, set()).add(st.split)
        by_query.setdefault(st.query_id, set()).add(st.split)
    for key, splits in by_group.items():
        if len(splits) > 1:
            _reject(
                "CROSS_SPLIT_GROUP",
                None,
                {"group": key, "splits": sorted(splits)},
            )
    for key, splits in by_query.items():
        if len(splits) > 1:
            _reject(
                "CROSS_SPLIT_QUERY",
                None,
                {"query_id": key, "splits": sorted(splits)},
            )
    # Duplicate states within one split are also a contamination signal.
    seen: dict[tuple[str, str], str] = {}
    for st in states:
        key = (st.split, st.snapshot_hash)
        if key in seen:
            _reject(
                "DUPLICATE_STATE_IN_SPLIT",
                None,
                {"split": st.split, "snapshot_hash": st.snapshot_hash},
            )
        seen[key] = st.state_id


def split_states(states: Sequence[ProbeCState]) -> dict[str, list[ProbeCState]]:
    out: dict[str, list[ProbeCState]] = {name: [] for name in SPLIT_NAMES}
    for st in states:
        out.setdefault(st.split, []).append(st)
    return out


def split_audit(states: Sequence[ProbeCState]) -> dict[str, Any]:
    """SPLIT_AUDIT.json content: query/scenario independence and coverage."""
    by_split = split_states(states)
    counts: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        rows = by_split.get(name) or []
        counts[name] = {
            "n_states": len(rows),
            "n_queries": len({s.query_id for s in rows}),
            "n_groups": len({s.group_key for s in rows}),
        }
    groups: dict[str, set[str]] = {}
    for st in states:
        groups.setdefault(st.group_key, set()).add(st.split)
    overlapping = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    return {
        "schema_version": PROBE_C_STATE_SCHEMA_VERSION,
        "counts": counts,
        "n_groups_total": len(groups),
        "cross_split_groups": overlapping,
        "group_disjoint": not overlapping,
        "report_actual_counts": True,
        "note": "state shortfall is reported as actual counts; rows are never "
        "duplicated to fake a target sample size",
    }


def write_states_jsonl(states: Sequence[ProbeCState], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for st in states:
            handle.write(json.dumps(st.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return p


def write_split_audit(states: Sequence[ProbeCState], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "SPLIT_AUDIT.json"
    path.write_text(
        json.dumps(split_audit(states), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


__all__ = [
    "PROBE_C_STATE_SCHEMA_VERSION",
    "ProbeCState",
    "StrictLoadError",
    "assert_group_disjoint",
    "env_content_hash",
    "load_strict_real_h0_states",
    "split_audit",
    "split_states",
    "validate_state_row",
    "write_split_audit",
    "write_states_jsonl",
]
