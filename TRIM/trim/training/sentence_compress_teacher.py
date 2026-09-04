"""Teacher side-branch for sentence_compress.

Teacher applies an observation transform, then emits a Student-legal
downstream action. Compressed text must never enter the Student prefix
or environment reward.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from trim.training.opd_events import HarnessEvent, model_action, obs_transform
from trim.training.rl_opd_types import StudentDecisionPoint

COMPONENT_ID = "sentence_compress"
COMPRESSED_VIEW_KEY = "compressed_teacher_view"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [p for p in parts if p]


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower()) if t}


def compress_text(query: str, text: str, *, max_sents: int = 4, max_chars: int = 480) -> str:
    """Extractive query-overlap compressor. Independent of V8D flags."""
    sents = split_sentences(text)
    if not sents:
        return (text or "")[:max_chars]
    q = tokenize(query)
    ranked = sorted(
        enumerate(sents),
        key=lambda it: (-(len(q & tokenize(it[1])) / max(1, len(q))), it[0]),
    )
    keep_idx = sorted(i for i, _ in ranked[: max(1, max_sents)])
    out = " ".join(sents[i] for i in keep_idx)
    return out[:max_chars]


def documents_from_wm(wm: Mapping[str, Any]) -> list[tuple[str, str]]:
    docs: dict[str, str] = {}

    def add(did: Any, rec: Any) -> None:
        key = str(did)
        if not key or key in docs:
            return
        if isinstance(rec, dict):
            text = str(rec.get("text") or rec.get("content") or rec.get("snippet") or "")
        else:
            text = str(rec or "")
        if text:
            docs[key] = text

    for rec in wm.get("documents") or []:
        if isinstance(rec, dict):
            add(rec.get("id") or rec.get("doc_id"), rec)
    for mapping_key in ("pool", "curated", "doc_store"):
        blob = wm.get(mapping_key) or {}
        if isinstance(blob, dict):
            for did, rec in blob.items():
                add(did, rec)
        elif isinstance(blob, list):
            for rec in blob:
                if isinstance(rec, dict):
                    add(rec.get("id") or rec.get("doc_id"), rec)
    return list(docs.items())


def score_doc(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    return len(q & tokenize(text)) / len(q)


def is_compression_active_state(wm: Mapping[str, Any], *, min_chars: int = 200) -> bool:
    return any(len(text) >= min_chars for _, text in documents_from_wm(wm))


def teacher_events_from_wm(
    wm: Mapping[str, Any],
    *,
    turn_id: int = 0,
    query: str | None = None,
) -> list[HarnessEvent]:
    q = str(query if query is not None else wm.get("query") or "")
    docs = documents_from_wm(wm)
    curated = {str(x) for x in (wm.get("curated_ids") or [])}
    compressed = {did: compress_text(q, text) for did, text in docs}
    transform = obs_transform(
        COMPONENT_ID,
        turn_id=turn_id,
        observation={
            COMPRESSED_VIEW_KEY: compressed,
            "same_underlying_docs": True,
            "original_chars": {did: len(text) for did, text in docs},
            "compressed_chars": {did: len(text) for did, text in compressed.items()},
        },
        visible_to_student=False,
        metadata={"owner": "teacher_full", "student_must_not_see": True},
    )
    if not docs:
        return [
            transform,
            model_action(
                "search_corpus",
                {"query": q},
                turn_id=turn_id,
                component_id=COMPONENT_ID,
            ),
        ]
    ranked = sorted(
        ((did, compressed[did]) for did, _ in docs if did not in curated),
        key=lambda it: (-score_doc(q, it[1]), it[0]),
    )
    add_ids = [did for did, _ in ranked[:2]]
    if not add_ids:
        add_ids = [docs[0][0]]
    return [
        transform,
        model_action(
            "curate",
            {"add_ids": add_ids, "remove_ids": []},
            turn_id=turn_id,
            component_id=COMPONENT_ID,
        ),
    ]


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(wm, turn_id=int(point.turn_id), query=wm.get("query"))


def compressed_text_leaked(payload: str) -> bool:
    lowered = (payload or "").lower()
    return COMPRESSED_VIEW_KEY.lower() in lowered or "teacher_only_observation" in lowered
