"""Model-family action encoding, online parser roundtrip, and visibility checks.

Standard RL+OPD CE must use on-policy Student prompt IDs and the same
tool-call surface as rollout. Missing IDs are rejected unless the row is
explicitly marked offline.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from trim.training.action_codec import canonicalize_action
from trim.training.opd_prompt_encoding import (
    assert_supervised_action_matches,
    encode_rollout_style_action,
    encode_rollout_style_prompt,
    render_rollout_action_text,
)
from trim.training.opd_realizability import accessible_doc_ids_of, referenced_doc_ids

TOOLS_WITHOUT_VISIBLE_DOC_REQUIREMENT = frozenset(
    {
        "search_corpus",
        "fan_out_search",
        "grep_corpus",
        "end_search",
        "init",
        "answer",
        "user_text",
        "unknown",
        "truncated",
        "None",
    }
)


def prompt_ids_hash(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(x)) for x in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decode_token_ids(enc: Any, ids: Sequence[int]) -> str:
    tokens = [int(x) for x in ids]
    if enc is None or not tokens:
        return ""
    if hasattr(enc, "decode_tokens"):
        return str(enc.decode_tokens(tokens))
    if hasattr(enc, "tokenizer") and enc.tokenizer is not None:
        return str(enc.tokenizer.decode(tokens, skip_special_tokens=False))
    try:
        return bytes(tokens).decode("utf-8", errors="replace")
    except Exception:
        return ""


def visible_doc_ids_from_snapshot(snapshot: Any) -> list[str]:
    if snapshot is None:
        return []
    return [str(x) for x in accessible_doc_ids_of(snapshot)]


def visible_doc_ids_for_decision(
    *,
    snapshot: Any = None,
    prompt_ids: Sequence[int] | None = None,
    enc: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """Docs the Student can actually condition on after context clipping.

    Prefer an explicit ``visible_doc_ids`` recorded at sample time. Otherwise
    intersect snapshot-accessible IDs with the decoded prompt text when a
    decoder is available.
    """
    meta = dict(metadata or {})
    if "visible_doc_ids" in meta and meta.get("visible_doc_ids") is not None:
        return [str(x) for x in list(meta["visible_doc_ids"])]
    snap_ids = visible_doc_ids_from_snapshot(snapshot)
    if not snap_ids:
        return []
    if enc is None or not prompt_ids:
        return snap_ids
    text = decode_token_ids(enc, prompt_ids)
    if not text:
        return snap_ids
    return [did for did in snap_ids if did and did in text]


def assert_action_visible(
    action: Mapping[str, Any] | Any,
    visible_doc_ids: Sequence[str],
) -> None:
    """Reject projected targets that reference docs dropped by clipping."""
    canon = canonicalize_action(action)
    name = str(canon.get("name") or "")
    if name in TOOLS_WITHOUT_VISIBLE_DOC_REQUIREMENT:
        return
    refs = referenced_doc_ids(canon)
    if not refs:
        return
    visible = {str(x) for x in visible_doc_ids}
    missing = [did for did in refs if did not in visible]
    if missing:
        raise ValueError(
            f"projected action {name} references docs not visible after clipping: {missing}"
        )


def resolve_student_prompt_ids(
    *,
    metadata: Mapping[str, Any] | None,
    prompt_reduced: str,
    encode: Any,
    allow_offline: bool = False,
) -> list[int]:
    meta = dict(metadata or {})
    raw = meta.get("student_prompt_token_ids")
    if raw:
        return [int(x) for x in list(raw)]
    offline = bool(allow_offline or meta.get("offline"))
    if not offline:
        raise ValueError(
            "missing student_prompt_token_ids for on-policy CE; "
            "refuse silent debug-string encode"
        )
    return list(encode(prompt_reduced))


def encode_supervised_action(
    *,
    target_action: Mapping[str, Any] | None,
    target_text: str,
    encode: Any,
    model_enc: Any | None = None,
) -> tuple[list[int], str]:
    """Encode a* with the rollout tool-call format when a model encoder exists."""
    if model_enc is not None and target_action:
        ids, text = encode_rollout_style_action(model_enc, dict(target_action))
        stops = [int(x) for x in (getattr(model_enc, "stop_token_ids", None) or [])]
        if stops and ids and int(ids[-1]) not in stops:
            ids = list(ids) + [stops[0]]
        assert_supervised_action_matches(
            model_enc,
            target_action=dict(target_action),
            target_token_ids=ids,
        )
        return list(ids), text
    text = str(target_text or "")
    return list(encode(text)), text


def validate_finite_weight(value: Any, *, what: str = "weight") -> float:
    w = float(value)
    if not math.isfinite(w):
        raise ValueError(f"{what} must be finite, got {value!r}")
    if w < 0.0:
        raise ValueError(f"{what} must be >= 0, got {w}")
    return w


def resolve_effective_weight(
    *,
    weight: Any,
    projection_confidence: Any,
    metadata: Mapping[str, Any] | None = None,
) -> float:
    """Single effective_weight: already includes confidence and candidate sharing.

    Never re-multiply confidence. Explicit 0 is skip-training; None is missing.
    """
    meta = dict(metadata or {})
    if meta.get("effective_weight") is not None:
        return validate_finite_weight(meta["effective_weight"], what="effective_weight")
    # Prefer the already-shared ``weight`` over truthy projection_confidence.
    if weight is not None:
        return validate_finite_weight(weight, what="step.weight")
    if projection_confidence is not None:
        return validate_finite_weight(projection_confidence, what="projection_confidence")
    return 1.0


__all__ = [
    "TOOLS_WITHOUT_VISIBLE_DOC_REQUIREMENT",
    "assert_action_visible",
    "assert_supervised_action_matches",
    "decode_token_ids",
    "encode_rollout_style_action",
    "encode_rollout_style_prompt",
    "encode_supervised_action",
    "prompt_ids_hash",
    "render_rollout_action_text",
    "resolve_effective_weight",
    "resolve_student_prompt_ids",
    "validate_finite_weight",
    "visible_doc_ids_for_decision",
    "visible_doc_ids_from_snapshot",
]
