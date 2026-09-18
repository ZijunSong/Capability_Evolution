"""Model-family action encoding, online parser roundtrip, and visibility checks.

Standard RL+OPD CE must use on-policy Student prompt IDs and the same
tool-call surface as rollout. Missing IDs are rejected unless the row is
explicitly marked offline.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trim.training.action_codec import canonicalize_action
from trim.training.opd_prompt_encoding import (
    assert_supervised_action_matches,
    encode_rollout_style_action,
    encode_rollout_style_prompt,
    is_harmony_encoder,
    render_harmony_tool_call_completion,
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


def _encode_text_tokens(enc: Any, text: str) -> list[int]:
    if enc is None or not text:
        return []
    if hasattr(enc, "encode"):
        return [int(x) for x in list(enc.encode(text))]
    if hasattr(enc, "tokenizer") and enc.tokenizer is not None:
        return [int(x) for x in list(enc.tokenizer.encode(text, add_special_tokens=False))]
    return []


def _find_token_span(
    haystack: Sequence[int], needle: Sequence[int], claimed: set[int]
) -> tuple[int, int] | None:
    if not needle:
        return None
    n = len(needle)
    hay = [int(x) for x in haystack]
    need = [int(x) for x in needle]
    for start in range(0, len(hay) - n + 1):
        end = start + n
        if hay[start:end] != need:
            continue
        if any(idx in claimed for idx in range(start, end)):
            continue
        return start, end
    return None


def prompt_visible_doc_ids_from_prompt(
    *,
    prompt_ids: Sequence[int] | None,
    accessible_ids: Sequence[str],
    enc: Any = None,
) -> list[str] | None:
    """Docs actually present in the fitted prompt, without substring matching.

    Matches structured WM prefixes such as ``  - {id}:`` so ``d1`` is not
    counted as visible just because ``d10`` appears. Returns None when the
    prompt cannot be inspected (unknown), and [] when it was inspected and
    contains none of the accessible IDs.
    """
    if enc is None or not prompt_ids:
        return None
    hay = [int(x) for x in prompt_ids]
    claimed: set[int] = set()
    visible: list[str] = []
    for did in sorted((str(x) for x in accessible_ids if x), key=len, reverse=True):
        needles = [
            _encode_text_tokens(enc, f"  - {did}:"),
            _encode_text_tokens(enc, f"  - {did} "),
            _encode_text_tokens(enc, f'"{did}"'),
            _encode_text_tokens(enc, f"'{did}'"),
            _encode_text_tokens(enc, f"[{did}]"),
        ]
        hit = None
        for needle in needles:
            hit = _find_token_span(hay, needle, claimed)
            if hit is not None:
                break
        if hit is None:
            continue
        start, end = hit
        claimed.update(range(start, end))
        visible.append(did)
    return visible


def _meta_doc_ids(meta: Mapping[str, Any], key: str) -> list[str] | None:
    if key not in meta:
        return None
    raw = meta.get(key)
    if raw is None:
        return None
    return [str(x) for x in list(raw)]


def visible_doc_ids_for_decision(
    *,
    snapshot: Any = None,
    prompt_ids: Sequence[int] | None = None,
    enc: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[str] | None:
    """Docs the Student can actually condition on after context clipping.

    ``None`` means unknown / not collected. An empty list means the fitted
    prompt has no visible document IDs. Never confuse the two with ``or``.
    """
    meta = dict(metadata or {})
    if "prompt_visible_doc_ids" in meta:
        recorded = _meta_doc_ids(meta, "prompt_visible_doc_ids")
        return recorded if recorded is not None else []
    if "visible_doc_ids" in meta:
        recorded = _meta_doc_ids(meta, "visible_doc_ids")
        if recorded is not None:
            return recorded
    accessible = _meta_doc_ids(meta, "accessible_doc_ids")
    if accessible is None:
        accessible = visible_doc_ids_from_snapshot(snapshot)
    if not accessible:
        return None
    inspected = prompt_visible_doc_ids_from_prompt(
        prompt_ids=prompt_ids,
        accessible_ids=accessible,
        enc=enc,
    )
    if inspected is not None:
        return inspected
    return None


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


def _termination_for(enc: Any) -> tuple[list[int], str]:
    if enc is None:
        return [], "none"
    if is_harmony_encoder(enc):
        from trim.eval.harmony_hf_encoding import HARMONY_CALL_ID

        return [int(HARMONY_CALL_ID)], "call"
    stops = [int(x) for x in (getattr(enc, "stop_token_ids", None) or [])]
    if stops:
        return [stops[0]], "eos"
    return [], "none"


@dataclass
class SupervisedActionEncoding:
    target_ids: list[int]
    target_text: str
    canonical_action: dict[str, Any]
    supervision_mask: list[bool]
    termination_kind: str = "none"

    def __iter__(self):
        yield self.target_ids
        yield self.target_text


def encode_supervised_action(
    *,
    target_action: Mapping[str, Any] | None,
    target_text: str,
    encode: Any,
    model_enc: Any | None = None,
) -> SupervisedActionEncoding:
    """Encode a* with the rollout tool-call format when a model encoder exists.

    Returns target ids, the already-validated canonical action, a mask built
    after the final encoding, and the protocol termination kind.
    """
    canon = canonicalize_action(target_action) if target_action else {}
    if model_enc is not None and canon:
        ids, text = encode_rollout_style_action(model_enc, dict(canon))
        term_ids, term_kind = _termination_for(model_enc)
        if term_kind == "eos" and term_ids and ids and int(ids[-1]) not in term_ids:
            ids = list(ids) + [term_ids[0]]
        assert_supervised_action_matches(
            model_enc,
            target_action=dict(canon),
            target_token_ids=ids,
        )
        mask = [True] * len(ids)
        return SupervisedActionEncoding(
            target_ids=list(ids),
            target_text=text,
            canonical_action=dict(canon),
            supervision_mask=mask,
            termination_kind=term_kind,
        )
    text = str(target_text or "")
    ids = list(encode(text)) if text else []
    return SupervisedActionEncoding(
        target_ids=ids,
        target_text=text,
        canonical_action=dict(canon),
        supervision_mask=[True] * len(ids),
        termination_kind="none",
    )


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
    "SupervisedActionEncoding",
    "assert_action_visible",
    "assert_supervised_action_matches",
    "decode_token_ids",
    "encode_rollout_style_action",
    "encode_rollout_style_prompt",
    "encode_supervised_action",
    "prompt_ids_hash",
    "prompt_visible_doc_ids_from_prompt",
    "render_harmony_tool_call_completion",
    "render_rollout_action_text",
    "resolve_effective_weight",
    "resolve_student_prompt_ids",
    "validate_finite_weight",
    "visible_doc_ids_for_decision",
    "visible_doc_ids_from_snapshot",
]
