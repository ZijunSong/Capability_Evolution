"""Align OPD training tokens with rollout chat/tool-call encoding."""

from __future__ import annotations

from typing import Any, Sequence

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.action_codec import render_action


def encode_rollout_style_action(enc: Any, action: dict[str, Any]) -> tuple[list[int], str]:
    """Encode a canonical action with the same tokenizer path rollout uses."""
    text = render_action(action)
    if enc is not None and hasattr(enc, "encode"):
        return list(enc.encode(text)), text
    if enc is not None and hasattr(enc, "tokenizer"):
        tok = enc.tokenizer
        return list(tok.encode(text, add_special_tokens=False)), text
    return list(text.encode("utf-8")) or [0], text


def encode_rollout_style_prompt(
    enc: Any,
    snapshot: EnvironmentSnapshot,
    *,
    component_id: str = "",
    acts: Sequence[tuple[Any, Any]] | None = None,
    wm_text: str = "",
) -> tuple[list[int], str]:
    """Build prompt token ids the same way batched rollout does."""
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids

    query = str(snapshot.query_text or (snapshot.working_memory or {}).get("query") or "")
    if enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
        if not acts:
            return list(enc.build_first_turn_prompt_ids(query)), query
        return list(
            enc.build_continuation_prompt_ids(query, actions_obs=list(acts), wm_text=wm_text)
        ), query
    if not acts:
        return build_first_turn_prompt_ids(query, enc=enc), query
    return build_continuation_prompt_ids(
        query, actions_obs=list(acts), wm_text=wm_text, enc=enc
    ), query
