"""Align OPD training tokens with rollout chat/tool-call encoding."""

from __future__ import annotations

import json
from typing import Any, Sequence

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.action_codec import canonicalize_action, render_action


def _is_harmony_family(family: str) -> bool:
    """True for Harmony-stack families (gpt-oss) whose tool-call surface form
    must use the canonical Harmony recipient+channel+call structure.

    render_action's ``to=<name>\\n{args}`` bare form is rejected by
    parse_harmony_tool_call ("no_executable_tool_call_region") for these
    families, so the OPD projected-action round-trip (encode -> decode ->
    parse_tool_call) must emit the same Harmony surface form the rollout
    sampler and canonical_examples() use.
    """
    from trim.eval.model_profiles import resolve_model_profile

    try:
        return bool(resolve_model_profile(family).is_harmony)
    except Exception:
        return family == "gpt-oss"


def render_rollout_action_text(enc: Any, action: dict[str, Any]) -> str:
    """Render an action in the same surface form rollout sampling uses."""
    canon = canonicalize_action(action)
    from trim.eval.model_profiles import is_hf_chat_family

    family = str(getattr(enc, "family", "") or "")
    if is_hf_chat_family(family):
        payload = {"name": canon["name"], "arguments": canon["arguments"]}
        return f"<tool>{json.dumps(payload, ensure_ascii=False)}</tool>"
    if _is_harmony_family(family):
        # Canonical Harmony tool-call surface form (see harmony_runtime.canonical_examples):
        # <|start|>assistant to=functions.<name><|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
        # render_action's bare "to=<name>\n{args}" is rejected by parse_harmony_tool_call
        # (no <|call|> executable region), so the OPD projected-action round-trip
        # (encode -> decode -> parse_tool_call) failed for gpt-oss. Emit the form
        # the Harmony parser accepts; this mirrors canonical_examples() exactly.
        args_json = json.dumps(canon["arguments"], ensure_ascii=False)
        return (
            f"<|start|>assistant to=functions.{canon['name']}"
            f"<|channel|>commentary <|constrain|>json"
            f"<|message|>{args_json}<|call|>"
        )
    return render_action(canon)


def encode_rollout_style_action(enc: Any, action: dict[str, Any]) -> tuple[list[int], str]:
    """Encode a canonical action with the same tokenizer path rollout uses."""
    text = render_rollout_action_text(enc, action)
    if enc is not None and hasattr(enc, "encode"):
        return list(enc.encode(text)), text
    if enc is not None and hasattr(enc, "tokenizer"):
        tok = enc.tokenizer
        return list(tok.encode(text, add_special_tokens=False)), text
    return list(text.encode("utf-8")) or [0], text


def _snapshot_acts_and_wm(snapshot: EnvironmentSnapshot) -> tuple[list[tuple[Any, Any]], str]:
    meta = dict(snapshot.metadata or {})
    acts = list(meta.get("rollout_actions_obs") or [])
    wm = str(meta.get("rollout_wm_text") or "")
    if acts:
        return acts, wm
    return [], wm


def _harmony_safe_actions_obs(
    acts: Sequence[tuple[Any, Any]] | None,
) -> list[tuple[Any, Any]]:
    """Drop Harness-G protocol feedback dicts; Harmony expects Action objects."""
    if not acts:
        return []
    try:
        from trim.eval.harness_g_runtime import is_protocol_feedback
    except ImportError:
        return list(acts)
    return [(a, o) for a, o in acts if not is_protocol_feedback(a)]


def encode_rollout_style_prompt(
    enc: Any,
    snapshot: EnvironmentSnapshot,
    *,
    component_id: str = "",
    acts: Sequence[tuple[Any, Any]] | None = None,
    wm_text: str = "",
) -> tuple[list[int], str]:
    """Build student prompt token ids the same way batched rollout does."""
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids
    from trim.training.opd_dataset import snapshot_query_text

    query = snapshot_query_text(snapshot)
    use_acts = _harmony_safe_actions_obs(
        list(acts) if acts is not None else _snapshot_acts_and_wm(snapshot)[0]
    )
    use_wm = wm_text or _snapshot_acts_and_wm(snapshot)[1]
    if enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
        if not use_acts:
            return list(enc.build_first_turn_prompt_ids(query)), query
        return list(
            enc.build_continuation_prompt_ids(query, actions_obs=use_acts, wm_text=use_wm)
        ), query
    if not use_acts:
        return build_first_turn_prompt_ids(query, enc=enc), query
    return build_continuation_prompt_ids(
        query, actions_obs=list(use_acts), wm_text=use_wm, enc=enc
    ), query


def encode_teacher_rollout_style_prompt(
    enc: Any,
    query: str,
    *,
    acts: Sequence[tuple[Any, Any]] | None = None,
    wm_text: str = "",
) -> tuple[list[int], str]:
    """Full-harness teacher prefix at the same decision timestep as the student."""
    use_acts = _harmony_safe_actions_obs(acts)
    if enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
        if not use_acts:
            return list(enc.build_first_turn_prompt_ids(query)), query
        return list(
            enc.build_continuation_prompt_ids(query, actions_obs=use_acts, wm_text=wm_text)
        ), query
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids

    if not use_acts:
        return build_first_turn_prompt_ids(query, enc=enc), query
    return build_continuation_prompt_ids(
        query, actions_obs=use_acts, wm_text=wm_text, enc=enc
    ), query


def assert_supervised_action_matches(
    enc: Any,
    *,
    target_action: dict[str, Any],
    target_token_ids: Sequence[int],
) -> None:
    """Decode supervised tokens and assert they match the projected action."""
    if enc is None or not target_token_ids:
        return
    if hasattr(enc, "decode_tokens"):
        text = enc.decode_tokens(list(target_token_ids))
    elif hasattr(enc, "tokenizer"):
        text = enc.tokenizer.decode(list(target_token_ids), skip_special_tokens=False)
    else:
        text = bytes(list(target_token_ids)).decode("utf-8", errors="replace")
    if hasattr(enc, "parse_tool_call"):
        parsed = enc.parse_tool_call(text, completion_ids=list(target_token_ids))
        if not parsed.parsed or not parsed.tool_name:
            raise ValueError(f"supervised tokens are not a legal tool call: {text[:200]!r}")
        got = canonicalize_action({"name": parsed.tool_name, "arguments": parsed.arguments or {}})
    else:
        from trim.training.action_codec import parse_action

        got = parse_action(text)
    want = canonicalize_action(target_action)
    if got != want:
        raise ValueError(f"supervised action mismatch: want={want} got={got}")
