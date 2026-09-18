"""Align OPD training tokens with rollout chat/tool-call encoding."""

from __future__ import annotations

import json
from typing import Any, Sequence

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.action_codec import canonicalize_action, render_action

HARMONY_CALL_TOKEN = "<|call|>"


def _family_of(enc: Any) -> str:
    return str(getattr(enc, "family", "") or "").lower()


def is_harmony_encoder(enc: Any) -> bool:
    family = _family_of(enc)
    if family in {"gpt-oss", "gptoss", "harmony", "o200k_harmony", "harness-1", "harness1"}:
        return True
    from trim.eval.model_profiles import is_hf_chat_family

    if is_hf_chat_family(family):
        return False
    stack = str(getattr(enc, "stack", "") or "").lower()
    return stack in {"harmony", "o200k_harmony"}


def render_harmony_tool_call_completion(action: dict[str, Any]) -> str:
    """Completion suffix after the Harmony assistant prefix.

    Matches ``to=functions.<name><|channel|>commentary … <|call|>`` so the
    strict parser's text branch accepts it. Do not prepend ``<|start|>assistant``.
    """
    canon = canonicalize_action(action)
    args = json.dumps(canon["arguments"], ensure_ascii=False)
    return (
        f"to=functions.{canon['name']}<|channel|>commentary "
        f"<|constrain|>json<|message|>{args}{HARMONY_CALL_TOKEN}"
    )


def render_rollout_action_text(enc: Any, action: dict[str, Any]) -> str:
    """Render an action in the same surface form rollout sampling uses."""
    canon = canonicalize_action(action)
    from trim.eval.model_profiles import is_hf_chat_family

    family = _family_of(enc)
    if is_hf_chat_family(family):
        payload = {"name": canon["name"], "arguments": canon["arguments"]}
        return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"
    if is_harmony_encoder(enc):
        return render_harmony_tool_call_completion(canon)
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


def _action_as_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        name = str(action.get("name") or action.get("tool") or "")
        arguments = dict(action.get("arguments") or {})
        return {"name": name, "arguments": arguments}
    tools = getattr(action, "tools", None) or []
    params = getattr(action, "params", None) or []
    name = ""
    if tools:
        schema = getattr(tools[0], "tool_schema", None)
        name = str(getattr(schema, "name", None) or "")
    arguments: dict[str, Any] = {}
    if params and isinstance(params[0], dict):
        arguments = dict(params[0])
    return {"name": name, "arguments": arguments}


def _obs_as_text(obs: Any) -> str:
    if obs is None:
        return ""
    if isinstance(obs, str):
        return obs
    texts = getattr(obs, "observations", None)
    if isinstance(texts, (list, tuple)) and texts:
        return str(texts[0])
    return str(obs)


def serialize_rollout_actions_obs(acts: Sequence[tuple[Any, Any]] | None) -> list[list[Any]]:
    out: list[list[Any]] = []
    for item in acts or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        out.append([_action_as_dict(item[0]), _obs_as_text(item[1])])
    return out


def hydrate_rollout_actions_obs(raw: Sequence[Any] | None) -> list[tuple[Any, Any]]:
    if not raw:
        return []
    try:
        from trim.eval.harmony_runtime import make_action, make_observation
    except Exception:
        make_action = None
        make_observation = None
    out: list[tuple[Any, Any]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        action, obs = item[0], item[1]
        if isinstance(action, dict) and action.get("name") and make_action is not None:
            try:
                action = make_action(str(action.get("name")), dict(action.get("arguments") or {}))
                obs = make_observation(str(obs)) if make_observation is not None else obs
            except Exception:
                pass
        out.append((action, obs))
    return out


def attach_prompt_context(
    snapshot: EnvironmentSnapshot,
    *,
    acts: Sequence[tuple[Any, Any]] | None = None,
    wm_text: str = "",
    teacher_wm_text: str | None = None,
) -> EnvironmentSnapshot:
    """Store JSON-safe decision-time refs without changing content_hash."""
    snapshot.metadata["_rollout_actions_obs"] = serialize_rollout_actions_obs(acts)
    snapshot.metadata["_rollout_wm_text"] = str(wm_text or "")
    if teacher_wm_text is not None:
        snapshot.metadata["_teacher_wm_text"] = str(teacher_wm_text)
    return snapshot


def _snapshot_acts_and_wm(snapshot: EnvironmentSnapshot) -> tuple[list[tuple[Any, Any]], str]:
    meta = dict(snapshot.metadata or {})
    acts = meta.get("_rollout_actions_obs") or meta.get("rollout_actions_obs") or []
    wm = str(meta.get("_rollout_wm_text") or meta.get("rollout_wm_text") or "")
    hydrated = hydrate_rollout_actions_obs(list(acts) if acts else [])
    return hydrated, wm


def snapshot_teacher_wm_text(snapshot: EnvironmentSnapshot) -> str:
    meta = dict(snapshot.metadata or {})
    return str(meta.get("_teacher_wm_text") or meta.get("teacher_wm_text") or "")


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
    """Full-harness teacher prefix at the same decision timestep as the student.

    Empty action history still uses the continuation template when teacher WM
    is non-empty, so first-turn capability observations are not dropped.
    """
    use_acts = _harmony_safe_actions_obs(acts)
    has_wm = bool(str(wm_text or "").strip())
    if enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
        if not use_acts and not has_wm:
            return list(enc.build_first_turn_prompt_ids(query)), query
        return list(
            enc.build_continuation_prompt_ids(query, actions_obs=use_acts, wm_text=wm_text)
        ), query
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids

    if not use_acts and not has_wm:
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
