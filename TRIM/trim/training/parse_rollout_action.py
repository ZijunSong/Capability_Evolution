"""Parse sampled token text into an env action (T02). No torch import."""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.action_codec import HARNESS_G_STUDENT_NATIVE_TOOLS, STUDENT_NATIVE_TOOLS
from trim.training.tool_mask import legal_tool_names

_HG_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "select": ("sid",),
    "lookup": ("eid",),
    "answer_with": ("sid",),
}


def _schema_ok(name: str, args: dict[str, Any] | None) -> bool:
    req = _HG_REQUIRED_ARGS.get(name, ())
    if not req:
        return True
    if not args:
        return False
    return all(str(args.get(k) or "").strip() for k in req)


def _legal_names(
    harness_mask: dict[str, bool] | None,
    *,
    teacher_mode: bool,
    g_mode: bool,
) -> set[str]:
    extra = ["verify"] if teacher_mode or (harness_mask or {}).get("verify_tool") else None
    legal = set(legal_tool_names(extra=extra, harness_mask=harness_mask if not teacher_mode else None))
    if teacher_mode:
        legal |= set(STUDENT_NATIVE_TOOLS) | set(HARNESS_G_STUDENT_NATIVE_TOOLS) | {"verify"}
    elif g_mode:
        legal |= set(HARNESS_G_STUDENT_NATIVE_TOOLS)
        if (harness_mask or {}).get("answer_with"):
            legal.add("answer_with")
    return legal


def parse_generated_action(
    text: str,
    completion_ids: list[int] | None,
    enc,
    *,
    harness_mask: dict[str, bool] | None = None,
    teacher_mode: bool = False,
    action_map: Mapping[str, Mapping[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> tuple[dict[str, Any], bool]:
    from trim.adapters.harness_profiles import is_harness_g

    g_mode = is_harness_g(mask=harness_mask)
    legal = _legal_names(harness_mask, teacher_mode=teacher_mode, g_mode=g_mode)
    finish = str(finish_reason or "")

    if g_mode and finish == "length":
        return {"name": "truncated", "arguments": {"finish_reason": "length"}}, False

    if g_mode:
        from trim.eval.harness_g_runtime import parse_harness_g_action

        g_action, g_ok = parse_harness_g_action(
            text,
            action_map=action_map,
            finish_reason=finish_reason,
        )
        g_name = str(g_action.get("name") or "")
        g_args = dict(g_action.get("arguments") or {})
        if not g_ok or not _schema_ok(g_name, g_args) or g_name not in legal:
            return {"name": g_name or "unknown", "arguments": g_args}, False
        if enc is not None and hasattr(enc, "parse_tool_call"):
            parsed = enc.parse_tool_call(text, completion_ids=completion_ids)
            if (
                parsed.parsed
                and parsed.legal
                and parsed.error is None
                and parsed.tool_name
            ):
                native_name = str(parsed.tool_name)
                native_args = dict(parsed.arguments or {}) if parsed.arguments is not None else {}
                if native_name != g_name or native_args != g_args:
                    return {"name": "unknown", "arguments": {}}, False
        return g_action, True

    from trim.eval.harmony_runtime import parse_harmony_tool_call

    if enc is not None and hasattr(enc, "parse_tool_call"):
        parsed = enc.parse_tool_call(text, completion_ids=completion_ids)
    else:
        parsed = parse_harmony_tool_call(text, completion_ids=completion_ids, enc=enc)

    name = parsed.tool_name
    args = dict(parsed.arguments or {}) if parsed.arguments is not None else {}
    schema_ok = _schema_ok(str(name or ""), args)
    harmony_ok = (
        parsed.parsed
        and parsed.legal
        and name in legal
        and schema_ok
        and parsed.error is None
    )
    if harmony_ok:
        return {"name": name, "arguments": args}, True

    if finish == "length":
        return {"name": "truncated", "arguments": {"finish_reason": "length"}}, False

    fallback_name = name or "unknown"
    return {"name": fallback_name, "arguments": args}, False
