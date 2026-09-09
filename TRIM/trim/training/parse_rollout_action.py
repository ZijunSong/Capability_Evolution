"""Parse sampled token text into an env action (T02). No torch import."""

from __future__ import annotations

from typing import Any

from trim.training.action_codec import HARNESS_G_STUDENT_NATIVE_TOOLS, STUDENT_NATIVE_TOOLS
from trim.training.tool_mask import legal_tool_names


def parse_generated_action(
    text: str,
    completion_ids: list[int] | None,
    enc,
    *,
    harness_mask: dict[str, bool] | None = None,
    teacher_mode: bool = False,
) -> tuple[dict[str, Any], bool]:
    from trim.eval.harmony_runtime import parse_harmony_tool_call

    if enc is not None and hasattr(enc, "parse_tool_call"):
        parsed = enc.parse_tool_call(text, completion_ids=completion_ids)
    else:
        parsed = parse_harmony_tool_call(text, completion_ids=completion_ids, enc=enc)
    name = parsed.tool_name
    extra = ["verify"] if teacher_mode or (harness_mask or {}).get("verify_tool") else None
    legal = set(legal_tool_names(extra=extra, harness_mask=harness_mask if not teacher_mode else None))
    if teacher_mode:
        legal |= set(STUDENT_NATIVE_TOOLS) | set(HARNESS_G_STUDENT_NATIVE_TOOLS) | {"verify"}
    if parsed.parsed and name in legal:
        return {"name": name, "arguments": dict(parsed.arguments or {})}, True
    from trim.eval.harness_g_runtime import parse_harness_g_action

    g_action, g_ok = parse_harness_g_action(text)
    if g_ok and g_action.get("name") in legal:
        return g_action, True
    return {"name": name or g_action.get("name") or "unknown", "arguments": dict(parsed.arguments or {})}, False
