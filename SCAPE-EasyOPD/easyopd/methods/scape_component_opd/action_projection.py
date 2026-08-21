from __future__ import annotations

from typing import Any

from .types import ToolAction


def project_curated_delta(
    *,
    curated_ids_pre: list[str],
    curated_ids_post: list[str],
    visible_doc_ids: list[str],
) -> tuple[ToolAction | None, dict[str, Any]]:
    pre = [str(x) for x in curated_ids_pre]
    post = [str(x) for x in curated_ids_post]
    visible = {str(x) for x in visible_doc_ids}
    add_ids = [x for x in post if x not in pre]
    remove_ids = [x for x in pre if x not in post]
    visible_valid = all(x in visible for x in add_ids)
    remove_valid = all(x in pre for x in remove_ids)
    valid = visible_valid and remove_valid and bool(add_ids or remove_ids)
    audit = {
        "add_ids": add_ids,
        "remove_ids": remove_ids,
        "visible_valid": visible_valid,
        "remove_valid": remove_valid,
        "projection_valid": valid,
        "on_policy_state": True,
        "target_source": "harness_effect_projection",
    }
    if not valid:
        return None, audit
    return ToolAction("curate", {"add_ids": add_ids, "remove_ids": remove_ids}), audit


def validate_projected_action(action: ToolAction, *, visible_doc_ids: list[str], curated_ids_pre: list[str]) -> dict[str, Any]:
    if action.name != "curate":
        return {"valid": False, "reason": "UNSUPPORTED_PROJECTED_TOOL"}
    args = action.arguments
    visible = {str(x) for x in visible_doc_ids}
    cur = {str(x) for x in curated_ids_pre}
    add_ids = [str(x) for x in args.get("add_ids") or []]
    remove_ids = [str(x) for x in args.get("remove_ids") or []]
    invalid_add = [x for x in add_ids if x not in visible]
    invalid_remove = [x for x in remove_ids if x not in cur]
    return {
        "valid": not invalid_add and not invalid_remove,
        "invalid_add_ids": invalid_add,
        "invalid_remove_ids": invalid_remove,
    }
