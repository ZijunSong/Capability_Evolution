"""Student / Teacher / eval component masks without CLI or torch imports.

Eval ``--component`` is the live switch set. Training uses a different
interpretation: listed ids are Teacher privileges / Student removals.
"""

from __future__ import annotations

from typing import Sequence

from trim.adapters.components import (
    all_component_ids,
    coalition_minus_mask,
    default_component_ids,
    full_mask,
    zero_mask,
)
from trim.adapters.harness_profiles import infer_harness_from_ids
from trim.upstream_harness1.v8d_flags import actual_eval_mask, all_enabled_mask


def student_mask_for_ids(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
) -> dict[str, bool]:
    resolved = harness or infer_harness_from_ids(component_ids)
    if not component_ids:
        return zero_mask(resolved)
    return coalition_minus_mask(component_ids, harness=resolved)


def teacher_mask_for_ids(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
    preset: str | None = None,
) -> dict[str, bool]:
    """Training Teacher / privileged mask. Not used for official eval."""
    resolved = harness or infer_harness_from_ids(component_ids)
    if preset == "zero" or not component_ids:
        return zero_mask(resolved)
    if preset == "all" or list(component_ids) == list(all_component_ids(resolved)):
        return all_enabled_mask(resolved)
    mask = full_mask(resolved)
    for cid in component_ids:
        mask[cid] = True
    return mask


def teacher_mask_for_component(
    component_id: str | None,
    *,
    harness: str | None = None,
) -> dict[str, bool]:
    """DualView fallback when a snapshot did not store Teacher mask."""
    key = str(component_id or "zero").strip().lower()
    resolved = harness or infer_harness_from_ids([component_id] if component_id else [])
    if key in {"", "zero"}:
        return teacher_mask_for_ids([], harness=resolved, preset="zero")
    if key == "all":
        return teacher_mask_for_ids(list(all_component_ids(resolved)), harness=resolved, preset="all")
    if key == "default":
        return teacher_mask_for_ids(default_component_ids(resolved), harness=resolved, preset="default")
    return teacher_mask_for_ids([str(component_id)], harness=resolved)


def eval_mask_for_ids(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
    preset: str | None = None,
) -> dict[str, bool]:
    """Official eval mask: listed components ON, remaining advanced components OFF."""
    return actual_eval_mask(component_ids, harness=harness, preset=preset)
