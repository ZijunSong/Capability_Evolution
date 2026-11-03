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
    # [TRIM_TEACHER_COMPONENTS] env-gated exact-coalition override (2026-10-26).
    # When set (comma list of canonical ids), Teacher runs with EXACTLY those
    # components ON and the rest OFF (eval-style mask), e.g. teacher=esp only.
    # Unset/empty = original derivation, unchanged (backwards compatible).
    import os
    _tc = os.environ.get("TRIM_TEACHER_COMPONENTS", "").strip()
    if _tc:
        _ids = [t.strip() for t in _tc.replace(";", ",").split(",") if t.strip()]
        _resolved = harness or infer_harness_from_ids(_ids)
        _on = set(_ids)
        return {cid: (cid in _on) for cid in all_component_ids(_resolved)}
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


def resolve_harness_pair(spec: object, bank: object | None = None) -> dict:
    """Legacy masks, or an explicit bank zero/all resolution."""
    from trim.harness_bank.resolution import resolve_harness_pair as _resolve

    return _resolve(spec, bank=bank)


def eval_mask_for_ids(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
    preset: str | None = None,
) -> dict[str, bool]:
    """Official eval mask: listed components ON, remaining advanced components OFF."""
    return actual_eval_mask(component_ids, harness=harness, preset=preset)


def add_pair_masks(
    component_ids: Sequence[str],
    *,
    harness: str | None = None,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Explicit Probe-C add pair masks: Student=H0 (zero), Teacher=H0+m.

    Diff between the two masks is exactly ``component_ids``. This does *not*
    change any legacy default: ``student_mask_for_ids`` / ``teacher_mask_for_ids``
    keep their Full-minus-m / Full semantics for historical reproduction.
    """
    resolved = harness or "Harness-1"
    ids = [str(c).strip() for c in component_ids if str(c).strip()]
    if not ids:
        raise ValueError("add_pair_masks: empty component_ids")
    known = set(all_component_ids(resolved))
    unknown = [c for c in ids if c not in known]
    if unknown:
        raise ValueError(f"unknown component id(s) {unknown} for {resolved}")
    student = zero_mask(resolved)
    teacher = zero_mask(resolved)
    for cid in ids:
        teacher[cid] = True
    return student, teacher
