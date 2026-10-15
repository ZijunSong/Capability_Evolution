"""Resolve bank zero/all without turning on the static ten-component mask."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from trim.adapters.components import zero_mask
from trim.adapters.role_masks import (
    eval_mask_for_ids,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from trim.harness_bank.loader import BankLoadError, load_bank
from trim.harness_bank.types import BankSpec, ResolvedHarness


def resolve_harness_pair(spec: Any, bank: BankSpec | None = None) -> dict[str, Any]:
    path = getattr(spec, "harness_bank", None)
    if not path and bank is None:
        harness = spec.harness
        components = tuple(spec.components)
        preset = spec.component_preset
        return {
            "mode": "legacy",
            "preset": preset,
            "bank_sha256": "",
            "active_components": [],
            "student_mask": student_mask_for_ids(components, harness=harness),
            "teacher_mask": teacher_mask_for_ids(components, harness=harness, preset=preset),
            "eval_mask": eval_mask_for_ids(components, harness=harness, preset=preset),
            "student_role": "legacy",
            "teacher_role": "legacy",
            "hooks_on_student": False,
            "hooks_on_teacher": False,
        }
    loaded = bank if bank is not None else load_bank(path)
    harness = getattr(spec, "harness", None) or "Harness-1"
    preset = spec.component_preset
    if preset not in {"zero", "all"}:
        raise BankLoadError("bank mode only accepts --component zero or --component all")
    student = zero_mask(harness)
    teacher = _legacy_mask(harness, loaded)
    if preset == "zero":
        resolved = ResolvedHarness(
            mode="bank",
            preset="zero",
            bank_sha256=loaded.bank_sha256,
            active_components=(),
            student_mask=student,
            teacher_mask=dict(student),
            eval_mask=dict(student),
            student_role="zero",
            teacher_role="zero",
            hooks_on_student=False,
            hooks_on_teacher=False,
        )
    else:
        resolved = ResolvedHarness(
            mode="bank",
            preset="all",
            bank_sha256=loaded.bank_sha256,
            active_components=tuple(loaded.active_components),
            student_mask=student,
            teacher_mask=teacher,
            eval_mask=dict(teacher),
            student_role="student",
            teacher_role="teacher",
            hooks_on_student=False,
            hooks_on_teacher=bool(loaded.active_components),
        )
    return {
        "mode": resolved.mode,
        "preset": resolved.preset,
        "bank_sha256": resolved.bank_sha256,
        "active_components": list(resolved.active_components),
        "student_mask": resolved.student_mask,
        "teacher_mask": resolved.teacher_mask,
        "eval_mask": resolved.eval_mask,
        "student_role": resolved.student_role,
        "teacher_role": resolved.teacher_role,
        "hooks_on_student": resolved.hooks_on_student,
        "hooks_on_teacher": resolved.hooks_on_teacher,
        "empty_bank": resolved.empty_bank,
        "bank": loaded,
    }


def _legacy_mask(harness: str, bank: BankSpec) -> dict[str, bool]:
    mask = zero_mask(harness)
    for key, value in bank.legacy_flags.items():
        if key not in mask:
            raise BankLoadError(f"legacy flag {key} is not in the {harness} static mask")
        mask[key] = bool(value)
    return mask


def apply_bank_launch(args: Any, spec: Any) -> Any:
    """Return a LaunchSpec whose static component list does not expand bank `all`."""
    from trim.cli.launch import LaunchError

    path = getattr(args, "harness_bank", None) or None
    if path in {"", None}:
        path = None
    base_contract = getattr(args, "base_contract", None)
    rsi_round = getattr(args, "rsi_round", None)
    init_manifest = getattr(args, "init_policy_manifest", None)
    if path is None:
        return replace(
            spec,
            harness_bank=None,
            base_contract=None if not base_contract else str(base_contract),
            rsi_round=None if not rsi_round else str(rsi_round),
            init_policy_manifest=None if not init_manifest else str(init_manifest),
            bank_sha256=None,
        )
    preset = spec.component_preset
    if preset not in {"zero", "all"}:
        raise LaunchError("with --harness-bank, --component must be exactly zero or all")
    try:
        loaded = load_bank(path)
    except BankLoadError as exc:
        raise LaunchError(str(exc)) from exc
    if getattr(args, "train_method", None) or spec.train_method:
        if preset != "all":
            raise LaunchError(
                "--harness-bank training requires --component all so the teacher is H0+B"
            )
        if not loaded.active_components:
            raise LaunchError("an empty bank cannot enter TRIM teacher training")
        backend = str(getattr(args, "training_backend", "hf_debug") or "hf_debug")
        if backend in {"verl", "fsdp2", "verl_fsdp2"}:
            raise LaunchError(
                f"training backend {backend} does not consume bank OPD fields; refusing to start"
            )
    static_ids = tuple(cid for cid, enabled in loaded.legacy_flags.items() if enabled)
    tag = loaded.bank_sha256[:12]
    out = spec.out
    if getattr(args, "out", None) is None:
        out = spec.out.parent / f"{spec.out.name}_bank{tag}"
    return replace(
        spec,
        components=static_ids,
        out=out,
        harness_bank=str(loaded.root),
        base_contract=None if not base_contract else str(base_contract),
        rsi_round=None if not rsi_round else str(rsi_round),
        init_policy_manifest=None if not init_manifest else str(init_manifest),
        bank_sha256=loaded.bank_sha256,
    )


def masks_from_args(args: Any, *, teacher_mode: bool) -> dict[str, bool] | None:
    if not getattr(args, "harness_bank", None):
        return None
    if teacher_mode:
        mask = getattr(args, "resolved_teacher_mask", None)
    else:
        official = bool(getattr(args, "official_eval", False))
        if official and getattr(args, "component", None) == "all":
            mask = getattr(args, "resolved_eval_mask", None)
        elif official:
            mask = getattr(args, "resolved_student_mask", None)
        else:
            mask = getattr(args, "resolved_student_mask", None)
    if mask is None:
        return None
    return dict(mask)
