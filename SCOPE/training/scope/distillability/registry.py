"""Capability-level OFF/PROC/FULL registry for E0 distillability probes."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

from harness.capability.capability_id import CapabilityId, parse_capability_id, resolve_e0_capability
from harness.harness_config import HarnessConfig, config_path, load_harness_config
from training.scope.distillability.modes import DistillabilityMode

# Process-wide active probe state (single-threaded rollout per process).
_ACTIVE_PROBE: tuple[CapabilityId, DistillabilityMode] | None = None


@dataclass(frozen=True)
class CapabilityProbeSpec:
    capability_id: CapabilityId
    shadow_capability: CapabilityId
    shadow_module: str
    probe_supported: bool = True
    proc_supported: bool = True
    harness_patches_off: dict[str, dict[str, Any]] = field(default_factory=dict)
    env_patches_off: dict[str, str] = field(default_factory=dict)
    env_patches_full: dict[str, str] = field(default_factory=dict)
    notes: str = ""


_PROBE_SPECS: dict[CapabilityId, CapabilityProbeSpec] = {
    CapabilityId.DUPLICATE_EVIDENCE: CapabilityProbeSpec(
        capability_id=CapabilityId.DUPLICATE_EVIDENCE,
        shadow_capability=CapabilityId.DUPLICATE_EVIDENCE,
        shadow_module="evidence_state",
        harness_patches_off={"evidence_state": {"content_dedup": False}},
        notes="OFF disables runtime dedup; PROC uses EvidenceShadow duplicate rules.",
    ),
    CapabilityId.STOP_DECISION: CapabilityProbeSpec(
        capability_id=CapabilityId.STOP_DECISION,
        shadow_capability=CapabilityId.PREMATURE_STOP,
        shadow_module="verification",
        harness_patches_off={
            "verification": {
                "stop_budget_hint": False,
                "verification_aware_curation": False,
            },
            "context_budget": {"stop_budget_hint": False},
        },
        env_patches_off={
            "CHAT_MIN_TURNS_BEFORE_END": "0",
            "CHAT_MIN_CURATED_BEFORE_END": "0",
            "SCOPE_STOP_CALIBRATION": "0",
        },
        env_patches_full={
            "CHAT_MIN_TURNS_BEFORE_END": "8",
            "CHAT_MIN_CURATED_BEFORE_END": "1",
            "SCOPE_STOP_CALIBRATION": "1",
        },
        notes="OFF removes soft stop guards; hard max_turns always retained.",
    ),
    CapabilityId.EVIDENCE_CURATION: CapabilityProbeSpec(
        capability_id=CapabilityId.EVIDENCE_CURATION,
        shadow_capability=CapabilityId.IRRELEVANT_EVIDENCE,
        shadow_module="evidence_state",
        harness_patches_off={
            "evidence_state": {
                "importance_tagging": False,
                "subtractive_curation": False,
            }
        },
        notes="OFF disables importance/subtractive curation; WorkingMemory retained.",
    ),
    CapabilityId.VERIFICATION_DECISION: CapabilityProbeSpec(
        capability_id=CapabilityId.VERIFICATION_DECISION,
        shadow_capability=CapabilityId.PREMATURE_STOP,
        shadow_module="verification",
        harness_patches_off={
            "verification": {
                "verification_aware_curation": False,
                "render_records": False,
            },
            "context_budget": {"stop_budget_hint": False},
        },
        notes="Decision-only probe; external verifier outcomes separated.",
    ),
    CapabilityId.EXTERNAL_VERIFICATION: CapabilityProbeSpec(
        capability_id=CapabilityId.EXTERNAL_VERIFICATION,
        shadow_capability=CapabilityId.INVALID_CITATION,
        shadow_module="verification",
        harness_patches_off={"verification": {"expose_verify_tool": False}},
        notes="OFF disables external verifier; PROC reads existing records only.",
    ),
    CapabilityId.DETERMINISTIC_TRUNCATION: CapabilityProbeSpec(
        capability_id=CapabilityId.DETERMINISTIC_TRUNCATION,
        shadow_capability=CapabilityId.UNKNOWN,
        shadow_module="",
        probe_supported=True,
        proc_supported=False,
        harness_patches_off={"context_budget": {"deterministic_truncation": False}},
        notes="Runtime sanity check; PROC not applicable.",
    ),
}


def get_probe_spec(capability_id: str | CapabilityId) -> CapabilityProbeSpec:
    cap = parse_capability_id(capability_id)
    if cap not in _PROBE_SPECS:
        raise KeyError(f"Unknown E0 capability probe: {capability_id}")
    return _PROBE_SPECS[cap]


def list_e0_capabilities() -> list[CapabilityId]:
    return list(_PROBE_SPECS.keys())


def set_capability_mode(
    capability_id: str | CapabilityId,
    mode: DistillabilityMode | str,
) -> tuple[HarnessConfig, dict[str, str]]:
    """Return harness config + env overrides for a capability probe mode."""
    global _ACTIVE_PROBE
    cap = parse_capability_id(capability_id)
    if isinstance(mode, DistillabilityMode):
        probe_mode = mode
    else:
        probe_mode = DistillabilityMode(str(mode).lower().split(".")[-1])
    spec = get_probe_spec(cap)

    base_path = config_path("modules_full_v2.yaml")
    cfg = load_harness_config(base_path)

    env_overrides: dict[str, str] = {}

    if probe_mode == DistillabilityMode.FULL:
        env_overrides.update(spec.env_patches_full)
        _ACTIVE_PROBE = (cap, probe_mode)
        return cfg, env_overrides

    if probe_mode == DistillabilityMode.OFF:
        cfg = _apply_harness_patches(cfg, spec.harness_patches_off)
        env_overrides.update(spec.env_patches_off)
        _ACTIVE_PROBE = (cap, probe_mode)
        return cfg, env_overrides

    if probe_mode == DistillabilityMode.PROC:
        if not spec.proc_supported:
            raise ValueError(f"PROC not supported for {cap.value}")
        # PROC: harness keeps capability OFF; procedural logic injected at runtime.
        cfg = _apply_harness_patches(cfg, spec.harness_patches_off)
        env_overrides.update(spec.env_patches_off)
        # Block external verifier in PROC for external_verification probe.
        if cap == CapabilityId.EXTERNAL_VERIFICATION:
            ver = cfg.verification
            ver.options["expose_verify_tool"] = False
            ver.enabled = False
        _ACTIVE_PROBE = (cap, probe_mode)
        return cfg, env_overrides

    raise ValueError(f"Unknown distillability mode: {mode}")


def get_active_probe() -> tuple[CapabilityId, DistillabilityMode] | None:
    return _ACTIVE_PROBE


def clear_active_probe() -> None:
    global _ACTIVE_PROBE
    _ACTIVE_PROBE = None


def _apply_harness_patches(
    cfg: HarnessConfig,
    patches: dict[str, dict[str, Any]],
) -> HarnessConfig:
    if not patches:
        return cfg
    data = copy.deepcopy(cfg.to_dict())
    for module_name, opts in patches.items():
        module = data.setdefault(module_name, {})
        for key, value in opts.items():
            module[key] = value
    return _from_dict(data)


def _from_dict(data: dict[str, Any]) -> HarnessConfig:
    from harness.harness_config import _apply_options_to_module

    return HarnessConfig(
        retrieval=_apply_options_to_module(data.get("retrieval", {}), "retrieval"),
        evidence_state=_apply_options_to_module(
            data.get("evidence_state", {}), "evidence_state"
        ),
        verification=_apply_options_to_module(
            data.get("verification", {}), "verification"
        ),
        context_budget=_apply_options_to_module(
            data.get("context_budget", {}), "context_budget"
        ),
        recovery=_apply_options_to_module(data.get("recovery", {}), "recovery"),
    )


def apply_probe_env(env_overrides: dict[str, str]) -> None:
    for key, value in env_overrides.items():
        os.environ[key] = value


def shadow_capability_for(capability_id: str | CapabilityId) -> CapabilityId:
    spec = get_probe_spec(capability_id)
    return resolve_e0_capability(spec.shadow_capability.value)
