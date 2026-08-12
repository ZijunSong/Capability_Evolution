"""Config diff: reject undeclared drift between variants."""

from __future__ import annotations

from typing import Any

from experiments.common.spec import ExperimentSpec

# Fields that may differ even when changed_factor is narrow.
ALWAYS_ALLOWED = {
    "experiment_id",
    "variant",
    "changed_factor",
    "output_dir",
    "notes",
    "parent_experiment",
    "gpu",
    "dry_run",
    "resume",
    "smoke_query_limit",
    "expected_metrics",
    "checkpoint",  # may differ after training; checked separately when frozen
}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def config_diff(
    base: ExperimentSpec | dict[str, Any],
    other: ExperimentSpec | dict[str, Any],
) -> dict[str, Any]:
    a = base.to_dict() if isinstance(base, ExperimentSpec) else dict(base)
    b = other.to_dict() if isinstance(other, ExperimentSpec) else dict(other)
    fa, fb = flatten(a), flatten(b)
    keys = sorted(set(fa) | set(fb))
    changed: dict[str, dict[str, Any]] = {}
    for k in keys:
        if fa.get(k) != fb.get(k):
            changed[k] = {"base": fa.get(k), "other": fb.get(k)}
    return {
        "n_changed": len(changed),
        "changed": changed,
    }


def declared_change_keys(changed_factor: str) -> set[str]:
    """Map changed_factor tokens to allowed differing field prefixes."""
    factor = (changed_factor or "").strip().lower()
    mapping = {
        "supervision_source": {"shadow_source", "extras.supervision_source", "extras.state_source"},
        "state_source": {"shadow_source", "extras.state_source", "extras.state_distribution"},
        "target_format": {"target_format", "objective", "extras.target_format"},
        "verification_mode": {"verification_mode", "extras.gates"},
        "routing_mode": {"routing_mode", "extras.route_balance", "extras.route_filter"},
        "objective": {"objective", "extras.loss_mode"},
        "lora_rank": {"lora_rank", "lora_alpha", "max_steps", "extras.lora_rank"},
        "decision_state_fields": {"extras.field_mask", "extras.drop_fields", "decision_state_schema"},
        "contract_threshold": {"threshold", "extras.contract", "extras.decision_contract"},
        "data_scale": {"train_manifest", "extras.n_samples", "extras.query_fraction", "dataset"},
        "fallback_policy": {"extras.fallback_policy", "runtime_config", "extras.router"},
        "injection_distribution": {"extras.injection_ratio", "extras.train_split", "extras.test_split"},
        "rollback_hierarchy": {"extras.hierarchy", "extras.oracle_operation", "extras.oracle_checkpoint"},
        "label_noise": {"extras.label_noise", "extras.verifier_error"},
        "multi_capability": {"capability", "extras.capability_schedule", "extras.capability_weights"},
        "probe_mode": {"extras.probe_mode"},
        "baseline_method": {"method", "runtime_config", "shadow_source", "objective", "target_format"},
        "runtime_modules": {"runtime_config"},
        "prompt_hint": {"prompt_renderer", "extras.prompt_hint"},
        "none": set(),
        "seed": {"seed", "rollout_seed"},
    }
    allowed: set[str] = set()
    for token in factor.replace("+", ",").split(","):
        token = token.strip()
        if token in mapping:
            allowed |= mapping[token]
        elif token:
            allowed.add(token)
    return allowed


def assert_single_factor_diff(
    base: ExperimentSpec,
    other: ExperimentSpec,
    *,
    extra_allowed: set[str] | None = None,
) -> dict[str, Any]:
    """Fail if undeclared fields differ between base and other."""
    diff = config_diff(base, other)
    allowed = ALWAYS_ALLOWED | declared_change_keys(other.changed_factor)
    if extra_allowed:
        allowed |= extra_allowed
    undeclared: dict[str, Any] = {}
    for key, vals in diff["changed"].items():
        top = key.split(".", 1)[0]
        if key in allowed or top in allowed:
            continue
        # Allow extras.* when changed_factor mentions the suffix
        if key.startswith("extras."):
            suffix = key[len("extras.") :]
            if suffix in allowed or any(suffix.startswith(a) for a in allowed):
                continue
        undeclared[key] = vals
    if undeclared:
        raise ValueError(
            f"undeclared config drift for {other.experiment_id} "
            f"(changed_factor={other.changed_factor}): {sorted(undeclared)}"
        )
    return {**diff, "undeclared": undeclared, "ok": True}
