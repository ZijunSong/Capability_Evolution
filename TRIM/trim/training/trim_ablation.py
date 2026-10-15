"""TRIM ablation identities. The train method is the only switch.

``only-opd`` / ``no-project`` / ``no-gate`` are not aliases of ``trim`` or
``opd``. Callers that are not one of these methods get ``None`` and keep
their existing RL, CE, and sampled-gap paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrimAblation:
    variant: str
    rl_enabled: bool
    projection_mode: str
    gate_mode: str


_PROFILES: dict[str, TrimAblation] = {
    "trim": TrimAblation("none", True, "full", "sigmoid"),
    "scape_seed": TrimAblation("none", True, "full", "sigmoid"),
    "scape+seed": TrimAblation("none", True, "full", "sigmoid"),
    "seed+opd": TrimAblation("none", True, "full", "sigmoid"),
    "only-opd": TrimAblation("only_opd", False, "full", "sigmoid"),
    "only_opd": TrimAblation("only_opd", False, "full", "sigmoid"),
    "no-project": TrimAblation("no_project", True, "native_only", "sigmoid"),
    "no_project": TrimAblation("no_project", True, "native_only", "sigmoid"),
    "no-gate": TrimAblation("no_gate", True, "full", "none"),
    "no_gate": TrimAblation("no_gate", True, "full", "none"),
}

NEW_TRIM_ABLATION_METHODS = frozenset({"only-opd", "no-project", "no-gate"})
ABLATION_CONTRACT_VERSION = "trim_ablation_v1"


def resolve_trim_ablation(method_or_mode: str) -> TrimAblation | None:
    key = str(method_or_mode or "").strip().lower().replace(" ", "")
    return _PROFILES.get(key)


def apply_ablation_fields(args: Any, method_or_mode: str) -> TrimAblation | None:
    """Overwrite ablation fields from the method. Stale namespace values do not win."""
    profile = resolve_trim_ablation(method_or_mode)
    if profile is None:
        return None
    args.rl_enabled = bool(profile.rl_enabled)
    args.projection_mode = str(profile.projection_mode)
    args.opd_gate_mode = str(profile.gate_mode)
    args.ablation_variant = str(profile.variant)
    return profile


def build_experiment_contract(
    *,
    train_method: str,
    training_mode: str,
    cell: str,
    harness: str,
    opd_loss: str,
    lambda_opd: float,
    opd_gate_beta: float,
    opd_states_per_trajectory: int,
    rl_reward_requested: str | None,
    rl_reward_effective: str,
    reward_protocol: str,
    student_mask: Mapping[str, Any] | None = None,
    teacher_mask: Mapping[str, Any] | None = None,
    max_turns: int | None = None,
    component: str | None = None,
) -> dict[str, Any]:
    profile = resolve_trim_ablation(training_mode) or resolve_trim_ablation(train_method)
    variant = "none" if profile is None else profile.variant
    rl_enabled = True if profile is None else bool(profile.rl_enabled)
    projection_mode = "full" if profile is None else profile.projection_mode
    gate_mode = "sigmoid" if profile is None else profile.gate_mode
    is_new = variant != "none"
    explicit_reward = str(rl_reward_effective) != "trim-legacy-v1"
    return {
        "contract_version": ABLATION_CONTRACT_VERSION if (is_new or explicit_reward) else "trim_legacy",
        "train_method": str(train_method),
        "training_mode": str(training_mode),
        "cell": str(cell),
        "harness": str(harness),
        "component": component,
        "student_harness": "H_zero" if is_new else None,
        "teacher_harness": "H_full" if is_new else None,
        "opd_loss": str(opd_loss),
        "rl_enabled": rl_enabled,
        "projection_mode": projection_mode,
        "skip_to_anchor_enabled": str(opd_loss) == "sr_opd_projected_gap",
        "gate_mode": gate_mode,
        "ablation_variant": variant,
        "lambda_opd": float(lambda_opd),
        "opd_gate_beta": float(opd_gate_beta),
        "opd_states_per_trajectory": int(opd_states_per_trajectory),
        "rl_reward_requested": rl_reward_requested,
        "rl_reward_effective": str(rl_reward_effective),
        "reward_protocol": str(reward_protocol),
        "reward_used_for_optimization": bool(rl_enabled),
        "max_turns": None if max_turns is None else int(max_turns),
        "student_mask": dict(student_mask or {}),
        "teacher_mask": dict(teacher_mask or {}),
        "reward_metric_contract": (
            "harness1_v3_defaults_on_trim_metrics_v1"
            if rl_reward_effective == "harness1-composite"
            else "trim_final_answer_recall_only_v1"
            if rl_reward_effective == "final-answer-recall"
            else "local_legacy_capped_shaping"
        ),
    }
