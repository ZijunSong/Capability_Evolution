"""Stop TRIM from silently training as RL when projected-gap rows disappear."""

from __future__ import annotations

from typing import Any

from trim.training.rl_opd_types import (
    UPDATE_OPD_ONLY_ZERO_RL,
    UPDATE_SKIPPED,
    uses_projected_seed,
)
from trim.training.tinker_rl_opd_trainer import classify_update_type


def effective_update_label(*, n_rl: int, n_opd: int) -> str:
    """Short label persisted each step. Configured TRIM is not this label."""
    kind = classify_update_type(n_rl=int(n_rl), n_opd=int(n_opd))
    if kind == UPDATE_OPD_ONLY_ZERO_RL:
        return "opd_only"
    if kind == UPDATE_SKIPPED:
        return "skipped"
    return str(kind)


def flatten_projection_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(stats or {})
    build = dict(src.get("build_stats") or {})
    recovery = dict(src.get("teacher_recovery") or {})

    def pick(*keys: str, default: int = 0) -> int:
        for key in keys:
            if src.get(key) is not None:
                return int(src.get(key) or 0)
            if build.get(key) is not None:
                return int(build.get(key) or 0)
        return int(default)

    return {
        "n_decision_points": pick("n_decision_points"),
        "n_structurally_valid": pick("n_structurally_valid"),
        "n_sampled_decision_points": pick("n_sampled_decision_points"),
        "n_attempt": pick("n_attempt"),
        "n_untriggered": pick("n_untriggered"),
        "n_unregistered": pick("n_unregistered"),
        "n_unrealizable": pick("n_unrealizable"),
        "n_projected_training_steps": pick("n_projected_training_steps"),
        "n_skip_missing_teacher": pick("n_skip_missing_teacher"),
        "n_skip_missing_student": pick("n_skip_missing_student"),
        "n_skip_zero_mask": pick("n_skip_zero_mask"),
        "n_skip_empty_target": pick("n_skip_empty_target"),
        "n_kept": pick("n_kept"),
        "missing_teacher_reasons": dict(src.get("missing_teacher_reasons") or build.get("missing_teacher_reasons") or {}),
        "teacher_recovery": recovery,
    }


def assess_opd_health(
    *,
    lambda_opd: float,
    opd_loss: str,
    n_rl: int,
    n_opd: int,
    projection_stats: dict[str, Any] | None,
    opd_empty_streak: int,
    max_empty_opd: int,
) -> dict[str, Any]:
    """Fail closed when projected steps were built and then all lost their teacher context.

    A teacher that truly does not trigger may produce a few empty OPD batches.
    That uses its own streak and does not reuse the RL+OPD ``empty_streak``.
    """
    fields = flatten_projection_stats(projection_stats)
    label = effective_update_label(n_rl=n_rl, n_opd=n_opd)
    streak = int(opd_empty_streak)
    reasons: list[str] = []
    expects_opd = float(lambda_opd) > 0.0
    if not expects_opd:
        return {
            "fatal": False,
            "message": "",
            "effective_update_type": label,
            "opd_empty_streak": streak,
            "fields": fields,
        }

    n_steps = int(fields["n_projected_training_steps"])
    n_skip_teacher = int(fields["n_skip_missing_teacher"])
    n_kept = int(fields["n_kept"])
    if (
        uses_projected_seed(opd_loss)
        and n_steps > 0
        and int(n_opd) <= 0
        and n_kept <= 0
        and n_skip_teacher >= n_steps
    ):
        reasons.append(
            "projected steps were built and then all dropped because teacher context "
            f"was missing (n_projected_training_steps={n_steps}, "
            f"n_skip_missing_teacher={n_skip_teacher}, reasons={fields['missing_teacher_reasons']}, "
            f"teacher_recovery={fields['teacher_recovery']}). "
            "Refusing to load the actor for an RL-only update."
        )

    if int(n_opd) <= 0:
        streak += 1
        limit = max(1, int(max_empty_opd))
        if streak >= limit:
            reasons.append(
                f"OPD datums were empty for {streak} consecutive batches "
                f"(limit {limit}, opd_loss={opd_loss}, lambda_opd={float(lambda_opd)}). "
                "An untriggered teacher may explain a short gap; this run is no longer a TRIM update."
            )
    else:
        streak = 0

    return {
        "fatal": bool(reasons),
        "message": " ".join(reasons),
        "effective_update_type": label,
        "opd_empty_streak": streak,
        "fields": fields,
    }
