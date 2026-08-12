"""A12: Hierarchical Rollback action factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


OPS = ("CONTINUE", "REPLAN", "ROLLBACK")


@dataclass
class HierPrediction:
    intervene: str  # CONTINUE | INTERVENE
    operation: str  # CONTINUE | REPLAN | ROLLBACK
    checkpoint_id: str | None = None
    executable: bool = True
    restore_ok: bool = True
    invariant_pass: bool = True


def stage1_intervene(operation: str) -> str:
    return "CONTINUE" if operation == "CONTINUE" else "INTERVENE"


def predict_hierarchical(
    state: dict[str, Any],
    *,
    variant: str,
    op_model: Callable[[dict[str, Any]], str],
    ckpt_ranker: Callable[[dict[str, Any]], str | None],
    oracle_operation: Callable[[dict[str, Any]], str] | None = None,
    oracle_checkpoint: Callable[[dict[str, Any]], str | None] | None = None,
    executability_check: Callable[[str | None], bool] | None = None,
) -> HierPrediction:
    if variant == "a12_flat_joint":
        op = op_model(state)
        ckpt = ckpt_ranker(state) if op == "ROLLBACK" else None
    elif variant == "a12_two_stage_operation_then_checkpoint":
        op = op_model(state)
        ckpt = ckpt_ranker(state) if op == "ROLLBACK" else None
    elif variant == "a12_operation_classifier_checkpoint_ranker":
        op = op_model(state)
        ckpt = ckpt_ranker(state) if op == "ROLLBACK" else None
    elif variant == "a12_operation_classifier_checkpoint_retriever":
        op = op_model(state)
        ckpt = ckpt_ranker(state) if op == "ROLLBACK" else None
    elif variant == "a12_oracle_operation_learned_checkpoint":
        if oracle_operation is None:
            raise ValueError("oracle_operation required")
        op = oracle_operation(state)
        ckpt = ckpt_ranker(state) if op == "ROLLBACK" else None
    elif variant == "a12_learned_operation_oracle_checkpoint":
        op = op_model(state)
        if oracle_checkpoint is None:
            raise ValueError("oracle_checkpoint required")
        ckpt = oracle_checkpoint(state) if op == "ROLLBACK" else None
    elif variant == "a12_oracle_operation_oracle_checkpoint":
        if oracle_operation is None or oracle_checkpoint is None:
            raise ValueError("both oracles required")
        op = oracle_operation(state)
        ckpt = oracle_checkpoint(state) if op == "ROLLBACK" else None
    else:
        raise ValueError(f"unknown A12 variant: {variant}")

    if op not in OPS:
        raise ValueError(f"invalid operation: {op}")
    exe = True
    restore = True
    if op == "ROLLBACK":
        if executability_check is not None:
            exe = bool(executability_check(ckpt))
        restore = exe and ckpt is not None
    return HierPrediction(
        intervene=stage1_intervene(op),
        operation=op,
        checkpoint_id=ckpt,
        executable=exe,
        restore_ok=restore,
        invariant_pass=restore if op == "ROLLBACK" else True,
    )


def offline_hard_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Gate before allowing 100q closed-loop. Returns pass/fail with reasons."""
    checks = {
        "operation_balanced_accuracy>=0.70": metrics.get("operation_type_balanced_accuracy", 0) >= 0.70,
        "CONTINUE_recall>=0.60": metrics.get("CONTINUE_recall", 0) >= 0.60,
        "checkpoint_acc>=0.50": metrics.get("checkpoint_selection_accuracy", 0) >= 0.50,
        "invalid_checkpoint<=0.01": metrics.get("invalid_checkpoint_rate", 1) <= 0.01,
        "restore_invariant>=0.99": metrics.get("post_action_invariant_pass_rate", 0) >= 0.99,
        "hf_vllm_parity==1": metrics.get("hf_vllm_operation_parity", 0) == 1,
        "seed_direction_consistent": bool(metrics.get("seed_direction_consistent", False)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "allow_100q_closed_loop": all(checks.values()),
    }
