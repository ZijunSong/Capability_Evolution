"""Dispatch ablation ExperimentSpec to smoke / offline runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.ablations.builders.build_field_ablation import analyze_field_ablation
from experiments.ablations.builders.build_rollback_hierarchy import (
    offline_hard_gate,
    predict_hierarchical,
)
from experiments.ablations.builders.build_state_source import build_and_report
from experiments.ablations.builders.build_supervision_source import build_from_paths
from experiments.ablations.builders.build_verification_ablation import (
    apply_gates,
    flags_for_variant,
)
from experiments.ablations.builders.fallback_router import (
    RouterTelemetry,
    make_policy,
    route_decision,
)
from experiments.common.spec import ExperimentSpec
from inference.scope.eval_common import (
    classification_metrics,
    dup_closed_loop_metrics,
    rollback_metrics,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _smoke_episodes(n: int, seed: int) -> list[dict[str, Any]]:
    eps = []
    for i in range(n):
        gold = "KEEP_EVIDENCE" if (i + seed) % 2 == 0 else "SKIP_DUPLICATE"
        pred = gold if i % 3 != 0 else ("SKIP_DUPLICATE" if gold == "KEEP_EVIDENCE" else "KEEP_EVIDENCE")
        eps.append(
            {
                "query_id": f"smoke_q{i}",
                "recall": 1.0 if gold == pred else 0.0,
                "trajectory_recall": 1.0 if gold == pred else 0.0,
                "final_answer_recall": 1.0 if gold == pred else 0.0,
                "reward": 1.0 if gold == pred else 0.0,
                "turns": 5 + i,
                "tool_calls": 3,
                "dup_curate_rate": 0.1,
                "n_curated": 4,
                "errors": [],
                "valid_decision_count": 2,
                "decisions": [{"gold": gold, "pred": pred}],
            }
        )
    return eps


def run_a1(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    n = spec.smoke_query_limit or 16
    result = build_from_paths(
        spec.variant,
        n_target=max(n, 16),
        output_path=out / "predictions.jsonl",
    )
    _write_jsonl(out / "telemetry.jsonl", [{"event": "a1_build", **result["report"]}])
    return {
        "metrics": result["report"],
        "n_queries": result["report"]["n_samples"],
        "errors": [],
        "status": "smoke_pipeline",
    }


def run_a2(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    n = spec.smoke_query_limit or 16
    diag = build_and_report(spec.variant, out, n=n, seed=spec.seed)
    _write_jsonl(out / "predictions.jsonl", [{"diagnostics": True}])
    _write_jsonl(out / "telemetry.jsonl", [diag])
    return {"metrics": diag, "n_queries": diag["unique_query_count"], "errors": [], "status": "smoke_pipeline"}


def run_a4(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    flags = flags_for_variant(spec.variant)
    candidates = []
    for i in range(spec.smoke_query_limit or 8):
        candidates.append(
            {
                "id": f"c{i}",
                "visibility_ok": i % 5 != 0,
                "schema_ok": i % 7 != 0,
                "executable": i % 6 != 0,
                "mutation_ok": i % 8 != 0,
                "verified": i % 4 != 0,
                "route": "ENDORSE" if i % 2 == 0 else "CORRECT",
                "action": "KEEP_EVIDENCE",
            }
        )

    def hard(c: dict[str, Any]) -> bool:
        # Outer ActionRealizer hard constraint — never disabled.
        return c.get("action") not in (None, "DANGEROUS", "NONEXISTENT")

    kept, tel = apply_gates(candidates, flags, hard_realizer_check=hard)
    _write_jsonl(out / "predictions.jsonl", kept)
    _write_jsonl(out / "telemetry.jsonl", [tel.to_dict()])
    return {
        "metrics": tel.to_dict(),
        "n_queries": len(candidates),
        "errors": [],
        "status": "smoke_pipeline",
    }


def run_a7(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    rows = []
    for i in range(spec.smoke_query_limit or 16):
        rows.append(
            {
                "label": "KEEP_EVIDENCE" if i % 2 == 0 else "SKIP_DUPLICATE",
                "state": {
                    "query_id": f"q{i // 2}",
                    "candidate_text": f"text {i % 3}",
                    "candidate_id": f"c{i % 3}",
                    "curated_history": [f"h{i}"],
                    "dup_statistics": {"n": i},
                    "query": f"question {i // 4}",
                    "query_context": f"ctx {i // 4}",
                    "checkpoint_registry": {"ck": i},
                    "checkpoint_metadata": {"m": i},
                    "recovery_budget": 3,
                    "failure_history": [],
                    "previous_operation": "CONTINUE",
                    "candidate_checkpoint_id": f"ck{i % 2}",
                },
            }
        )
    report = analyze_field_ablation(rows, variant=spec.variant)
    _write_jsonl(out / "predictions.jsonl", rows)
    _write_jsonl(out / "telemetry.jsonl", [report])
    return {"metrics": report, "n_queries": len(rows), "errors": [], "status": "smoke_pipeline"}


def run_a10(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    policy = make_policy(spec.variant)
    tel = RouterTelemetry()

    def internalized(state):
        return {"operation": "KEEP_EVIDENCE", "confidence": float(state.get("confidence", 0.4))}

    def module(state):
        return {"operation": "SKIP_DUPLICATE", "confidence": 1.0, "from_module": True}

    preds = []
    for i in range(spec.smoke_query_limit or 8):
        state = {"confidence": 0.3 if i % 2 == 0 else 0.9, "i": i}
        d = route_decision(
            state,
            internalized_policy=internalized,
            runtime_module=module,
            fallback_policy=policy,
            telemetry=tel,
            latency_ms=1.0,
        )
        preds.append({"query_id": f"q{i}", **d})
    _write_jsonl(out / "predictions.jsonl", preds)
    _write_jsonl(out / "telemetry.jsonl", [tel.to_dict()])
    return {"metrics": tel.to_dict(), "n_queries": len(preds), "errors": [], "status": "smoke_pipeline"}


def run_a12(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    rows = []
    preds = []
    for i in range(spec.smoke_query_limit or 8):
        gold_op = ["CONTINUE", "REPLAN", "ROLLBACK"][i % 3]
        state = {
            "query_id": f"q{i}",
            "gold_operation": gold_op,
            "gold_checkpoint_id": f"ck{i}" if gold_op == "ROLLBACK" else None,
            "candidates": [f"ck{i}", f"ck{i+1}"],
        }

        def op_model(s, _i=i):
            # imperfect learned op
            return ["CONTINUE", "REPLAN", "ROLLBACK"][(_i + 1) % 3]

        def ckpt_ranker(s):
            return (s.get("candidates") or [None])[0]

        def oracle_op(s):
            return s["gold_operation"]

        def oracle_ck(s):
            return s.get("gold_checkpoint_id")

        hp = predict_hierarchical(
            state,
            variant=spec.variant,
            op_model=op_model,
            ckpt_ranker=ckpt_ranker,
            oracle_operation=oracle_op,
            oracle_checkpoint=oracle_ck,
            executability_check=lambda c: c is not None,
        )
        row = {
            "gold_operation": gold_op,
            "pred_operation": hp.operation,
            "gold_checkpoint_id": state.get("gold_checkpoint_id"),
            "pred_checkpoint_id": hp.checkpoint_id,
            "checkpoint_executable": hp.executable,
            "restore_success": hp.restore_ok,
            "recovery_budget_violation": False,
            "post_action_invariant_pass": hp.invariant_pass,
            "task_recall": 1.0 if hp.operation == gold_op else 0.0,
            "reward": 1.0 if hp.operation == gold_op else 0.0,
        }
        rows.append(row)
        preds.append(row)
    metrics = rollback_metrics(rows)
    metrics["hf_vllm_operation_parity"] = 1
    metrics["seed_direction_consistent"] = True
    gate = offline_hard_gate(metrics)
    _write_jsonl(out / "predictions.jsonl", preds)
    _write_jsonl(out / "telemetry.jsonl", [{"gate": gate}])
    return {
        "metrics": {**metrics, "hard_gate": gate},
        "n_queries": len(rows),
        "errors": [],
        "status": "smoke_pipeline",
    }


def run_generic_smoke(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    n = spec.smoke_query_limit or 4
    eps = _smoke_episodes(n, spec.seed)
    metrics = dup_closed_loop_metrics(eps)
    # also emit classification sanity on decisions
    gold = [d["gold"] for e in eps for d in e["decisions"]]
    pred = [d["pred"] for e in eps for d in e["decisions"]]
    metrics["offline_classification"] = classification_metrics(gold, pred)
    _write_jsonl(out / "predictions.jsonl", eps)
    _write_jsonl(out / "telemetry.jsonl", [{"event": "generic_smoke", "variant": spec.variant}])
    return {
        "metrics": metrics,
        "n_queries": n,
        "errors": [],
        "status": "smoke_pipeline",
        "notes": f"generic smoke for {spec.group}/{spec.variant}; not a research conclusion",
    }


def run_a6(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    """Validate A6 objective/capacity wiring without full GPU training."""
    from training.scope.classification_head import (
        ClassificationHeadConfig,
        DupOperationClassificationHead,
        trainable_parameter_count,
    )
    from training.scope.losses import LossMode
    from training.scope.operation_objectives import objective_math_description

    mode = spec.objective or spec.extras.get("loss_mode") or "discriminative_ce"
    # Map compact/full token aliases used in registry
    alias = {
        "sample_normalized_action_ce": LossMode.SAMPLE_NORMALIZED_ACTION_CE.value,
        "legacy_token_ce": LossMode.LEGACY_TOKEN_CE.value,
        "discriminative_ce": LossMode.DISCRIMINATIVE_CE.value,
        "operation_ce": LossMode.OPERATION_CE.value,
        "pairwise_margin": LossMode.PAIRWISE_MARGIN.value,
        "classification_head": LossMode.CLASSIFICATION_HEAD.value,
        "sequence_ce_plus_operation": LossMode.SEQUENCE_CE_PLUS_OPERATION.value,
    }
    loss_mode = alias.get(mode, mode)
    try:
        LossMode(loss_mode)
    except ValueError as exc:
        raise ValueError(f"A6 unknown loss_mode={loss_mode}") from exc

    head_params = 0
    if loss_mode == LossMode.CLASSIFICATION_HEAD.value:
        # Hidden size proxy for Qwen2.5-7B; real trainer resolves from model.config.
        head = DupOperationClassificationHead(ClassificationHeadConfig(hidden_size=3584))
        head_params = trainable_parameter_count([head])["trainable_parameters"]

    flops = float(
        (spec.max_steps or 100)
        * spec.effective_batch_size
        * spec.max_tokens
        * spec.lora_rank
    )
    metrics = {
        "loss_mode": loss_mode,
        "lora_rank": spec.lora_rank,
        "lora_alpha": spec.lora_alpha,
        "max_steps": spec.max_steps,
        "optimizer_steps_proxy": spec.max_steps or 100,
        "token_flops_proxy": flops,
        "classification_head_trainable_params": head_params,
        "objective_math": objective_math_description(loss_mode),
        "kernel_ready": True,
    }
    _write_jsonl(out / "predictions.jsonl", [{"variant": spec.variant, **metrics}])
    _write_jsonl(out / "telemetry.jsonl", [{"event": "a6_kernel_smoke", **metrics}])
    return {
        "metrics": metrics,
        "n_queries": spec.smoke_query_limit or 0,
        "errors": [],
        "status": "smoke_pipeline",
        "notes": "A6 kernel smoke — validates loss_mode wiring; not a training result",
    }


DISPATCH = {
    "a1_supervision_source": run_a1,
    "a2_state_source": run_a2,
    "a4_verification_gate": run_a4,
    "a6_objective": run_a6,
    "a6_capacity": run_a6,
    "a7_decision_state_fields": run_a7,
    "a10_module_retirement": run_a10,
    "a12_rollback_hierarchy": run_a12,
}


def dispatch_ablation(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    fn = DISPATCH.get(spec.group, run_generic_smoke)
    result = fn(spec, out)
    return {
        "schema_version": "iclr_summary_v1",
        "experiment_id": spec.experiment_id,
        "status": result.get("status", "completed"),
        "metrics": result.get("metrics", {}),
        "n_queries": int(result.get("n_queries", 0)),
        "errors": result.get("errors", []),
        "notes": result.get("notes", ""),
    }
