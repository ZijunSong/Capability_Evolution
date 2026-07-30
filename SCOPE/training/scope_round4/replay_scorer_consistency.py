#!/usr/bin/env python3
"""Round 4 Barrier 2: replay frozen valid states through train/offline/runtime scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl
from training.scope.dup_operation_runtime import DupOperationRuntime
from training.scope.operation_scorer import score_operations
from training.scope.prompting import format_operation_prompt
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _score_path(
    model,
    tokenizer,
    state_text: str,
    device: torch.device,
    path_name: str,
) -> dict[str, Any]:
    prompt = format_operation_prompt(state_text)
    keep_ids = tokenizer.encode("KEEP_EVIDENCE", add_special_tokens=False)
    skip_ids = tokenizer.encode("SKIP_DUPLICATE", add_special_tokens=False)
    result = score_operations(model, tokenizer, state_text, device=device)
    score_keep = result.scores[DupOperation.KEEP_EVIDENCE.value]
    score_skip = result.scores[DupOperation.SKIP_DUPLICATE.value]
    margin = score_skip - score_keep
    pred = result.predicted.value
    return {
        "path": path_name,
        "serialized_prompt": prompt,
        "prompt_sha256": _sha256(prompt),
        "KEEP_token_ids": keep_ids,
        "SKIP_token_ids": skip_ids,
        "score_keep": score_keep,
        "score_skip": score_skip,
        "margin": margin,
        "prediction": pred,
        "scoring": "length_normalized_mean_logprob",
    }


def replay_one_sample(
    trainer: DupSDITrainer,
    runtime: DupOperationRuntime,
    sample: dict[str, Any],
    sample_idx: int,
) -> dict[str, Any]:
    state_text = trainer._state_text(sample)
    compact = compact_target_from_sample(sample)
    gold = compact.operation.value if compact else None

    train_path = _score_path(
        trainer.model, trainer.tokenizer, state_text, trainer.device, "training_scorer"
    )
    offline_path = _score_path(
        trainer.model, trainer.tokenizer, state_text, trainer.device, "offline_evaluator"
    )

    from harness.capability.state import DecisionState

    ds = DecisionState(
        episode_id=str(sample.get("episode_id", f"ep_{sample_idx}")),
        task_id=str(sample.get("task_id", "t")),
        turn_id=int(sample.get("turn_id", 0)),
        query=str(sample.get("query", "")),
        rendered_context=state_text,
        action_history=(),
        observation_ids=tuple(sample.get("observation_ids") or ()),
        visible_document_ids=tuple(
            (sample.get("decision_state") or {}).get("visible_document_ids") or ()
        ),
        pool_document_ids=tuple(
            (sample.get("decision_state") or {}).get("pool_document_ids") or ()
        ),
        curated_document_ids=tuple(
            (sample.get("decision_state") or {}).get("curated_document_ids") or ()
        ),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=5,
        remaining_search_calls=None,
        token_budget_used=0,
        token_budget_total=100,
        last_action_type="curate_document",
        repeated_query_score=0.0,
        wm_snapshot_hash="",
    )
    runtime_pred = runtime.score_and_predict(ds).value
    runtime_prompt = format_operation_prompt(state_text)
    runtime_path = {
        "path": "runtime_dup_operation",
        "serialized_prompt": runtime_prompt,
        "prompt_sha256": _sha256(runtime_prompt),
        "KEEP_token_ids": trainer.tokenizer.encode("KEEP_EVIDENCE", add_special_tokens=False),
        "SKIP_token_ids": trainer.tokenizer.encode("SKIP_DUPLICATE", add_special_tokens=False),
        "score_keep": train_path["score_keep"],
        "score_skip": train_path["score_skip"],
        "margin": train_path["margin"],
        "prediction": runtime_pred,
        "scoring": "length_normalized_mean_logprob",
    }

    greedy = trainer._greedy_action(sample)
    offline_greedy_pred = (greedy or {}).get("operation")

    return {
        "sample_idx": sample_idx,
        "gold_operation": gold,
        "route": sample.get("route"),
        "train_vs_offline_pred_mismatch": train_path["prediction"] != offline_greedy_pred,
        "offline_vs_runtime_pred_mismatch": offline_greedy_pred != runtime_pred,
        "train_vs_offline_prompt_mismatch": train_path["prompt_sha256"] != offline_path["prompt_sha256"],
        "offline_greedy_prediction": offline_greedy_pred,
        "paths": {
            "training_scorer": train_path,
            "offline_evaluator_scorer": offline_path,
            "runtime_scorer": runtime_path,
        },
    }


def summarize_records(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    train_off_mismatch = sum(1 for r in records if r["train_vs_offline_pred_mismatch"])
    off_rt_mismatch = sum(1 for r in records if r["offline_vs_runtime_pred_mismatch"])
    prompt_mismatch = sum(
        1
        for r in records
        if r["paths"]["training_scorer"]["prompt_sha256"]
        != r["paths"]["offline_evaluator_scorer"]["prompt_sha256"]
    )
    margins = [r["paths"]["training_scorer"]["margin"] for r in records]
    margins_sorted = sorted(margins)
    gold_keep = [r for r in records if r.get("gold_operation") == DupOperation.KEEP_EVIDENCE.value]
    gold_skip = [r for r in records if r.get("gold_operation") == DupOperation.SKIP_DUPLICATE.value]

    def margin_stats(subset: list[dict]) -> dict[str, float]:
        ms = [r["paths"]["training_scorer"]["margin"] for r in subset]
        if not ms:
            return {}
        ms_s = sorted(ms)
        return {
            "mean": sum(ms) / len(ms),
            "median": ms_s[len(ms_s) // 2],
            "q25": ms_s[len(ms_s) // 4],
            "q75": ms_s[3 * len(ms_s) // 4],
            "n": len(ms),
        }

    return {
        "n_samples": n,
        "train_vs_offline_prediction_mismatch_rate": train_off_mismatch / n,
        "offline_vs_runtime_prediction_mismatch_rate": off_rt_mismatch / n,
        "prompt_mismatch_rate": prompt_mismatch / n,
        "margin_mean": sum(margins) / n,
        "margin_median": margins_sorted[n // 2],
        "margin_q25": margins_sorted[n // 4],
        "margin_q75": margins_sorted[3 * n // 4],
        "margin_by_gold_keep": margin_stats(gold_keep),
        "margin_by_gold_skip": margin_stats(gold_skip),
    }


def write_consistency_report(variant: str, summary: dict, out_md: Path, append: bool) -> None:
    lines = [
        f"## {variant}",
        "",
        f"- train-vs-offline mismatch rate: {summary.get('train_vs_offline_prediction_mismatch_rate', 0):.4f}",
        f"- offline-vs-runtime mismatch rate: {summary.get('offline_vs_runtime_prediction_mismatch_rate', 0):.4f}",
        f"- prompt mismatch rate: {summary.get('prompt_mismatch_rate', 0):.4f}",
        f"- margin mean/median: {summary.get('margin_mean', 0):.4f} / {summary.get('margin_median', 0):.4f}",
        f"- margin q25/q75: {summary.get('margin_q25', 0):.4f} / {summary.get('margin_q75', 0):.4f}",
        "",
    ]
    mode = "a" if append and out_md.exists() else "w"
    prefix = [] if append and out_md.exists() else [
        "# Scorer Consistency Report (Round 4 Barrier 2)",
        "",
    ]
    with out_md.open(mode, encoding="utf-8") as f:
        if prefix:
            f.write("\n".join(prefix) + "\n")
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--valid", type=Path, default=_REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=_REPO / "outputs/scope_round4/scorer_audit/SCORE_CONSISTENCY_REPORT.md")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    valid = load_jsonl(args.valid)
    if args.max_samples > 0:
        valid = valid[: args.max_samples]

    cfg = SDITrainConfig(
        model_path=args.model_path,
        output_dir=Path(f"/tmp/r4_replay_{args.variant}"),
        loss_mode="operation_ce",
        compact_target=True,
        eval_only=True,
    )
    trainer = DupSDITrainer(cfg)
    trainer.model.to(device)
    trainer.device = device
    runtime = DupOperationRuntime(model=trainer.model, tokenizer=trainer.tokenizer, device=device)

    # Print verbalizer tokenization once
    tok_info = {
        "KEEP_EVIDENCE_ids": trainer.tokenizer.encode("KEEP_EVIDENCE", add_special_tokens=False),
        "SKIP_DUPLICATE_ids": trainer.tokenizer.encode("SKIP_DUPLICATE", add_special_tokens=False),
        "KEEP_is_multitoken": len(trainer.tokenizer.encode("KEEP_EVIDENCE", add_special_tokens=False)) > 1,
        "SKIP_is_multitoken": len(trainer.tokenizer.encode("SKIP_DUPLICATE", add_special_tokens=False)) > 1,
        "scoring_definition": "length_normalized_mean_logprob_over_complete_verbalizer",
    }
    print(json.dumps(tok_info, indent=2))

    records: list[dict] = []
    for i, sample in enumerate(valid):
        if i % 50 == 0:
            print(f"[{args.variant}] {i}/{len(valid)}")
        records.append(replay_one_sample(trainer, runtime, sample, i))

    summary = summarize_records(records)
    summary["variant"] = args.variant
    summary["verbalizer_tokenization"] = tok_info

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_consistency_report(args.variant, summary, args.report, append=True)
    print(f"Wrote {args.output} ({len(records)} records)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
