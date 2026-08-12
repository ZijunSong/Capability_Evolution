#!/usr/bin/env python3
"""Barrier 2.2: Prior shift and margin diagnosis on live splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.rollback_operation_objectives import score_rollback_prompt
from training.scope.operation_objectives import ScoreNorm
from training.scope_round10.common import BASE_MODEL, DATA, OUT, R9_OUT, binary_operation, load_jsonl, write_json
from transformers import AutoModelForCausalLM, AutoTokenizer

PRIOR_DIR = OUT / "prior_shift"
SHARD_DIR = PRIOR_DIR / "shards"
SPLIT_DIR = DATA / "live_split"

MODEL_PATHS = {
    "rollback_o7_seed42": _REPO / "outputs/scope_round8/merged/rollback_o7_seed42",
    "rollback_o7_seed43": _REPO / "outputs/scope_round8/merged/rollback_o7_seed43",
    "rollback_o7_seed44": _REPO / "outputs/scope_round8/merged/rollback_o7_seed44",
    "rollback_hier_o7_seed42": R9_OUT / "wave_b/rollback_hier_o7_seed42/merged",
    "rollback_hier_o7_seed43": R9_OUT / "wave_b/rollback_hier_o7_seed43/merged",
    "rollback_hier_o7_seed44": R9_OUT / "wave_b/rollback_hier_o7_seed44/merged",
    "rollback_flat_o7_seed42_repro": R9_OUT / "wave_b/rollback_flat_o7_seed42_repro/merged",
    "rollback_hier_prompt_hint_seed42": R9_OUT / "wave_b/rollback_hier_prompt_hint_seed42/merged",
    "base_agent_core": Path(BASE_MODEL),
}


def resolve_model(path: Path) -> Path:
    if (path / "merged/config.json").exists():
        return path / "merged"
    if (path / "config.json").exists():
        return path
    return Path(BASE_MODEL)


def score_row(model, tokenizer, row: dict, device) -> dict:
    text = row.get("effective_input_text") or ""
    if not text and row.get("decision_state"):
        from training.scope.rollback_effective_input import build_rollback_effective_input

        eff = build_rollback_effective_input(row, tokenizer, max_length=2048)
        text = eff.text
    s_cont, s_replan, s_roll = score_rollback_prompt(
        model, tokenizer, text, device=device, norm=ScoreNorm.MEAN
    )
    margin = float(s_roll.item()) - float(s_cont.item())
    pred = "ROLLBACK_TO" if margin >= 0 else "CONTINUE"
    gold = binary_operation(row) or "CONTINUE"
    return {
        "event_id": row.get("event_id"),
        "query_id": row.get("query_id"),
        "turn": row.get("turn", 0),
        "gold_operation": gold,
        "pred_operation": pred,
        "margin": margin,
        "score_continue": float(s_cont.item()),
        "score_rollback": float(s_roll.item()),
        "n_candidates": len(row.get("candidate_list") or []),
        "state_source": row.get("state_source", "live"),
        "truncated": row.get("truncated", False),
        "token_length": row.get("token_length_after", 0),
        "correct": pred == gold,
    }


def aggregate_events(events: list[dict]) -> dict:
    if not events:
        return {}
    matrix: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        matrix[e["gold_operation"]][e["pred_operation"]] += 1
    ops = ["CONTINUE", "ROLLBACK_TO"]
    per_class_recall = {}
    for op in ops:
        support = sum(matrix[op].values())
        per_class_recall[op] = matrix[op][op] / max(support, 1)
    bal_acc = sum(per_class_recall.values()) / len(ops)
    margins_by_gold: dict[str, list[float]] = defaultdict(list)
    for e in events:
        margins_by_gold[e["gold_operation"]].append(e["margin"])
    return {
        "n_events": len(events),
        "confusion_matrix": {k: dict(v) for k, v in matrix.items()},
        "class_prior": dict(Counter(e["gold_operation"] for e in events)),
        "balanced_accuracy": bal_acc,
        "ContinueRecall": per_class_recall.get("CONTINUE", 0),
        "RollbackRecall": per_class_recall.get("ROLLBACK_TO", 0),
        "mean_margin_by_gold": {
            k: sum(v) / max(len(v), 1) for k, v in margins_by_gold.items()
        },
    }


def score_models(model_names: list[str], device) -> dict:
    splits = {}
    for name in ("live_calibration", "live_train", "live_valid", "live_test"):
        path = SPLIT_DIR / f"{name}.jsonl"
        if path.exists():
            splits[name] = load_jsonl(path)

    per_model = {}
    for model_name in model_names:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path:
            print(f"skip unknown model {model_name}")
            continue
        resolved = resolve_model(model_path)
        if not (resolved / "config.json").exists():
            print(f"skip {model_name}: no model at {resolved}")
            continue
        print(f"scoring {model_name} from {resolved}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        model.eval()
        model_metrics = {}
        events = []
        for split_name, rows in splits.items():
            split_events = [score_row(model, tokenizer, r, device) for r in rows]
            for e in split_events:
                e["model"] = model_name
                e["split"] = split_name
            events.extend(split_events)
            model_metrics[split_name] = aggregate_events(split_events)
        per_model[model_name] = model_metrics
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        write_json(SHARD_DIR / f"{model_name}_metrics.json", model_metrics)
        with (SHARD_DIR / f"{model_name}_events.jsonl").open("w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()
    return per_model


def merge_shards() -> dict:
    per_model = {}
    all_events = []
    for metrics_path in sorted(SHARD_DIR.glob("*_metrics.json")):
        model_name = metrics_path.name.replace("_metrics.json", "")
        per_model[model_name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        ev_path = SHARD_DIR / f"{model_name}_events.jsonl"
        if ev_path.exists():
            all_events.extend(load_jsonl(ev_path))

    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PRIOR_DIR / "per_model_metrics.json", per_model)
    with (PRIOR_DIR / "margin_events.jsonl").open("w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    lines = [
        "# Prior Shift Report",
        "",
        "## Key question: CONTINUE collapse on live vs train prior",
        "",
    ]
    for mn, sm in per_model.items():
        if "live_test" not in sm:
            continue
        lt = sm["live_test"]
        lines.append(
            f"- **{mn}** live_test: bal_acc={lt.get('balanced_accuracy', 0):.3f}, "
            f"ContinueRecall={lt.get('ContinueRecall', 0):.3f}, "
            f"RollbackRecall={lt.get('RollbackRecall', 0):.3f}, "
            f"prior={lt.get('class_prior')}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Low ContinueRecall with high RollbackRecall on live_test indicates "
        "operation prior shift (model over-predicts ROLLBACK_TO). "
        "Shared threshold calibration is the first remediation (Barrier 3).",
    ]
    (PRIOR_DIR / "PRIOR_SHIFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Prior shift merge complete")
    return per_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--models", default="", help="Comma-separated model names; default=all")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    if args.merge_only:
        merge_shards()
        return

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    if args.models.strip():
        names = [x.strip() for x in args.models.split(",") if x.strip()]
    else:
        names = list(MODEL_PATHS.keys())

    per_model = score_models(names, device)
    # Single-process full run also writes merged outputs
    if len(names) == len(MODEL_PATHS):
        PRIOR_DIR.mkdir(parents=True, exist_ok=True)
        write_json(PRIOR_DIR / "per_model_metrics.json", per_model)
        all_margin_events = []
        for mn in names:
            all_margin_events.extend(load_jsonl(SHARD_DIR / f"{mn}_events.jsonl"))
        with (PRIOR_DIR / "margin_events.jsonl").open("w", encoding="utf-8") as f:
            for e in all_margin_events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        merge_shards()
    else:
        print(f"shard written for {names}")


if __name__ == "__main__":
    main()
