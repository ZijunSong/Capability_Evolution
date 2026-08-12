#!/usr/bin/env python3
"""Threshold-only diagnostic: sweep τ on offline_valid, freeze, apply to base_live.

Uses P0 seed42 merged checkpoint. Does not train. Threshold chosen only on offline_valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope_round9.aggregate_frozen_replay import operation_metrics
from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy, _confusion_matrix


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def preds_from_hf(rows: list[dict], threshold: float) -> list[dict]:
    out = []
    for r in rows:
        scores = r.get("hf_logits") or {}
        d = decide_rollback_operation(
            score_continue=float(scores.get("CONTINUE", -1e9)),
            score_replan=float(scores.get("REPLAN", -1e9)),
            score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
            threshold=threshold,
            disable_replan=True,
        )
        out.append({**r, "pred_operation": d.predicted_operation.value})
    return out


def metrics(rows: list[dict]) -> dict:
    m = operation_metrics(rows)
    matrix = m["confusion_matrix"]
    def recall(op: str):
        s = sum(matrix.get(op, {}).values()) if isinstance(matrix.get(op), dict) else 0
        return None if s <= 0 else matrix[op][op] / s
    return {
        **m,
        "ContinueRecall": recall("CONTINUE"),
        "RollbackRecall": recall("ROLLBACK_TO"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--offline", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hf-offline-replay", type=Path, default=None)
    p.add_argument("--hf-holdout-replay", type=Path, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer existing P0 HF logits to avoid reloading the model for this diagnostic.
    off_src = args.hf_offline_replay or (
        _REPO / "outputs/scope_round9/wave_b_p0/rollback_hier_o7_seed42/eval_offline_valid/hf_replay.jsonl"
    )
    hold_src = args.hf_holdout_replay or (
        _REPO / "outputs/scope_round9/wave_b_p0/rollback_hier_o7_seed42/eval_holdout/hf_replay.jsonl"
    )
    offline = load_jsonl(off_src)
    holdout = load_jsonl(hold_src)

    sweep = []
    best = None
    for i in range(0, 51):
        thr = i / 100.0
        m = metrics(preds_from_hf(offline, thr))
        row = {"threshold": thr, **{k: m.get(k) for k in (
            "operation_balanced_accuracy", "ContinueRecall", "RollbackRecall"
        )}}
        sweep.append(row)
        score = (
            (m.get("operation_balanced_accuracy") or 0),
            (m.get("ContinueRecall") or 0),
            (m.get("RollbackRecall") or 0),
        )
        if best is None or score > best["score"]:
            best = {"threshold": thr, "score": score, "offline_metrics": row}

    thr = float(best["threshold"])
    hold_m = metrics(preds_from_hf(holdout, thr))
    report = {
        "variant": "r10_threshold_only_p0_seed42",
        "model_path": str(args.model_path),
        "selected_threshold_on_offline_valid": thr,
        "offline_sweep": sweep,
        "offline_at_selected": best["offline_metrics"],
        "holdout_at_selected": {
            "operation_balanced_accuracy": hold_m.get("operation_balanced_accuracy"),
            "ContinueRecall": hold_m.get("ContinueRecall"),
            "RollbackRecall": hold_m.get("RollbackRecall"),
            "prediction_prior": hold_m.get("prediction_prior"),
        },
        "note": "Diagnostic control only; main methods use threshold=0",
    }
    (args.out_dir / "THRESHOLD_SWEEP_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "TRAIN_AND_EVAL_REPORT.json").write_text(
        json.dumps(
            {
                "variant": "r10_threshold_only_p0_seed42",
                "threshold_only": True,
                "selected_threshold": thr,
                "offline_valid": {"hf_metrics": best["offline_metrics"]},
                "holdout": {"hf_metrics": hold_m},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_threshold": thr, "holdout": report["holdout_at_selected"]}, indent=2))


if __name__ == "__main__":
    main()
