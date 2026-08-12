#!/usr/bin/env python3
"""Barrier2.2: historical vs fresh on-policy distribution shift (diagnostic only)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round13/phase_a_shift"
RAW = _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"
HIST_OFF = _REPO / "artifacts/datasets/scope_round10/frozen_replay/offline_valid.jsonl"
HIST_LIVE = _REPO / "artifacts/datasets/scope_round10/frozen_replay/base_live.jsonl"

FEATURE_KEYS = [
    "turn",
    "candidate_count",
    "rollback_budget_remaining",
    "latest_checkpoint_age",
    "successful_checkpoint_count",
    "failure_count",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_raw(split: str) -> list[dict]:
    rows: list[dict] = []
    root = RAW / split
    if not root.exists():
        return rows
    for p in sorted(root.glob("*/rollback_events.jsonl")):
        rows.extend(load_jsonl(p))
    return rows


def gold_op(r: dict) -> str:
    ta = r.get("target_action") or {}
    return str(
        ta.get("operation")
        or r.get("gold_operation")
        or r.get("operation")
        or r.get("shadow_operation")
        or "CONTINUE"
    )


def student_op(r: dict) -> str | None:
    if r.get("student_operation"):
        return str(r["student_operation"])
    return None


def margin_of(r: dict) -> float | None:
    if r.get("student_margin") is not None:
        return float(r["student_margin"])
    scores = r.get("student_scores") or r.get("vllm_logits") or r.get("canonical_logits") or {}
    if "ROLLBACK_TO" in scores and "CONTINUE" in scores:
        return float(scores["ROLLBACK_TO"]) - float(scores["CONTINUE"])
    return None


def feats(r: dict) -> dict[str, float]:
    sv = r.get("student_visible_features") or {}
    ds = r.get("decision_state") or {}
    cands = list(ds.get("available_checkpoints") or r.get("candidate_list") or [])
    turn = float(sv.get("turn") if sv.get("turn") is not None else ds.get("turn_id") or r.get("turn") or 0)
    out = {
        "turn": turn,
        "candidate_count": float(
            sv.get("candidate_count") if sv.get("candidate_count") is not None else len(cands)
        ),
        "rollback_budget_remaining": float(
            sv.get("rollback_budget_remaining")
            if sv.get("rollback_budget_remaining") is not None
            else ds.get("remaining_recovery_budget") or 0
        ),
        "latest_checkpoint_age": float(sv.get("latest_checkpoint_age") or 0),
        "successful_checkpoint_count": float(sv.get("successful_checkpoint_count") or 0),
        "failure_count": float(sv.get("failure_count") or ds.get("repeated_query_count") or 0),
    }
    return out


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    qs = [1, 5, 25, 50, 75, 95, 99]
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        **{f"q{q:02d}": float(np.percentile(a, q)) for q in qs},
    }


def smd_ks(a: list[float], b: list[float]) -> dict[str, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if len(aa) == 0 or len(bb) == 0:
        return {"smd": float("nan"), "ks": float("nan")}
    pooled = math.sqrt(0.5 * (aa.var() + bb.var()) + 1e-12)
    smd = float((aa.mean() - bb.mean()) / pooled)
    # simple KS
    xa = np.sort(aa)
    xb = np.sort(bb)
    grid = np.unique(np.concatenate([xa, xb]))
    fa = np.searchsorted(xa, grid, side="right") / len(xa)
    fb = np.searchsorted(xb, grid, side="right") / len(xb)
    ks = float(np.max(np.abs(fa - fb)))
    return {"smd": smd, "ks": ks}


def summarize(name: str, rows: list[dict]) -> dict[str, Any]:
    golds = [gold_op(r) for r in rows]
    studs = [student_op(r) for r in rows if student_op(r)]
    margins = [m for r in rows if (m := margin_of(r)) is not None]
    by_gold: dict[str, dict] = {}
    for g in ("CONTINUE", "ROLLBACK_TO"):
        idxs = [r for r in rows if gold_op(r) == g]
        ms = [m for r in idxs if (m := margin_of(r)) is not None]
        err = 0
        n = 0
        for r in idxs:
            s = student_op(r)
            if s is None:
                continue
            n += 1
            err += int(s != g)
        by_gold[g] = {
            "n": len(idxs),
            "margin": quantiles(ms),
            "student_error_rate": err / max(n, 1),
        }
    feat_vals = {k: [] for k in FEATURE_KEYS}
    for r in rows:
        f = feats(r)
        for k in FEATURE_KEYS:
            feat_vals[k].append(f[k])
    return {
        "name": name,
        "n": len(rows),
        "gold_continue_prior": golds.count("CONTINUE") / max(len(golds), 1),
        "gold_rollback_prior": golds.count("ROLLBACK_TO") / max(len(golds), 1),
        "student_predicted_prior": dict(Counter(studs)),
        "margin": quantiles(margins),
        "by_gold_class": by_gold,
        "feature_means": {k: float(np.mean(v)) if v else None for k, v in feat_vals.items()},
        "_feat_vals": feat_vals,
    }


def domain_auc(hist_rows: list[dict], fresh_rows: list[dict]) -> float:
    """Logistic-regression-free AUC via Mann–Whitney on a linear score of features."""
    Xh = np.array([[feats(r)[k] for k in FEATURE_KEYS] for r in hist_rows], dtype=np.float64)
    Xf = np.array([[feats(r)[k] for k in FEATURE_KEYS] for r in fresh_rows], dtype=np.float64)
    if len(Xh) == 0 or len(Xf) == 0:
        return float("nan")
    mu = np.concatenate([Xh, Xf], axis=0).mean(axis=0)
    sd = np.concatenate([Xh, Xf], axis=0).std(axis=0) + 1e-6
    Xh = (Xh - mu) / sd
    Xf = (Xf - mu) / sd
    # score = mean feature difference direction
    w = Xf.mean(axis=0) - Xh.mean(axis=0)
    sh = Xh @ w
    sf = Xf @ w
    # AUC
    all_s = np.concatenate([sh, sf])
    all_y = np.concatenate([np.zeros(len(sh)), np.ones(len(sf))])
    order = np.argsort(all_s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_s) + 1)
    pos_ranks = ranks[all_y == 1]
    n_pos = len(sf)
    n_neg = len(sh)
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
    return float(auc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist_off = load_jsonl(HIST_OFF)
    hist_live = load_jsonl(HIST_LIVE)
    fresh_tr = load_raw("train")
    fresh_va = load_raw("valid")

    blocks = {
        "historical_offline_valid": summarize("historical_offline_valid", hist_off),
        "historical_base_live": summarize("historical_base_live", hist_live),
        "r13_train_onpolicy": summarize("r13_train_onpolicy", fresh_tr),
        "r13_valid_onpolicy": summarize("r13_valid_onpolicy", fresh_va),
    }

    # pairwise feature shift: hist offline vs fresh valid
    shift = {}
    a = blocks["historical_offline_valid"]["_feat_vals"]
    b = blocks["r13_valid_onpolicy"]["_feat_vals"]
    for k in FEATURE_KEYS:
        shift[k] = smd_ks(a[k], b[k])

    auc = domain_auc(hist_off, fresh_va)
    for v in blocks.values():
        v.pop("_feat_vals", None)

    report = {
        "splits": blocks,
        "feature_shift_hist_offline_vs_r13_valid": shift,
        "domain_classifier_auc_hist_offline_vs_r13_valid": auc,
        "note": "diagnostic only; not used for model selection",
    }
    (OUT / "DISTRIBUTION_SHIFT.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# DISTRIBUTION_SHIFT\n",
        f"- domain_classifier_auc (hist offline vs R13 valid) = {auc:.4f}\n",
        f"- R13 train n={blocks['r13_train_onpolicy']['n']} "
        f"gold_RB={blocks['r13_train_onpolicy']['gold_rollback_prior']:.3f}\n",
        f"- R13 valid n={blocks['r13_valid_onpolicy']['n']} "
        f"gold_RB={blocks['r13_valid_onpolicy']['gold_rollback_prior']:.3f}\n",
    ]
    for k, v in shift.items():
        md.append(f"- {k}: smd={v['smd']:.3f} ks={v['ks']:.3f}\n")
    (OUT / "DISTRIBUTION_SHIFT.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "splits"}, indent=2))


if __name__ == "__main__":
    main()
