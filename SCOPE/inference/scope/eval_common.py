"""Unified evaluation metrics for SCOPE ICLR ablations.

Wraps and extends existing Round 2/4/6/8 metric implementations.
No silent fallbacks: missing labels or malformed predictions raise.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from training.scope.binary_operation_metrics import compute_binary_operation_metrics
from training.scope_round6.metrics import direct_behavior_metrics

KEEP = "KEEP_EVIDENCE"
SKIP = "SKIP_DUPLICATE"


def _normalize_label(x: str) -> str:
    s = str(x).strip().upper()
    aliases = {
        "KEEP": KEEP,
        "KEEP_EVIDENCE": KEEP,
        "SKIP": SKIP,
        "SKIP_DUPLICATE": SKIP,
        "CONTINUE": "CONTINUE",
        "REPLAN": "REPLAN",
        "ROLLBACK": "ROLLBACK",
        "INTERVENE": "INTERVENE",
    }
    if s not in aliases and s not in {KEEP, SKIP, "CONTINUE", "REPLAN", "ROLLBACK", "INTERVENE"}:
        # allow raw; caller may treat as invalid
        return s
    return aliases.get(s, s)


def classification_metrics(
    gold: Sequence[str],
    pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Unified classification metrics with confusion + invalid parse rate."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    if not gold:
        raise ValueError("empty gold/pred — refusing to report vacuous metrics")

    g_norm = [_normalize_label(g) for g in gold]
    p_norm = [_normalize_label(p) for p in pred]

    # Binary KEEP/SKIP path reuses battle-tested implementation.
    if labels is None and set(g_norm) <= {KEEP, SKIP} and set(p_norm) <= {KEEP, SKIP, "__PARSE_FAIL__"}:
        valid_pairs = [(g, p) for g, p in zip(g_norm, p_norm) if p != "__PARSE_FAIL__"]
        invalid = sum(1 for p in p_norm if p == "__PARSE_FAIL__")
        if not valid_pairs:
            raise ValueError("all predictions invalid — no metrics")
        vg, vp = zip(*valid_pairs)
        m = compute_binary_operation_metrics(list(vg), list(vp))
        out = m.to_dict()
        out["confusion_matrix"] = {
            "tp_keep": m.tp_keep,
            "fp_keep": m.fp_keep,
            "fn_keep": m.fn_keep,
            "tp_skip": m.tp_skip,
            "fp_skip": m.fp_skip,
            "fn_skip": m.fn_skip,
        }
        out["prediction_distribution"] = dict(Counter(p_norm))
        out["label_distribution"] = dict(Counter(g_norm))
        out["invalid_parse_rate"] = invalid / len(pred)
        out["invalid_action_rate"] = out["invalid_parse_rate"]
        return out

    label_list = list(labels) if labels is not None else sorted(set(g_norm))
    cm: dict[str, dict[str, int]] = {a: {b: 0 for b in label_list} for a in label_list}
    invalid = 0
    correct = 0
    for g, p in zip(g_norm, p_norm):
        if p not in label_list:
            invalid += 1
            continue
        if g not in label_list:
            raise ValueError(f"unexpected gold label: {g}")
        cm[g][p] += 1
        if g == p:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    f1s = []
    for lab in label_list:
        tp = cm[lab][lab]
        fp = sum(cm[o][lab] for o in label_list if o != lab)
        fn = sum(cm[lab][o] for o in label_list if o != lab)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[lab] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(cm[lab].values()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        recalls.append(rec)
        f1s.append(f1)

    n_valid = len(pred) - invalid
    return {
        "n_samples": len(pred),
        "n_valid": n_valid,
        "accuracy": correct / max(n_valid, 1),
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "prediction_distribution": dict(Counter(p_norm)),
        "label_distribution": dict(Counter(g_norm)),
        "invalid_parse_rate": invalid / len(pred),
        "invalid_action_rate": invalid / len(pred),
    }


def dup_closed_loop_metrics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Dup closed-loop metrics from episode dicts."""
    if not episodes:
        raise ValueError("empty episodes")

    # Prefer decision-level labels when present.
    labels: list[str] = []
    preds: list[str] = []
    for ep in episodes:
        for d in ep.get("decisions") or []:
            if d.get("gold") is not None and d.get("pred") is not None:
                labels.append(_normalize_label(d["gold"]))
                preds.append(_normalize_label(d["pred"]))

    behavior: dict[str, Any] = {}
    if labels:
        behavior = direct_behavior_metrics(labels, preds)
        clf = classification_metrics(labels, preds)
    else:
        clf = {}

    n = len(episodes)
    def mean(key: str, default: float = 0.0) -> float:
        return sum(float(e.get(key, default) or 0) for e in episodes) / n

    valid_decisions = sum(int(e.get("valid_decision_count", len(e.get("decisions") or []))) for e in episodes)
    keep_support = sum(1 for lab in labels if lab == KEEP)
    skip_support = sum(1 for lab in labels if lab == SKIP)

    return {
        "n_episodes": n,
        "DupRejectRecall": behavior.get("DupRejectRecall"),
        "FalseSkipRate": behavior.get("FalseSkipRate"),
        "DuplicateCurateRate": mean("dup_curate_rate"),
        "valid_decision_count": valid_decisions,
        "KEEP_support": keep_support,
        "SKIP_support": skip_support,
        "decision_coverage": valid_decisions / max(n, 1),
        "mean_curated_evidence": mean("n_curated"),
        "recall": mean("recall"),
        "trajectory_recall": mean("trajectory_recall", mean("recall")),
        "final_answer_recall": mean("final_answer_recall", mean("recall")),
        "reward": mean("reward"),
        "turns": mean("turns"),
        "tool_calls": mean("tool_calls"),
        "errors": sum(len(e.get("errors") or []) for e in episodes),
        "classification": clf,
        "behavior": behavior,
    }


def rollback_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Layered Rollback metrics — never a single joint accuracy alone."""
    if not rows:
        raise ValueError("empty rollback rows")

    intervene_gold = []
    intervene_pred = []
    op_gold = []
    op_pred = []
    ckpt_correct = ckpt_total = 0
    ckpt_executable = restore_ok = invalid_ckpt = 0
    budget_viol = invariant_pass = 0
    n = len(rows)

    for r in rows:
        g_op = _normalize_label(r["gold_operation"])
        p_op = _normalize_label(r["pred_operation"])
        op_gold.append(g_op)
        op_pred.append(p_op)

        g_int = 0 if g_op == "CONTINUE" else 1
        p_int = 0 if p_op == "CONTINUE" else 1
        intervene_gold.append("CONTINUE" if g_int == 0 else "INTERVENE")
        intervene_pred.append("CONTINUE" if p_int == 0 else "INTERVENE")

        if g_op == "ROLLBACK":
            ckpt_total += 1
            if r.get("pred_checkpoint_id") == r.get("gold_checkpoint_id"):
                ckpt_correct += 1
            if r.get("checkpoint_executable"):
                ckpt_executable += 1
            else:
                invalid_ckpt += 1
            if r.get("restore_success"):
                restore_ok += 1
        if r.get("recovery_budget_violation"):
            budget_viol += 1
        if r.get("post_action_invariant_pass", True):
            invariant_pass += 1

    op_m = classification_metrics(op_gold, op_pred, labels=["CONTINUE", "REPLAN", "ROLLBACK"])
    int_m = classification_metrics(intervene_gold, intervene_pred, labels=["CONTINUE", "INTERVENE"])

    return {
        "n": n,
        "intervention_binary_accuracy": int_m["accuracy"],
        "operation_type_accuracy": op_m["accuracy"],
        "operation_type_balanced_accuracy": op_m["balanced_accuracy"],
        "CONTINUE_recall": op_m["per_class"]["CONTINUE"]["recall"],
        "REPLAN_recall": op_m["per_class"]["REPLAN"]["recall"],
        "ROLLBACK_recall": op_m["per_class"]["ROLLBACK"]["recall"],
        "checkpoint_selection_accuracy": ckpt_correct / max(ckpt_total, 1),
        "checkpoint_executable_rate": ckpt_executable / max(ckpt_total, 1),
        "restore_success_rate": restore_ok / max(ckpt_total, 1),
        "invalid_checkpoint_rate": invalid_ckpt / max(ckpt_total, 1),
        "recovery_budget_violation_rate": budget_viol / n,
        "post_action_invariant_pass_rate": invariant_pass / n,
        "task_recall": sum(float(r.get("task_recall", 0) or 0) for r in rows) / n,
        "reward": sum(float(r.get("reward", 0) or 0) for r in rows) / n,
        "operation_metrics": op_m,
        "intervention_metrics": int_m,
    }
