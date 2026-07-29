"""Dup-only SDI training-loop diagnostics (offline held-out eval, supervision audit)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.capability.action_space import CapabilityAction
from harness.capability.adapters import render_capability_action
from harness.shadow.action_realizer import ActionRealizer
from harness.artifacts.schema import PrivilegedArtifact


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def action_bucket(action: dict[str, Any] | None) -> str:
    if not action:
        return "none"
    cap = CapabilityAction.from_dict(action)
    at = cap.action_type.value
    args = cap.arguments
    if at == "curate_document":
        adds = args.get("add_ids") or []
        rems = args.get("remove_ids") or []
        if not adds and not rems:
            return "skip_curate_empty"
        if not adds and rems:
            return "skip_curate_remove"
        if adds and rems:
            return "curate_replace"
        return "curate_add"
    return at


def normalize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    return CapabilityAction.from_dict(action).to_dict()


def analyze_route_target_distribution(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    route_target = Counter()
    route_student_target: dict[str, Counter] = defaultdict(Counter)
    student_eq_target = 0
    for r in rows:
        route = str(r.get("route", "")).upper()
        st = action_bucket(r.get("student_action"))
        tg = action_bucket(r.get("target_action"))
        route_target[(route, tg)] += 1
        route_student_target[f"{route}|{st}"][tg] += 1
        if normalize_action(r.get("student_action")) == normalize_action(
            r.get("target_action")
        ):
            student_eq_target += 1

    by_route = defaultdict(lambda: Counter())
    for (route, tg), n in route_target.items():
        by_route[route][tg] += n

    return {
        "n_samples": len(rows),
        "student_eq_target_rate": student_eq_target / max(len(rows), 1),
        "route_x_target_action": {
            route: dict(cnt) for route, cnt in by_route.items()
        },
        "route_x_student_x_target": {
            k: dict(v) for k, v in sorted(route_student_target.items())
        },
    }


def audit_skip_curate_realization(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check skip_curate artifact → ActionRealizer → rendered training tokens."""
    realizer = ActionRealizer()
    examples: list[dict[str, Any]] = []
    op_counts = Counter()
    for r in samples:
        art_raw = r.get("artifact") or {}
        if not art_raw:
            continue
        op = str(art_raw.get("recommended_operation") or "")
        if "skip" not in op.lower() and action_bucket(r.get("target_action")) not in {
            "skip_curate_empty",
            "skip_curate_remove",
        }:
            continue
        try:
            art = PrivilegedArtifact.from_dict(art_raw)
            state_raw = r.get("decision_state") or {}
            from harness.capability.state import DecisionState

            state = DecisionState.from_dict(state_raw)
            cand = realizer.realize(state, art)
            target = r.get("target_action")
            rendered_target = (
                render_capability_action(CapabilityAction.from_dict(target))
                if target
                else None
            )
            rendered_cand = (
                render_capability_action(cand.action) if cand else None
            )
            op_counts[op or "inferred_skip"] += 1
            examples.append(
                {
                    "event_id": r.get("event_id"),
                    "route": r.get("route"),
                    "recommended_operation": op,
                    "student_action": action_bucket(r.get("student_action")),
                    "target_bucket": action_bucket(target),
                    "realizer_source": cand.source if cand else None,
                    "realizer_notes": cand.notes if cand else None,
                    "realizer_action_type": (
                        cand.action.action_type.value if cand else None
                    ),
                    "rendered_target": rendered_target,
                    "rendered_realizer": rendered_cand,
                    "is_continue_proxy": (
                        cand is not None
                        and cand.action.action_type.value
                        not in {"curate_document", "review_docs"}
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            examples.append({"event_id": r.get("event_id"), "error": str(exc)})

    return {
        "n_skip_related": len(examples),
        "recommended_operation_counts": dict(op_counts),
        "continue_proxy_rate": sum(1 for e in examples if e.get("is_continue_proxy"))
        / max(len(examples), 1),
        "examples": examples[:12],
    }


@dataclass
class OfflineEvalResult:
    model_tag: str
    n: int
    target_action_accuracy: float
    correct_accuracy: float
    endorse_accuracy: float
    parse_rate: float
    teacher_forced_token_acc: float
    by_target_bucket: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "n": self.n,
            "target_action_accuracy": self.target_action_accuracy,
            "correct_accuracy": self.correct_accuracy,
            "endorse_accuracy": self.endorse_accuracy,
            "parse_rate": self.parse_rate,
            "teacher_forced_token_acc": self.teacher_forced_token_acc,
            "by_target_bucket": self.by_target_bucket,
        }


def offline_capability_eval(
    trainer,
    samples: list[dict[str, Any]],
    *,
    model_tag: str,
) -> OfflineEvalResult:
    trainer.model.eval()
    n = len(samples)
    target_ok = endorse_ok = endorse_n = correct_ok = correct_n = 0
    parse_ok = 0
    tf_correct = 0.0
    tf_total = 0
    bucket_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "hit": 0}
    )

    for sample in samples:
        pred = trainer._greedy_action(sample)
        if pred is not None:
            parse_ok += 1
        tgt = trainer._action_dict(sample.get("target_action"))
        pred_norm = trainer._action_dict(pred) if pred else None
        tg_bucket = action_bucket(sample.get("target_action"))
        bucket_stats[tg_bucket]["n"] += 1
        if pred_norm == tgt:
            target_ok += 1
            bucket_stats[tg_bucket]["hit"] += 1
        route = str(sample.get("route", "")).upper()
        if route == "ENDORSE":
            endorse_n += 1
            if pred_norm == trainer._action_dict(sample.get("student_action")):
                endorse_ok += 1
        elif route == "CORRECT":
            correct_n += 1
            if pred_norm == tgt:
                correct_ok += 1
        tf_acc, tf_n = trainer._teacher_forced_token_acc(sample)
        tf_correct += tf_acc * tf_n
        tf_total += tf_n

    by_bucket = {
        k: {
            "n": int(v["n"]),
            "accuracy": v["hit"] / max(v["n"], 1),
        }
        for k, v in sorted(bucket_stats.items())
    }
    return OfflineEvalResult(
        model_tag=model_tag,
        n=n,
        target_action_accuracy=target_ok / max(n, 1),
        correct_accuracy=correct_ok / max(correct_n, 1),
        endorse_accuracy=endorse_ok / max(endorse_n, 1),
        parse_rate=parse_ok / max(n, 1),
        teacher_forced_token_acc=tf_correct / max(tf_total, 1),
        by_target_bucket=by_bucket,
    )


def sample_balanced_subset(
    samples: list[dict[str, Any]],
    *,
    n_endorse: int,
    n_correct: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    import random

    endorse = [s for s in samples if str(s.get("route", "")).upper() == "ENDORSE"]
    correct = [s for s in samples if str(s.get("route", "")).upper() == "CORRECT"]
    rng = random.Random(seed)
    rng.shuffle(endorse)
    rng.shuffle(correct)
    return endorse[:n_endorse] + correct[:n_correct]


def filter_by_routes(
    samples: list[dict[str, Any]], routes: set[str]
) -> list[dict[str, Any]]:
    want = {r.upper() for r in routes}
    return [s for s in samples if str(s.get("route", "")).upper() in want]
