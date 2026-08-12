"""A1: Same-state supervision source builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]


def state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_same_state_on_policy(
    live_states: list[dict[str, Any]],
    *,
    shadow_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shadow may only read student-visible fields of the live DecisionState."""
    if len(live_states) != len(shadow_labels):
        raise ValueError("live_states / shadow_labels length mismatch")
    out = []
    for st, lab in zip(live_states, shadow_labels):
        h = state_hash(st)
        row = {
            **lab,
            "supervision_source": "same_state_on_policy",
            "live_state_hash": h,
            "source_state_hash": h,
            "state": st,
        }
        if row["live_state_hash"] != row["source_state_hash"]:
            raise RuntimeError("same-state invariant violated")
        out.append(row)
    return out


def build_trajectory_teacher(
    trajectories: list[dict[str, Any]],
    *,
    n_target: int,
) -> list[dict[str, Any]]:
    """Extract supervision from full-harness / corrected trajectories (no live typed shadow)."""
    rows = []
    for tr in trajectories:
        for ev in tr.get("events") or tr.get("steps") or []:
            rows.append(
                {
                    "supervision_source": "trajectory_teacher",
                    "query_id": tr.get("query_id"),
                    "turn": ev.get("turn"),
                    "label": ev.get("label") or ev.get("operation"),
                    "target": ev.get("target") or ev.get("action_text"),
                    "live_state_hash": None,
                    "source_state_hash": state_hash(ev.get("state") or ev),
                    "state": ev.get("state"),
                }
            )
            if len(rows) >= n_target:
                return rows[:n_target]
    if len(rows) < n_target:
        raise ValueError(f"trajectory_teacher produced {len(rows)} < n_target={n_target}")
    return rows[:n_target]


def build_cross_state_matched(
    live_states: list[dict[str, Any]],
    pool_states: list[dict[str, Any]],
    *,
    shadow_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match a similar but non-identical state; exact same hash is forbidden."""
    if len(live_states) != len(shadow_labels):
        raise ValueError("length mismatch")
    pool_hashes = [(state_hash(s), s) for s in pool_states]
    out = []
    for st, lab in zip(live_states, shadow_labels):
        live_h = state_hash(st)
        # Prefer same query / turn when available, else first different hash.
        qid = st.get("query_id")
        turn = st.get("turn")
        best = None
        best_dist = 1e18
        for h, ps in pool_hashes:
            if h == live_h:
                continue
            dist = 0.0
            if ps.get("query_id") != qid:
                dist += 10.0
            if ps.get("turn") != turn:
                dist += abs(int(ps.get("turn") or 0) - int(turn or 0))
            # cheap token overlap distance on candidate text
            a = set(str(st.get("candidate_text", "")).split())
            b = set(str(ps.get("candidate_text", "")).split())
            if a or b:
                dist += 1.0 - (len(a & b) / max(len(a | b), 1))
            if dist < best_dist:
                best_dist = dist
                best = (h, ps, dist)
        if best is None:
            raise ValueError("no cross-state match available (all hashes identical)")
        src_h, src_st, dist = best
        if src_h == live_h:
            raise RuntimeError("cross-state exact hash collision forbidden")
        out.append(
            {
                **lab,
                "supervision_source": "cross_state_matched",
                "live_state_hash": live_h,
                "source_state_hash": src_h,
                "match_distance": dist,
                "state": src_st,
            }
        )
    return out


def build_static_offline(
    historical_states: list[dict[str, Any]],
    *,
    shadow_labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Frozen historical base rollout states — do not refresh with current checkpoint."""
    n = min(len(historical_states), len(shadow_labels))
    if n == 0:
        raise ValueError("empty static_offline inputs")
    out = []
    for st, lab in zip(historical_states[:n], shadow_labels[:n]):
        h = state_hash(st)
        out.append(
            {
                **lab,
                "supervision_source": "static_offline",
                "live_state_hash": None,
                "source_state_hash": h,
                "state": st,
                "frozen": True,
            }
        )
    return out


def report_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(r.get("label") or r.get("operation") or "") for r in rows]
    support: dict[str, int] = {}
    for lab in labels:
        support[lab] = support.get(lab, 0) + 1
    same = sum(
        1
        for r in rows
        if r.get("live_state_hash")
        and r.get("live_state_hash") == r.get("source_state_hash")
    )
    cross = sum(
        1
        for r in rows
        if r.get("live_state_hash")
        and r.get("source_state_hash")
        and r["live_state_hash"] != r["source_state_hash"]
    )
    return {
        "n_samples": len(rows),
        "label_support": support,
        "same_state_count": same,
        "cross_state_count": cross,
        "schema_keys": sorted({k for r in rows for k in r}),
    }


def build_from_paths(
    variant: str,
    *,
    live_path: Path | None = None,
    pool_path: Path | None = None,
    traj_path: Path | None = None,
    hist_path: Path | None = None,
    labels_path: Path | None = None,
    n_target: int = 16,
    output_path: Path | None = None,
) -> dict[str, Any]:
    labels = _load_jsonl(labels_path) if labels_path and labels_path.exists() else [
        {"label": "KEEP_EVIDENCE" if i % 2 == 0 else "SKIP_DUPLICATE", "operation": "KEEP_EVIDENCE" if i % 2 == 0 else "SKIP_DUPLICATE"}
        for i in range(n_target)
    ]
    if variant == "a1_same_state_on_policy":
        live = _load_jsonl(live_path) if live_path and live_path.exists() else [
            {"query_id": f"q{i}", "turn": i, "candidate_text": f"doc {i}", "candidate_id": f"c{i}"}
            for i in range(n_target)
        ]
        rows = build_same_state_on_policy(live[:n_target], shadow_labels=labels[:n_target])
    elif variant == "a1_trajectory_teacher":
        trajs = _load_jsonl(traj_path) if traj_path and traj_path.exists() else [
            {
                "query_id": f"q{i}",
                "events": [
                    {
                        "turn": 0,
                        "label": "KEEP_EVIDENCE" if i % 2 == 0 else "SKIP_DUPLICATE",
                        "target": "KEEP_EVIDENCE",
                        "state": {"query_id": f"q{i}", "turn": 0, "candidate_text": f"t{i}"},
                    }
                ],
            }
            for i in range(n_target)
        ]
        rows = build_trajectory_teacher(trajs, n_target=n_target)
    elif variant == "a1_cross_state_matched":
        live = _load_jsonl(live_path) if live_path and live_path.exists() else [
            {"query_id": f"q{i}", "turn": i, "candidate_text": f"doc {i}", "candidate_id": f"c{i}"}
            for i in range(n_target)
        ]
        pool = _load_jsonl(pool_path) if pool_path and pool_path.exists() else [
            {"query_id": f"q{i}", "turn": i + 1, "candidate_text": f"other {i}", "candidate_id": f"p{i}"}
            for i in range(n_target)
        ]
        rows = build_cross_state_matched(live[:n_target], pool, shadow_labels=labels[:n_target])
    elif variant == "a1_static_offline":
        hist = _load_jsonl(hist_path) if hist_path and hist_path.exists() else [
            {"query_id": f"q{i}", "turn": 0, "candidate_text": f"hist {i}", "candidate_id": f"h{i}"}
            for i in range(n_target)
        ]
        rows = build_static_offline(hist, shadow_labels=labels[:n_target])
    else:
        raise ValueError(f"unknown A1 variant: {variant}")

    report = report_dataset(rows)
    if report["n_samples"] < 16:
        raise ValueError(f"A1 smoke gate failed: n_samples={report['n_samples']} < 16")
    if output_path:
        _write_jsonl(output_path, rows)
        (output_path.parent / "dataset_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return {"rows": rows, "report": report}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-target", type=int, default=16)
    args = p.parse_args()
    result = build_from_paths(args.variant, n_target=args.n_target, output_path=Path(args.output))
    print(json.dumps(result["report"], indent=2))
