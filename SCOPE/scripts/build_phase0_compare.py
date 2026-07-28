#!/usr/bin/env python3
"""Build Phase-0 compare JSON: Bare / Harness v1 / Harness v2 / Minimal Runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]


def _normalize_answer(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def grade_bare_success(response_text: str, gold_answer: str) -> bool:
    if not response_text or not gold_answer:
        return False
    resp = _normalize_answer(response_text)
    gold = _normalize_answer(gold_answer)
    if not gold:
        return False
    if gold in resp:
        return True
    return gold_answer.strip().lower() in response_text.lower()


def load_gold() -> dict[str, str]:
    path = _REPO / "external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", ""))
            ans = obj.get("answer") or obj.get("final_answer") or ""
            if qid:
                out[qid] = str(ans)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_harness(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(rows)
    def mean(k: str) -> float:
        return sum(float(r.get(k) or 0) for r in rows) / n if n else 0.0

    recalls = [float(r.get("recall") or 0) for r in rows]
    traj = [float(r.get("trajectory_recall") or 0) for r in rows]
    fa = [float(r.get("final_answer_recall") or 0) for r in rows]
    errors = sum(1 for r in rows if r.get("error"))
    return {
        "name": name,
        "n": n,
        "recall": mean("recall"),
        "trajectory_recall": mean("trajectory_recall"),
        "final_answer_recall": mean("final_answer_recall"),
        "precision": mean("precision"),
        "reward": mean("reward"),
        "recall_gt0_n": sum(1 for x in recalls if x > 0),
        "recall_gt0_rate": sum(1 for x in recalls if x > 0) / n if n else 0.0,
        "traj_gt0_n": sum(1 for x in traj if x > 0),
        "traj_gt0_rate": sum(1 for x in traj if x > 0) / n if n else 0.0,
        "fa_gt0_n": sum(1 for x in fa if x > 0),
        "fa_gt0_rate": sum(1 for x in fa if x > 0) / n if n else 0.0,
        "mean_turns": mean("turns"),
        "mean_n_curated": mean("n_curated"),
        "mean_n_pool": mean("n_pool"),
        "error_n": errors,
        "error_rate": errors / n if n else 0.0,
        "mean_elapsed_s": mean("elapsed_s"),
    }


def main() -> None:
    bare = load_jsonl(_REPO / "outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl")
    v1 = load_jsonl(_REPO / "outputs/harness_rollout_browsecomp_full/harness_rollouts.jsonl")
    v2 = load_jsonl(
        _REPO / "outputs/harness_rollout_browsecomp_full_v2/harness_rollouts.jsonl"
    )
    minimal_path = _REPO / "outputs/minimal_runtime_browsecomp_full830/episodes.jsonl"
    if not minimal_path.exists():
        minimal_path = (
            _REPO / "outputs/minimal_runtime_browsecomp_full830/harness_rollouts.jsonl"
        )
    minimal = load_jsonl(minimal_path)
    gold = load_gold()

    bare_ok = 0
    for r in bare:
        qid = str(r.get("query_id", ""))
        if grade_bare_success(str(r.get("response_text", "")), gold.get(qid, "")):
            bare_ok += 1

    bare_summary = {
        "name": "bare",
        "n": len(bare),
        "answer_match_n": bare_ok,
        "answer_match_acc": bare_ok / len(bare) if bare else 0.0,
        "note": "single-shot freeform; graded by normalized substring match vs gold",
        "recall": None,
        "trajectory_recall": None,
        "final_answer_recall": None,
        "precision": None,
        "reward": None,
        "mean_turns": 1.0,
        "error_n": 0,
        "error_rate": 0.0,
    }

    report = {
        "phase": "phase0",
        "n_queries": 830,
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "BrowseComp+",
        "retriever": "BM25",
        "max_turns": 35,
        "bare": bare_summary,
        "harness_v1": summarize_harness(v1, "harness_v1"),
        "harness_v2": summarize_harness(v2, "harness_v2"),
        "minimal_runtime": summarize_harness(minimal, "minimal_runtime"),
        "sources": {
            "bare": "outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl",
            "harness_v1": "outputs/harness_rollout_browsecomp_full/harness_rollouts.jsonl",
            "harness_v2": "outputs/harness_rollout_browsecomp_full_v2/harness_rollouts.jsonl",
            "minimal_runtime": str(minimal_path.relative_to(_REPO)),
        },
    }

    out = _REPO / "outputs/minimal_runtime_browsecomp_full830/compare_phase0_full830.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # also mirror into artifacts/baselines
    art = _REPO / "artifacts/baselines"
    art.mkdir(parents=True, exist_ok=True)
    (art / "bare_metrics.json").write_text(
        json.dumps(bare_summary, indent=2) + "\n", encoding="utf-8"
    )
    (art / "full_harness_v1_metrics.json").write_text(
        json.dumps(report["harness_v1"], indent=2) + "\n", encoding="utf-8"
    )
    (art / "full_harness_v2_metrics.json").write_text(
        json.dumps(report["harness_v2"], indent=2) + "\n", encoding="utf-8"
    )
    (art / "minimal_runtime_metrics.json").write_text(
        json.dumps(report["minimal_runtime"], indent=2) + "\n", encoding="utf-8"
    )
    (art / "compare_phase0_full830.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
