#!/usr/bin/env python3
"""Wave 4 diagnostic: compare 4 checkpoints with dup operation plumbing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round2.eval_paired import load_jsonl, summarize


def merge_shards(shard_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps: list[dict] = []
    states: list[dict] = []
    events: list[dict] = []
    for d in shard_dirs:
        eps.extend(load_jsonl(d / "episodes.jsonl"))
        if (d / "decision_states.jsonl").exists():
            states.extend(load_jsonl(d / "decision_states.jsonl"))
        if (d / "dup_admission_events.jsonl").exists():
            events.extend(load_jsonl(d / "dup_admission_events.jsonl"))
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in eps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if states:
        with (out_dir / "decision_states.jsonl").open("w", encoding="utf-8") as f:
            for row in states:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if events:
        with (out_dir / "dup_admission_events.jsonl").open("w", encoding="utf-8") as f:
            for row in events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summ = summarize(eps)
    summ["n_episodes"] = len(eps)
    summ["telemetry_events"] = len(events)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=_REPO / "outputs/scope_round3/wave4_diagnostic")
    args = p.parse_args()
    root = args.root
    variants = {
        "base": [root / "base/shard0", root / "base/shard1"],
        "round1": [root / "round1/shard0", root / "round1/shard1"],
        "round2_main": [root / "round2_main/shard0", root / "round2_main/shard1"],
        "round2_legacy": [root / "round2_legacy/shard0", root / "round2_legacy/shard1"],
    }
    report: dict = {}
    for name, shards in variants.items():
        existing = [s for s in shards if (s / "episodes.jsonl").exists()]
        if not existing:
            continue
        merge_dir = root / name / "merged"
        merge_shards(existing, merge_dir)
        eps = load_jsonl(merge_dir / "episodes.jsonl")
        report[name] = summarize(eps)
        report[name]["plumbing_ok"] = all(
            ep.get("dup_telemetry", {}).get("telemetry_complete", True) for ep in eps[:5]
        ) if name != "base" else True

    comparison = {
        "variants": report,
        "wave4_barrier": all(
            v.get("plumbing_ok", True) for v in report.values()
        ),
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = ["# Wave 4 Diagnostic\n", "| Variant | DupCurateRate | FalseSkipRate | Recall | Reward | Plumbing |"]
    lines.append("|---------|---------------|---------------|--------|--------|----------|")
    for name, m in report.items():
        lines.append(
            f"| {name} | {m.get('duplicate_curate_rate', m.get('dup_curate_rate', 'n/a')):.4f} | "
            f"{m.get('false_skip_rate', 0):.4f} | {m.get('recall', 0):.4f} | "
            f"{m.get('reward', 0):.4f} | {m.get('plumbing_ok', True)} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
