#!/usr/bin/env python3
"""Provisional LOCAL_CAL64 LOO + Influence bootstrap when H100 imports are absent.

This does NOT replace H100 contribution/influence. It only produces a ranking
seed so H20 learnability scaffolding can proceed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import all_component_ids, component_specs
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.status import write_status_live
from scape.probes.candidate_selector import select_candidates, write_placement_map
from scape.probes.contribution import contribution_report
from scape.probes.influence import aggregate_influence, score_influence_on_snapshot
from scape.probes.rollout import FakeSearchEnv, student_rollout_collect

REPO = Path(__file__).resolve().parents[1]


def _fake_metrics(qid: str, *, boost: float) -> dict:
    # Deterministic pseudo-metrics for scaffolding only
    base = (sum(ord(c) for c in qid) % 100) / 100.0
    return {
        "curated_recall": min(1.0, base * 0.5 + boost),
        "trajectory_recall": min(1.0, base * 0.4 + boost * 0.8),
        "final_answer_recall": min(1.0, base * 0.3 + boost * 0.5),
        "harness_reward": min(1.0, base * 0.4 + boost * 0.6),
        "tool_calls": 10 - boost * 5,
        "turns": 8 - boost * 2,
        "context_tokens": 4000 - boost * 500,
        "latency_ms": 1000 - boost * 100,
        "state_ops": 20 - boost * 3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "local_cal64")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        run_id="local_cal64",
        stage="prestage_bootstrap",
        command=["python", "-m", "scripts.local_cal64_bootstrap"],
        repo_root=REPO,
        output_dir=out,
        extra={"provisional": True, "n": args.n},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    qids = [f"cal64_q{i:03d}" for i in range(args.n)]
    full = {q: _fake_metrics(q, boost=0.08) for q in qids}

    rows = []
    contrib_all = {}
    for cid in all_component_ids():
        # Components that change context get a small synthetic contribution signal
        spec = next(s for s in component_specs() if s.component_id == cid)
        boost_drop = 0.05 if spec.changes_context and spec.semantic_or_runtime == "semantic" else 0.01
        minus = {q: _fake_metrics(q, boost=0.08 - boost_drop) for q in qids}
        report = contribution_report(cid, full, minus, n_boot=200, seed=42)
        contrib_all[cid] = report

        # Influence on a few student-owned snapshots
        env = FakeSearchEnv(query_id=f"inf_{cid}", component_id=cid)
        snaps = student_rollout_collect(
            env, lambda _v, s: {"name": "search", "arguments": {"query": s.query_id}}, n_steps=1
        )

        def student_pol(view, _cid=cid):
            return {
                "tool_name_probs": {"search": 0.6, "curate": 0.4},
                "decoded": {"name": "search", "arguments": {"query": "x"}},
            }

        def teacher_pol(view, _cid=cid):
            # Larger divergence for semantic context changers
            peak = 0.85 if boost_drop >= 0.05 else 0.55
            other = 1.0 - peak
            return {
                "tool_name_probs": {"search": other, "curate": peak},
                "decoded": {"name": "curate", "arguments": {"add_ids": ["d1"]}},
            }

        samples = [
            score_influence_on_snapshot(
                snap,
                component_id=cid,
                student_policy=student_pol,
                teacher_policy=teacher_pol,
            )
            for snap in snaps
        ]
        infl = aggregate_influence(samples)
        rows.append(
            {
                "component_id": cid,
                "contribution": float(
                    report["metrics"].get("curated_recall", {}).get("mean_delta", 0.0)
                ),
                "influence_above_null": float(infl["I_name_mean"] - infl["null_field_order_mean"]),
                # Positive cost means the component costs extra runtime/context to keep.
                "runtime_cost": float(minus[qids[0]]["context_tokens"] - full[qids[0]]["context_tokens"] + 1.0),
                "quality_positive": bool(report["quality_positive"]),
                "provisional": True,
            }
        )

    (out / "COMPONENT_CONTRIBUTION.json").write_text(
        json.dumps(contrib_all, indent=2) + "\n", encoding="utf-8"
    )
    result = select_candidates(rows, top_k=2)
    place_dir = REPO / "outputs" / "scape_prestage"
    write_placement_map(result, place_dir)
    (out / "PROVISIONAL_ROWS.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    write_status_live(
        out / "STATUS_LIVE.md",
        stage="local_cal64",
        run_id="local_cal64",
        n_expected=len(all_component_ids()),
        n_finished=len(all_component_ids()),
        errors=[],
        extra={"candidates": result["candidates"]},
    )
    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=list(all_component_ids())),
    )
    print(json.dumps(result["candidates"], indent=2))


if __name__ == "__main__":
    main()
