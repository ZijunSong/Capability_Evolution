#!/usr/bin/env python3
"""Normalize harness rollout outputs into Phase-0 Minimal Runtime artifact names."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def summarize_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def col(key: str) -> list[float]:
        return [float(r.get(key) or 0.0) for r in rows]

    recalls = col("recall")
    traj = col("trajectory_recall")
    fa = col("final_answer_recall")
    prec = col("precision")
    reward = col("reward")
    turns = col("turns")
    curated = col("n_curated")
    pool = col("n_pool")
    errors = [r for r in rows if r.get("error")]
    return {
        "n": n,
        "recall": _mean(recalls),
        "trajectory_recall": _mean(traj),
        "final_answer_recall": _mean(fa),
        "precision": _mean(prec),
        "reward": _mean(reward),
        "recall_gt0_n": sum(1 for x in recalls if x > 0),
        "recall_gt0_rate": sum(1 for x in recalls if x > 0) / n,
        "traj_gt0_n": sum(1 for x in traj if x > 0),
        "traj_gt0_rate": sum(1 for x in traj if x > 0) / n,
        "fa_gt0_n": sum(1 for x in fa if x > 0),
        "fa_gt0_rate": sum(1 for x in fa if x > 0) / n,
        "mean_turns": _mean(turns),
        "mean_n_curated": _mean(curated),
        "mean_n_pool": _mean(pool),
        "error_n": len(errors),
        "error_rate": len(errors) / n,
        "mean_elapsed_s": _mean(col("elapsed_s")),
        "driver": rows[0].get("driver"),
        "policy": rows[0].get("policy"),
        "model": rows[0].get("model"),
    }


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--harness-config", required=True)
    p.add_argument("--scope-config", default="")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]

    src_jsonl = out / "harness_rollouts.jsonl"
    if not src_jsonl.exists():
        raise SystemExit(f"missing {src_jsonl}")

    rows = [
        json.loads(line)
        for line in src_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # episodes.jsonl (canonical Phase-0 name)
    episodes_path = out / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # errors.jsonl
    errors_path = out / "errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row.get("error"):
                fh.write(
                    json.dumps(
                        {
                            "query_id": row.get("query_id"),
                            "error_message": row.get("error_message"),
                            "turns": row.get("turns"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # events.jsonl — episode-level only (tool-level events not emitted by this pipeline)
    events_path = out / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "event_type": "episode_summary",
                        "query_id": row.get("query_id"),
                        "turns": row.get("turns"),
                        "recall": row.get("recall"),
                        "trajectory_recall": row.get("trajectory_recall"),
                        "final_answer_recall": row.get("final_answer_recall"),
                        "n_curated": row.get("n_curated"),
                        "n_pool": row.get("n_pool"),
                        "error": row.get("error"),
                        "driver": row.get("driver"),
                        "early_end_blocks": row.get("early_end_blocks"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # resolved_config.yaml
    resolved_src = out / "harness_resolved_config.yaml"
    resolved_dst = out / "resolved_config.yaml"
    if resolved_src.exists():
        resolved_dst.write_text(resolved_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # fallback: copy harness yaml
        resolved_dst.write_text(
            Path(args.harness_config).read_text(encoding="utf-8"), encoding="utf-8"
        )

    summary = summarize_episodes(rows)
    summary.update(
        {
            "runtime": "minimal_runtime",
            "harness_config": str(args.harness_config),
            "scope_config": args.scope_config or None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "events_note": (
                "episode-level summaries only; tool-level shadow/OPD events disabled"
            ),
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    legacy_manifest = {}
    legacy_path = out / "harness_rollout_manifest.json"
    if legacy_path.exists():
        legacy_manifest = json.loads(legacy_path.read_text(encoding="utf-8"))

    manifest = {
        "runtime": "minimal_runtime",
        "phase": "phase0",
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "BrowseComp+",
        "n_queries": summary.get("n", 0),
        "retriever": "BM25",
        "bm25_index_path": legacy_manifest.get("bm25_index_path"),
        "max_turns": legacy_manifest.get("max_turns", 35),
        "max_tokens": legacy_manifest.get("max_tokens", 2048),
        "temperature": legacy_manifest.get("temperature", 1.0),
        "max_model_len": legacy_manifest.get("max_model_len", 32768),
        "parallel": legacy_manifest.get("parallel", 2),
        "policy_backend": legacy_manifest.get("policy_backend", "api"),
        "driver": summary.get("driver"),
        "harness_config": str(args.harness_config),
        "scope_config": args.scope_config or None,
        "git_commit": git_commit(repo),
        "v8d_env": {
            k: os.environ.get(k)
            for k in (
                "V8D_SUBTRACTIVE_CURATION",
                "V8D_IMPORTANCE_TAGGING",
                "V8D_AUTO_POPULATE_FIRST_SEARCH",
                "V8D_EVIDENCE_GRAPH",
                "V8D_SENTENCE_COMPRESS",
                "V8D_CONTENT_DEDUP",
                "V8D_VERIFY_TOOL",
                "V8D_TOKEN_BUDGET_MARKER",
            )
        },
        "artifacts": {
            "episodes": str(episodes_path),
            "events": str(events_path),
            "summary": str(out / "summary.json"),
            "resolved_config": str(resolved_dst),
            "errors": str(errors_path),
        },
        "summary": summary,
        "legacy_harness_manifest": legacy_manifest,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "n": summary.get("n"), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
