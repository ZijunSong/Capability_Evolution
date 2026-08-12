#!/usr/bin/env python3
"""Generate H100-3 same-state influence artifacts.

This runner exercises the SCAPE same-environment-state contract using the
repository's deterministic offline rollout/scorer. It is intentionally marked as
an offline scorer in the manifest; production runs can swap in Harness-1/model
policies while preserving the artifact schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import all_component_ids
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.probes.influence import InfluenceSample, aggregate_influence, score_influence_on_snapshot
from scape.probes.rollout import FakeSearchEnv, replay_parity, student_rollout_collect
from scape.rendering.dual_view import DualViewRenderer

TOOL_TYPES = ("search", "read", "review", "curate", "verify", "end")


def _phase_for_step(step: int) -> str:
    return TOOL_TYPES[min(step, len(TOOL_TYPES) - 1)]


def _bucket_for_component(component_id: str, step: int) -> dict[str, str]:
    if component_id == "evidence_graph":
        return {"graph_stage": ("empty", "sparse", "mature")[min(step, 2)]}
    if component_id == "verify_tool":
        return {"verify_support": "verify-eligible" if step >= 3 else "verify-ineligible"}
    if component_id == "token_budget_marker":
        return {"budget_bucket": ("early", "middle", "late", "near-limit")[min(step, 3)]}
    if component_id == "auto_populate_first_search":
        return {"first_action_region": "first-search" if step <= 1 else "later"}
    return {}


def _student_action(_view: dict[str, Any], snap) -> dict[str, Any]:
    phase = _phase_for_step(snap.step)
    if phase == "search":
        return {"name": "search", "arguments": {"query": snap.query_id}}
    if phase == "curate":
        return {"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}}
    if phase == "verify":
        return {"name": "verify", "arguments": {"doc_id": "d1"}}
    if phase == "end":
        return {"name": "end_search", "arguments": {}}
    return {"name": "read_document", "arguments": {"doc_id": "d1"}}


def _policy_probs(preferred: str, influence_strength: float) -> dict[str, float]:
    names = ["search", "grep", "read_document", "curate", "verify", "end_search"]
    low = max(0.01, (1.0 - influence_strength) / (len(names) - 1))
    probs = {name: low for name in names}
    probs[preferred] = influence_strength
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()}


def _component_strength(component_id: str, step: int) -> float:
    base = {
        "subtractive_curation": 0.88,
        "importance_tagging": 0.82,
        "auto_populate_first_search": 0.74 if step <= 1 else 0.55,
        "evidence_graph": 0.50 + min(step, 3) * 0.10,
        "sentence_compress": 0.70,
        "verify_tool": 0.82 if step >= 3 else 0.52,
        "adaptive_rerank_instruction": 0.68,
        "content_dedup": 0.50,
        "chunk_neighbors": 0.48,
        "token_budget_marker": 0.46 + min(step, 3) * 0.05,
    }.get(component_id, 0.55)
    return min(0.93, max(0.34, base))


def _student_policy(view: dict[str, Any]) -> dict[str, Any]:
    step = int(view.get("step", 0))
    phase = _phase_for_step(step)
    decoded = {
        "search": {"name": "search", "arguments": {"query": view.get("query_id", "")}},
        "read": {"name": "read_document", "arguments": {"doc_id": "d1"}},
        "review": {"name": "read_document", "arguments": {"doc_id": "d1"}},
        "curate": {"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}},
        "verify": {"name": "verify", "arguments": {"doc_id": "d1"}},
        "end": {"name": "end_search", "arguments": {}},
    }[phase]
    return {"tool_name_probs": _policy_probs(decoded["name"], 0.62), "decoded": decoded}


def _teacher_policy(component_id: str, view: dict[str, Any]) -> dict[str, Any]:
    step = int(view.get("step", 0))
    phase = _phase_for_step(step)
    strength = _component_strength(component_id, step)
    preferred = {
        "search": "search",
        "read": "read_document",
        "review": "curate" if component_id in {"subtractive_curation", "importance_tagging"} else "read_document",
        "curate": "curate",
        "verify": "verify" if component_id == "verify_tool" else "curate",
        "end": "end_search",
    }[phase]
    decoded_args: dict[str, Any]
    if preferred == "search":
        decoded_args = {"query": f"{view.get('query_id', '')} evidence"}
    elif preferred == "read_document":
        decoded_args = {"doc_id": "d1"}
    elif preferred == "curate":
        decoded_args = {"add_ids": ["d1"], "remove_ids": []}
    elif preferred == "verify":
        decoded_args = {"doc_id": "d1"}
    else:
        decoded_args = {}
    return {
        "tool_name_probs": _policy_probs(preferred, strength),
        "decoded": {"name": preferred, "arguments": decoded_args},
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _write_snapshot_schema(path: Path) -> None:
    path.write_text(
        "# SNAPSHOT_SCHEMA\n\n"
        "`xi_t` is a serializable `EnvironmentSnapshot` captured from reduced "
        "student rollout under `H_-m`. Hashing excludes ephemeral metadata and "
        "uses sorted JSON keys. Required stable fields: `query_id`, `step`, "
        "`created_at_step`, `harness_mask`, `working_memory`, `tool_history`, "
        "`observations`, and non-ephemeral `metadata`. The full-view teacher "
        "renders from the same snapshot and must not step the environment.\n",
        encoding="utf-8",
    )


def _write_dual_view_parity(path: Path, parity_rows: list[dict[str, Any]]) -> None:
    n = len(parity_rows)
    same = sum(1 for r in parity_rows if r["same_snapshot"])
    diff = sum(1 for r in parity_rows if r["views_differ"])
    path.write_text(
        "# DUAL_VIEW_PARITY\n\n"
        f"- n_pairs: {n}\n"
        f"- same_snapshot: {same}/{n}\n"
        f"- views_differ: {diff}/{n}\n"
        "- full_teacher_independent_trajectory: forbidden/not used\n",
        encoding="utf-8",
    )


def _write_null_report(path: Path, rows: list[dict[str, Any]]) -> None:
    null_same = [float(r["null_I_name"]) for r in rows]
    norm = [float(r["normalized_influence"]) for r in rows]
    path.write_text(
        "# NULL_CONTROL_REPORT\n\n"
        "Null controls are computed on the same student-owned snapshots. `N0` is "
        "same render vs same render; `N1` is field-order-only perturbation.\n\n"
        f"- components: {len(rows)}\n"
        f"- mean_null_I_name: {(sum(null_same) / len(null_same) if null_same else 0.0):.6f}\n"
        f"- mean_normalized_influence: {(sum(norm) / len(norm) if norm else 0.0):.6f}\n"
        "- scorer: deterministic_offline_stub; replace with model logprob scorer for final H100 report\n",
        encoding="utf-8",
    )


def _write_component_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# INFLUENCE_BY_COMPONENT",
        "",
        "| component | n_queries | n_states | event_support | I_name_mean | I_name_median | I_arg_key | I_arg_value | null_I_name | normalized | support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['component']} | {r['n_queries']} | {r['n_states']} | {r['event_support']} | "
            f"{r['I_name_mean']:.6f} | {r['I_name_median']:.6f} | "
            f"{r['I_arg_key']:.6f} | {r['I_arg_value']:.6f} | "
            f"{r['null_I_name']:.6f} | {r['normalized_influence']:.6f} | {r['support_label']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--max-states-per-query", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_3_influence")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_id="h100_3_influence_offline_cal64",
        stage="h100_3_influence",
        command=["python", "scripts/run_h100_3_influence.py"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "n_queries": args.n_queries,
            "max_states_per_query": args.max_states_per_query,
            "scorer": "deterministic_offline_stub",
            "training": False,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    renderer = DualViewRenderer()
    per_state_path = out / "INFLUENCE_PER_STATE.jsonl"
    component_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    completed: list[str] = []

    with per_state_path.open("w", encoding="utf-8") as state_f:
        for cid in all_component_ids():
            samples: list[InfluenceSample] = []
            event_support = 0
            by_tool: dict[str, list[float]] = {t: [] for t in TOOL_TYPES}
            for qidx in range(args.n_queries):
                env = FakeSearchEnv(query_id=f"inf_{cid}_q{qidx:03d}", component_id=cid, max_steps=args.max_states_per_query)
                snaps = student_rollout_collect(env, _student_action, n_steps=args.max_states_per_query - 1)
                for snap in snaps[: args.max_states_per_query]:
                    sample = score_influence_on_snapshot(
                        snap,
                        component_id=cid,
                        student_policy=_student_policy,
                        teacher_policy=lambda view, _cid=cid: _teacher_policy(_cid, view),
                        renderer=renderer,
                    )
                    tool_type = _phase_for_step(snap.step)
                    by_tool[tool_type].append(sample.I_name)
                    event_support += 1
                    record = {
                        "component": cid,
                        "snapshot_hash": sample.snapshot_hash,
                        "query_id": sample.query_id,
                        "step": sample.step,
                        "tool_type": tool_type,
                        "I_name": sample.I_name,
                        "I_args": sample.I_args,
                        "I_arg_key": sample.I_args * 0.4,
                        "I_arg_value": sample.I_args * 0.6,
                        **_bucket_for_component(cid, snap.step),
                        **sample.extras,
                    }
                    state_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    samples.append(sample)
                    parity_rows.append(replay_parity(snap, component_id=cid, renderer=renderer))

            agg = aggregate_influence(samples)
            name_values = [s.I_name for s in samples]
            null_i = max(float(agg["null_same_render_mean"]), float(agg["null_field_order_mean"]))
            normalized = float(agg["I_name_mean"] - null_i)
            row = {
                "component": cid,
                "n_queries": args.n_queries,
                "n_states": len(samples),
                "event_support": event_support,
                "I_name_mean": float(agg["I_name_mean"]),
                "I_name_median": _median(name_values),
                "I_arg_key": float(agg["I_args_mean"]) * 0.4,
                "I_arg_value": float(agg["I_args_mean"]) * 0.6,
                "tool_name_disagreement": sum(float(s.extras.get("tool_name_disagreement", 0.0)) for s in samples) / len(samples),
                "exact_call_disagreement": sum(float(s.extras.get("exact_tool_call_disagreement", 0.0)) for s in samples) / len(samples),
                "null_I_name": null_i,
                "normalized_influence": normalized,
                "support_label": "OK" if event_support >= args.n_queries else "LOW_EVENT_SUPPORT",
                "by_tool_type": {k: (sum(v) / len(v) if v else 0.0) for k, v in by_tool.items()},
                "scorer": "deterministic_offline_stub",
            }
            component_rows.append(row)
            completed.append(cid)
            write_status_live(
                out / "STATUS_LIVE.md",
                stage="h100_3_influence",
                run_id=manifest["run_id"],
                n_expected=len(all_component_ids()),
                n_finished=len(completed),
                errors=[],
                extra={"last_component": cid, "scorer": "deterministic_offline_stub"},
            )

    csv_path = out / "INFLUENCE_BY_COMPONENT.csv"
    fieldnames = [
        "component",
        "n_queries",
        "n_states",
        "event_support",
        "I_name_mean",
        "I_name_median",
        "I_arg_key",
        "I_arg_value",
        "tool_name_disagreement",
        "exact_call_disagreement",
        "null_I_name",
        "normalized_influence",
        "support_label",
        "scorer",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(component_rows)

    (out / "INFLUENCE_BY_COMPONENT.json").write_text(
        json.dumps(component_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_component_markdown(out / "INFLUENCE_BY_COMPONENT.md", component_rows)
    _write_snapshot_schema(out / "SNAPSHOT_SCHEMA.md")
    _write_dual_view_parity(out / "DUAL_VIEW_PARITY.md", parity_rows)
    _write_null_report(out / "NULL_CONTROL_REPORT.md", component_rows)

    files = [
        out / "RUN_MANIFEST.json",
        out / "STATUS_LIVE.md",
        out / "SNAPSHOT_SCHEMA.md",
        out / "DUAL_VIEW_PARITY.md",
        out / "INFLUENCE_PER_STATE.jsonl",
        out / "INFLUENCE_BY_COMPONENT.csv",
        out / "INFLUENCE_BY_COMPONENT.md",
        out / "INFLUENCE_BY_COMPONENT.json",
        out / "NULL_CONTROL_REPORT.md",
    ]
    write_sha256sums(out, files)
    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=completed),
    )
    # Recompute after final manifest update.
    write_sha256sums(out, files)
    print(json.dumps({"out_dir": str(out), "components": len(component_rows)}, indent=2))


if __name__ == "__main__":
    main()
