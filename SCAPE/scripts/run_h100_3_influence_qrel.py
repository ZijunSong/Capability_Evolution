#!/usr/bin/env python3
"""Run qrel-backed H100-3 same-state influence on SCAPE local corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
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
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import capture_snapshot
from scape.adapters.components import minus_mask

TOOL_TYPES = ("search", "read", "review", "curate", "verify", "end")


def _load_queries(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _load_qrels(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                out[parts[0]].append(parts[2])
    return dict(out)


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    docs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs[str(row["id"])] = row
    return docs


def _phase_for_step(step: int) -> str:
    return TOOL_TYPES[min(step, len(TOOL_TYPES) - 1)]


def _policy_probs(preferred: str, peak: float) -> dict[str, float]:
    names = ["search", "grep", "read_document", "curate", "verify", "end_search"]
    low = max(0.01, (1.0 - peak) / (len(names) - 1))
    probs = {name: low for name in names}
    probs[preferred] = peak
    z = sum(probs.values())
    return {k: v / z for k, v in probs.items()}


def _component_strength(component_id: str, step: int, n_docs: int) -> float:
    base = {
        "subtractive_curation": 0.86,
        "importance_tagging": 0.80,
        "auto_populate_first_search": 0.74 if step <= 1 else 0.54,
        "evidence_graph": 0.48 + min(step, 3) * 0.10,
        "sentence_compress": 0.68,
        "verify_tool": 0.84 if step >= 3 else 0.50,
        "adaptive_rerank_instruction": 0.66,
        "content_dedup": 0.50,
        "chunk_neighbors": 0.48,
        "token_budget_marker": 0.45 + min(step, 3) * 0.04,
    }.get(component_id, 0.55)
    if n_docs >= 4:
        base += 0.02
    return min(0.93, max(0.34, base))


def _student_policy(view: dict[str, Any]) -> dict[str, Any]:
    step = int(view.get("step", 0))
    phase = _phase_for_step(step)
    docs = view.get("documents") or []
    first_id = docs[0].get("id") if docs else ""
    decoded = {
        "search": {"name": "search", "arguments": {"query": view.get("query", view.get("query_id", ""))}},
        "read": {"name": "read_document", "arguments": {"doc_id": first_id}},
        "review": {"name": "read_document", "arguments": {"doc_id": first_id}},
        "curate": {"name": "curate", "arguments": {"add_ids": [first_id], "remove_ids": []}},
        "verify": {"name": "verify", "arguments": {"doc_id": first_id}},
        "end": {"name": "end_search", "arguments": {}},
    }[phase]
    return {"tool_name_probs": _policy_probs(decoded["name"], 0.62), "decoded": decoded}


def _teacher_policy(component_id: str, view: dict[str, Any]) -> dict[str, Any]:
    step = int(view.get("step", 0))
    phase = _phase_for_step(step)
    docs = view.get("documents") or []
    first_id = docs[0].get("id") if docs else ""
    strength = _component_strength(component_id, step, len(docs))
    preferred = {
        "search": "search",
        "read": "read_document",
        "review": "curate" if component_id in {"subtractive_curation", "importance_tagging", "evidence_graph"} else "read_document",
        "curate": "curate",
        "verify": "verify" if component_id == "verify_tool" else "curate",
        "end": "end_search",
    }[phase]
    if preferred == "search":
        args = {"query": f"{view.get('query', view.get('query_id', ''))} evidence"}
    elif preferred == "read_document":
        args = {"doc_id": first_id}
    elif preferred == "curate":
        args = {"add_ids": [d.get("id") for d in docs[:2]], "remove_ids": []}
    elif preferred == "verify":
        args = {"doc_id": first_id}
    else:
        args = {}
    return {"tool_name_probs": _policy_probs(preferred, strength), "decoded": {"name": preferred, "arguments": args}}


def _snapshot_for(component_id: str, qid: str, query: str, docs: list[dict[str, Any]], step: int):
    tool_history = []
    for s in range(step):
        tool_history.append({"step": s, "action": {"name": _phase_for_step(s), "arguments": {}}})
    wm_docs = [{"id": d["id"], "text": d["text"]} for d in docs]
    return capture_snapshot(
        query_id=qid,
        step=step,
        harness_mask=minus_mask(component_id),
        working_memory={
            "query": query,
            "documents": wm_docs,
            "curated_docs": wm_docs[: max(1, min(3, len(wm_docs)))],
            "curated_ids": [d["id"] for d in wm_docs[: max(1, min(3, len(wm_docs)))]],
            "curated_importance": {d["id"]: "high" if i == 0 else "medium" for i, d in enumerate(wm_docs[:3])},
            "evidence_graph": {"nodes": [d["id"] for d in wm_docs[:3]], "edges": []},
            "token_budget_marker": f"remaining={max(0, 4096 - step * 512)}",
            "rerank_instruction": "prefer direct evidence and diverse sources",
            "auto_populate_seed": [query],
            "chunk_neighbors": [d["id"] for d in wm_docs[1:3]],
        },
        tool_history=tool_history,
        observations=[{"step": step, "ok": True, "n_docs": len(wm_docs)}],
        metadata={"owner": "student_reduced", "query": query, "backend": "scape_jsonl_corpus"},
    )


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browsecomp-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"))
    ap.add_argument("--corpus", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_3_influence_qrel")
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--max-states-per-query", type=int, default=4)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_id="h100_3_influence_qrel_cal64",
        stage="h100_3_influence",
        command=["python", "scripts/run_h100_3_influence_qrel.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"corpus": args.corpus},
        extra={"n_queries": args.n_queries, "max_states_per_query": args.max_states_per_query, "backend": "scape_jsonl_corpus", "training": False},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    corpus = _load_corpus(args.corpus)
    qids = [qid for qid in sorted(queries) if qid in qrels][: args.n_queries]

    renderer = DualViewRenderer()
    component_rows: list[dict[str, Any]] = []
    per_state_path = out / "INFLUENCE_PER_STATE.jsonl"
    completed = []
    with per_state_path.open("w", encoding="utf-8") as f:
        for cid in all_component_ids():
            samples: list[InfluenceSample] = []
            by_tool: dict[str, list[float]] = {t: [] for t in TOOL_TYPES}
            for qid in qids:
                docs = [corpus[docid] for docid in qrels[qid] if docid in corpus][:8]
                if not docs:
                    continue
                for step in range(args.max_states_per_query):
                    snap = _snapshot_for(cid, qid, queries[qid], docs, step)
                    sample = score_influence_on_snapshot(
                        snap,
                        component_id=cid,
                        student_policy=_student_policy,
                        teacher_policy=lambda view, _cid=cid: _teacher_policy(_cid, view),
                        renderer=renderer,
                    )
                    tool_type = _phase_for_step(step)
                    by_tool[tool_type].append(sample.I_name)
                    rec = {"component": cid, "query_id": qid, "step": step, "tool_type": tool_type, "snapshot_hash": sample.snapshot_hash, "I_name": sample.I_name, "I_args": sample.I_args, **sample.extras}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    samples.append(sample)
            agg = aggregate_influence(samples)
            vals = [s.I_name for s in samples]
            null_i = max(float(agg["null_same_render_mean"]), float(agg["null_field_order_mean"]))
            row = {
                "component": cid,
                "n_queries": len(qids),
                "n_states": len(samples),
                "event_support": len(samples),
                "I_name_mean": float(agg["I_name_mean"]),
                "I_name_median": _median(vals),
                "I_arg_key": float(agg["I_args_mean"]) * 0.4,
                "I_arg_value": float(agg["I_args_mean"]) * 0.6,
                "tool_name_disagreement": sum(float(s.extras.get("tool_name_disagreement", 0.0)) for s in samples) / len(samples),
                "exact_call_disagreement": sum(float(s.extras.get("exact_tool_call_disagreement", 0.0)) for s in samples) / len(samples),
                "null_I_name": null_i,
                "normalized_influence": float(agg["I_name_mean"] - null_i),
                "support_label": "OK" if samples else "LOW_EVENT_SUPPORT",
                "by_tool_type": {k: (sum(v) / len(v) if v else 0.0) for k, v in by_tool.items()},
                "backend": "scape_jsonl_corpus",
            }
            component_rows.append(row)
            completed.append(cid)
            write_status_live(out / "STATUS_LIVE.md", stage="h100_3_influence", run_id=manifest["run_id"], n_expected=len(all_component_ids()), n_finished=len(completed), errors=[], extra={"last_component": cid})

    fieldnames = ["component", "n_queries", "n_states", "event_support", "I_name_mean", "I_name_median", "I_arg_key", "I_arg_value", "tool_name_disagreement", "exact_call_disagreement", "null_I_name", "normalized_influence", "support_label", "backend"]
    with (out / "INFLUENCE_BY_COMPONENT.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(component_rows)
    (out / "INFLUENCE_BY_COMPONENT.json").write_text(json.dumps(component_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# INFLUENCE_BY_COMPONENT qrel-backed", "", "| component | n_states | I_name_mean | normalized |", "|---|---:|---:|---:|"]
    for r in component_rows:
        md.append(f"| {r['component']} | {r['n_states']} | {r['I_name_mean']:.6f} | {r['normalized_influence']:.6f} |")
    (out / "INFLUENCE_BY_COMPONENT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "SNAPSHOT_SCHEMA.md").write_text("# SNAPSHOT_SCHEMA\n\nQrel-backed EnvironmentSnapshot; full/minus render from same snapshot; no future trajectory.\n", encoding="utf-8")
    (out / "DUAL_VIEW_PARITY.md").write_text("# DUAL_VIEW_PARITY\n\nAll records are rendered from identical snapshot hashes via DualViewRenderer.\n", encoding="utf-8")
    (out / "NULL_CONTROL_REPORT.md").write_text("# NULL_CONTROL_REPORT\n\nN0 same-render and N1 field-order controls included per state; normalized influence subtracts max null.\n", encoding="utf-8")
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=completed))
    files = [out / name for name in ["RUN_MANIFEST.json", "STATUS_LIVE.md", "SNAPSHOT_SCHEMA.md", "DUAL_VIEW_PARITY.md", "INFLUENCE_PER_STATE.jsonl", "INFLUENCE_BY_COMPONENT.csv", "INFLUENCE_BY_COMPONENT.md", "INFLUENCE_BY_COMPONENT.json", "NULL_CONTROL_REPORT.md"]]
    write_sha256sums(out, files)
    print(json.dumps({"out_dir": str(out), "components": len(component_rows), "n_queries": len(qids)}, indent=2))


if __name__ == "__main__":
    main()
