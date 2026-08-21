#!/usr/bin/env python3
"""Formal importance_tagging always-on versus always-off live fork.

The frozen cohort is reconstructed from the original deterministic collection
protocol and accepted only when every ordered snapshot hash matches the source
K4/K8 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import run_h100_2_live_fork_replay as base
import run_h100_2_live_fork_replay_stream as stream
from scape.common.sha256sums import write_sha256sums
from scape.rendering.dual_view import DualViewRenderer

COMPONENT = "importance_tagging"
SEEDS = (8423, 8424)
HORIZONS = (4, 8)
SOURCE = REPO / "outputs/0820_importance_tagging_single_128_rerun"
OUT_DEFAULT = REPO / "outputs/0821_importance_tagging_always_on_off_256"
MODEL_DEFAULT = Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
BCP_DEFAULT = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_path(seed: int, k: int) -> Path:
    return SOURCE / f"K{k}_seed{seed}" / "shards" / f"{COMPONENT}_K{k}.jsonl"


def canonical_action(action: Mapping[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_branch(
    start: base.LiveState,
    first_action: Mapping[str, Any],
    *,
    k: int,
    scorer: stream.BatchedHFContinuationScorer,
    renderer: DualViewRenderer,
    label: str,
    full: bool,
) -> tuple[base.LiveState, list[dict[str, Any]]]:
    st = start.clone(label)
    trace: list[dict[str, Any]] = []
    st.execute(first_action)
    trace.append({
        "branch": label,
        "phase": "forced_first",
        "policy_view": "full" if full else "reduced",
        "action": dict(first_action),
        "metrics": st.metrics(),
    })
    for index in range(k):
        action, dist, dual = base.policy_action(
            st, scorer, renderer, component=COMPONENT, full=full
        )
        st.execute(action)
        trace.append({
            "branch": label,
            "phase": f"continue_{index + 1}",
            "policy_view": "full" if full else "reduced",
            "action": action,
            "top_prob": max(dist["tool_name_probs"].values()),
            "snapshot_hash": dual["snapshot_hash"],
            "metrics": st.metrics(),
        })
    return st, trace


def reconstruct_states(
    *,
    seed: int,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    searcher: Any,
    scorer: stream.BatchedHFContinuationScorer,
    renderer: DualViewRenderer,
    n_states: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src4 = load_jsonl(source_path(seed, 4))
    src8 = load_jsonl(source_path(seed, 8))
    ordered4 = [(str(row["query_id"]), str(row["snapshot_hash"])) for row in src4]
    ordered8 = [(str(row["query_id"]), str(row["snapshot_hash"])) for row in src8]
    if ordered4 != ordered8:
        raise RuntimeError(f"source K4/K8 frozen cohort mismatch for seed {seed}")
    if len(ordered4) != n_states:
        raise RuntimeError(f"source seed {seed} has {len(ordered4)} states, expected {n_states}")

    manifest = json.loads(
        (SOURCE / f"K4_seed{seed}" / "manifests/UTILITY_LIVE256.json").read_text(encoding="utf-8")
    )
    qids = [str(qid) for qid in manifest["query_ids"]]
    states = []
    for item in stream.candidate_items(COMPONENT, qids, queries, qrels, searcher, scorer, renderer):
        states.append(item)
        if len(states) >= n_states:
            break
    rebuilt = [(str(item["query_id"]), str(item["snapshot_hash"])) for item in states]
    first_actions_match = sum(
        canonical_action(item["a_S"]) == canonical_action(src4[index]["a_S"])
        and canonical_action(item["a_T"]) == canonical_action(src4[index]["a_T"])
        for index, item in enumerate(states)
    )
    audit = {
        "seed": seed,
        "source_k4_k8_ordered_match": ordered4 == ordered8,
        "source_n": len(ordered4),
        "rebuilt_n": len(rebuilt),
        "ordered_query_snapshot_matches": sum(a == b for a, b in zip(ordered4, rebuilt)),
        "ordered_query_snapshot_all_match": rebuilt == ordered4,
        "source_first_actions_matches": first_actions_match,
        "source_first_actions_all_match": first_actions_match == n_states,
    }
    if rebuilt != ordered4 or first_actions_match != n_states:
        raise RuntimeError(f"frozen-state reconstruction failed: {json.dumps(audit, sort_keys=True)}")
    return states, audit


def run_cell(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    audit_dir = args.out_dir / "audits"
    shard_dir.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)

    queries = base._load_queries(args.browsecomp_root / "topics-qrels/queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels/qrel_evidence.txt")
    searcher, search_backend = base.build_searcher(args.index_path, args.corpus_path)
    scorer = stream.BatchedHFContinuationScorer(
        args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens
    )
    renderer = DualViewRenderer()
    states, reconstruction = reconstruct_states(
        seed=args.seed,
        queries=queries,
        qrels=qrels,
        searcher=searcher,
        scorer=scorer,
        renderer=renderer,
        n_states=args.n_states,
    )

    output = shard_dir / f"IMPORTANCE_ALWAYS_ON_OFF_K{args.K}_SEED{args.seed}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(states):
            start = base.state_from_snapshot(
                item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, COMPONENT
            )
            student, trace_s = run_branch(
                start, item["a_S"], k=args.K, scorer=scorer, renderer=renderer, label="S", full=False
            )
            teacher, trace_t = run_branch(
                start, item["a_T"], k=args.K, scorer=scorer, renderer=renderer, label="T", full=True
            )
            metrics_s = student.metrics()
            metrics_t = teacher.metrics()
            row = {
                "schema_version": "importance_always_on_off_v1",
                "component": COMPONENT,
                "contract": "Teacher Full importance_tagging on at every decision; Student Reduced importance_tagging off at every decision",
                "seed": args.seed,
                "K": args.K,
                "state_id": f"importance_always_K{args.K}_seed{args.seed}_{index:03d}",
                "query_id": item["query_id"],
                "turn_id": item["turn_id"],
                "snapshot_hash": item["snapshot_hash"],
                "branch_S_initial_state_hash": start.snapshot().content_hash(),
                "branch_T_initial_state_hash": start.snapshot().content_hash(),
                "branch_S_final_state_hash": student.snapshot().content_hash(),
                "branch_T_final_state_hash": teacher.snapshot().content_hash(),
                "a_S": item["a_S"],
                "a_T": item["a_T"],
                "first_action_disagreement": int(canonical_action(item["a_T"]) != canonical_action(item["a_S"])),
                "branch_S_metrics": metrics_s,
                "branch_T_metrics": metrics_t,
                "tool_cost_delta": metrics_t["tool_search_cost"] - metrics_s["tool_search_cost"],
                "utility_delta": metrics_t["objective_utility"] - metrics_s["objective_utility"],
                "branch_S_trace": trace_s,
                "branch_T_trace": trace_t,
                "teacher_policy_views": sorted({step["policy_view"] for step in trace_t}),
                "student_policy_views": sorted({step["policy_view"] for step in trace_s}),
                "full_harness_takeover": False,
                "search_backend": search_backend,
                "runner": "importance_tagging_always_on_off_hf_bm25_batched",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 8 == 0:
                print(json.dumps({"seed": args.seed, "K": args.K, "finished": index + 1}), flush=True)

    cell_audit = {
        **reconstruction,
        "K": args.K,
        "output": str(output),
        "output_rows": args.n_states,
        "source_snapshot_sha256": hashlib.sha256(
            "\n".join(item["snapshot_hash"] for item in states).encode()
        ).hexdigest(),
    }
    (audit_dir / f"RECONSTRUCTION_K{args.K}_SEED{args.seed}.json").write_text(
        json.dumps(cell_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(cell_audit, indent=2, ensure_ascii=False), flush=True)
    return 0


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
    }


def aggregate(args: argparse.Namespace) -> int:
    rows_all: list[dict[str, Any]] = []
    reconstruction_audits = []
    for k in HORIZONS:
        for seed in SEEDS:
            path = args.out_dir / "shards" / f"IMPORTANCE_ALWAYS_ON_OFF_K{k}_SEED{seed}.jsonl"
            rows = load_jsonl(path)
            if len(rows) != args.n_states:
                raise RuntimeError(f"{path} has {len(rows)} rows, expected {args.n_states}")
            rows_all.extend(rows)
            reconstruction_audits.append(json.loads(
                (args.out_dir / "audits" / f"RECONSTRUCTION_K{k}_SEED{seed}.json").read_text(encoding="utf-8")
            ))

    summaries = []
    for k in HORIZONS:
        rows = [row for row in rows_all if int(row["K"]) == k]
        ordered = {
            seed: [(row["query_id"], row["snapshot_hash"]) for row in rows if int(row["seed"]) == seed]
            for seed in SEEDS
        }
        source_k4_k8_matches = {
            str(seed): ordered[seed] == [
                (row["query_id"], row["snapshot_hash"])
                for row in rows_all
                if int(row["seed"]) == seed and int(row["K"]) == HORIZONS[0]
            ]
            for seed in SEEDS
        }
        disagreement = stats(rows, "first_action_disagreement")
        cost = stats(rows, "tool_cost_delta")
        utility = stats(rows, "utility_delta")
        summaries.append({
            "K": k,
            "n_paired_states": len(rows),
            "first_action_disagreement_rate": disagreement["mean"],
            "first_action_disagreement_percent": disagreement["mean"] * 100.0,
            "tool_cost_delta": cost,
            "utility_delta": utility,
            "utility_delta_percent": utility["mean"] * 100.0,
            "teacher_cost_mean": statistics.mean(float(row["branch_T_metrics"]["tool_search_cost"]) for row in rows),
            "student_cost_mean": statistics.mean(float(row["branch_S_metrics"]["tool_search_cost"]) for row in rows),
            "teacher_utility_mean": statistics.mean(float(row["branch_T_metrics"]["objective_utility"]) for row in rows),
            "student_utility_mean": statistics.mean(float(row["branch_S_metrics"]["objective_utility"]) for row in rows),
            "branch_initial_hash_mismatch": sum(
                row["branch_S_initial_state_hash"] != row["branch_T_initial_state_hash"]
                for row in rows
            ),
            "teacher_policy_view_failure": sum(row["teacher_policy_views"] != ["full"] for row in rows),
            "student_policy_view_failure": sum(row["student_policy_views"] != ["reduced"] for row in rows),
            "full_harness_takeover": sum(bool(row["full_harness_takeover"]) for row in rows),
            "ordered_snapshot_matches_k4": source_k4_k8_matches,
        })

    if any(
        summary["n_paired_states"] != args.n_states * len(SEEDS)
        or summary["branch_initial_hash_mismatch"]
        or summary["teacher_policy_view_failure"]
        or summary["student_policy_view_failure"]
        or summary["full_harness_takeover"]
        or not all(summary["ordered_snapshot_matches_k4"].values())
        for summary in summaries
    ):
        raise RuntimeError("aggregate audit failed")
    if not all(audit["ordered_query_snapshot_all_match"] and audit["source_first_actions_all_match"] for audit in reconstruction_audits):
        raise RuntimeError("reconstruction audit failed")

    per_state = args.out_dir / "IMPORTANCE_ALWAYS_ON_OFF_PER_STATE.jsonl"
    with per_state.open("w", encoding="utf-8") as handle:
        for row in rows_all:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "completed_formal",
        "component": COMPONENT,
        "contract": "Teacher-always-on Full view versus Student-always-off Reduced view; forced first action plus K continuation actions",
        "seeds": list(SEEDS),
        "states_per_seed": args.n_states,
        "paired_states_per_K": args.n_states * len(SEEDS),
        "horizons": list(HORIZONS),
        "model": str(args.model),
        "source": str(SOURCE),
        "summaries": summaries,
        "reconstruction_audits": reconstruction_audits,
    }
    (args.out_dir / "IMPORTANCE_ALWAYS_ON_OFF_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# importance_tagging always-on/off gain",
        "",
        "- contract: Teacher Full view at every decision; Student Reduced view at every decision",
        f"- frozen cohort: seeds {SEEDS[0]}/{SEEDS[1]}, {args.n_states} states each",
        "",
        "| Horizon | paired states | first action disagreement | tool cost delta | Utility delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| K{summary['K']} | {summary['n_paired_states']} | "
            f"{summary['first_action_disagreement_percent']:.2f}% | "
            f"{summary['tool_cost_delta']['mean']:+.6f} | "
            f"{summary['utility_delta_percent']:+.4f}% |"
        )
    (args.out_dir / "IMPORTANCE_ALWAYS_ON_OFF_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = [path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    write_sha256sums(args.out_dir, files)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cell", "aggregate"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--K", type=int, choices=HORIZONS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--browsecomp-root", type=Path, default=BCP_DEFAULT)
    parser.add_argument("--index-path", type=Path, default=BCP_DEFAULT / "indexes/bm25")
    parser.add_argument("--corpus-path", type=Path, default=REPO / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--n-states", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    os.environ.setdefault("JAVA_HOME", "/opt/scape-jdk21")
    if args.mode == "cell":
        if args.seed is None or args.K is None:
            parser.error("--seed and --K are required in cell mode")
        return run_cell(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
