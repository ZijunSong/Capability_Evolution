#!/usr/bin/env python3
"""Sentence-compress always-on/off paired utility fork over frozen 128 states."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sentence_compress_formal_fork as base


def load_states(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_branch(
    start: base.LiveState,
    first_action: dict,
    *,
    k: int,
    scorer: base.HFContinuationScorer,
    renderer: base.DualViewRenderer,
    component: str,
    label: str,
    full: bool,
) -> tuple[base.LiveState, list[dict]]:
    state = start.clone(label)
    trace = []
    state.execute(first_action)
    trace.append({
        "branch": label,
        "phase": "forced_first",
        "view": "full" if full else "reduced",
        "action": dict(first_action),
        "metrics": state.metrics(),
    })
    for index in range(max(0, k - 1)):
        action, distribution, dual = base.policy_action(
            state,
            scorer,
            renderer,
            component=component,
            full=full,
        )
        state.execute(action)
        trace.append({
            "branch": label,
            "phase": f"continue_{index + 1}",
            "view": "full" if full else "reduced",
            "action": action,
            "top_prob": max(distribution["tool_name_probs"].values()),
            "snapshot_hash": dual["snapshot_hash"],
            "metrics": state.metrics(),
        })
    return state, trace


def run(args: argparse.Namespace) -> None:
    out = args.out_dir
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    states = load_states(args.states_manifest)
    if len(states) != 128:
        raise RuntimeError(f"expected 128 frozen states, found {len(states)}")

    searcher, backend = base.build_searcher(args.index_path, args.corpus_path)
    scorer = base.HFContinuationScorer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    renderer = base.DualViewRenderer()
    rows = []
    raw_path = raw_dir / f"sentence_compress_always_on_off_K{args.K}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(states):
            qid = str(item["query_id"])
            start = base.state_from_snapshot(
                item["snapshot"], queries[qid], qrels[qid], searcher, "sentence_compress"
            )
            reconstructed_hash = start.snapshot().content_hash()
            frozen_hash = base.EnvironmentSnapshot.from_dict(item["snapshot"]).content_hash()
            if frozen_hash != item["snapshot_hash"]:
                raise RuntimeError(
                    f"frozen snapshot integrity failure at row {index}: "
                    f"{frozen_hash} != {item['snapshot_hash']}"
                )
            student_final, student_trace = run_branch(
                start,
                item["a_S"],
                k=args.K,
                scorer=scorer,
                renderer=renderer,
                component="sentence_compress",
                label="S",
                full=False,
            )
            teacher_final, teacher_trace = run_branch(
                start,
                item["a_T"],
                k=args.K,
                scorer=scorer,
                renderer=renderer,
                component="sentence_compress",
                label="T",
                full=True,
            )
            student_metrics = student_final.metrics()
            teacher_metrics = teacher_final.metrics()
            student_actions = [step["action"] for step in student_trace]
            teacher_actions = [step["action"] for step in teacher_trace]
            if len(student_actions) != args.K or len(teacher_actions) != args.K:
                raise RuntimeError(f"K-step contract failed at row {index}")
            row = {
                "component": "sentence_compress",
                "protocol": "Teacher-always-on_vs_Student-always-off",
                "K": args.K,
                "state_id": item.get("state_id", f"sentence_compress_K{args.K}_{index:03d}"),
                "query_id": qid,
                "turn_id": item["turn_id"],
                "snapshot_hash": item["snapshot_hash"],
                "frozen_snapshot_integrity_hash": frozen_hash,
                "reconstructed_initial_state_hash": reconstructed_hash,
                "teacher_view": "full_component_on_all_steps",
                "student_view": "reduced_component_off_all_steps",
                "teacher_actions": teacher_actions,
                "student_actions": student_actions,
                "first_action_disagreement": int(teacher_actions[0] != student_actions[0]),
                "branch_T_metrics": teacher_metrics,
                "branch_S_metrics": student_metrics,
                "tool_cost_delta": teacher_metrics["tool_search_cost"] - student_metrics["tool_search_cost"],
                "utility_delta": teacher_metrics["objective_utility"] - student_metrics["objective_utility"],
                "full_harness_takeover": False,
                "search_backend": backend,
                "branch_T_trace": teacher_trace,
                "branch_S_trace": student_trace,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (index + 1) % 16 == 0:
                print(json.dumps({"K": args.K, "finished": index + 1, "path": str(raw_path)}), flush=True)

    n = len(rows)
    summary = {
        "component": "sentence_compress",
        "protocol": "Teacher-always-on_vs_Student-always-off",
        "horizon_contract": "first_action_counts_in_K",
        "K": args.K,
        "n_states": n,
        "states_manifest": str(args.states_manifest),
        "states_manifest_sha256": sha256(args.states_manifest),
        "first_action_disagreement_rate": sum(row["first_action_disagreement"] for row in rows) / n,
        "tool_cost_delta": sum(row["tool_cost_delta"] for row in rows) / n,
        "utility_delta": sum(row["utility_delta"] for row in rows) / n,
        "full_harness_takeover": any(row["full_harness_takeover"] for row in rows),
        "frozen_snapshot_integrity_match": all(
            row["snapshot_hash"] == row["frozen_snapshot_integrity_hash"] for row in rows
        ),
        "reconstructed_initial_hash_unique_count": len({
            row["reconstructed_initial_state_hash"] for row in rows
        }),
        "teacher_all_steps_full": all(
            step["view"] == "full" for row in rows for step in row["branch_T_trace"]
        ),
        "student_all_steps_reduced": all(
            step["view"] == "reduced" for row in rows for step in row["branch_S_trace"]
        ),
        "exact_K_actions_per_branch": all(
            len(row["branch_T_trace"]) == args.K and len(row["branch_S_trace"]) == args.K
            for row in rows
        ),
        "raw": str(raw_path),
    }
    summary_path = out / f"SUMMARY_K{args.K}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    default_states = Path(
        "/mnt/songzijun/Capability_Evolution/SCAPE/outputs/"
        "0820_sentence_compress_formal_fork_k128_frozen_pool1024/manifests/"
        "sentence_compress_states_n128_seed2214.jsonl"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, choices=[4, 8], required=True)
    parser.add_argument("--states-manifest", type=Path, default=default_states)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    parser.add_argument("--browsecomp-root", type=Path, default=bcp)
    parser.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=HERE.parent / "outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
