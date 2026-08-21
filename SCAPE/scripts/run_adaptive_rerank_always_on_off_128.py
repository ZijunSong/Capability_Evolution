#!/usr/bin/env python3
"""Adaptive-rerank Teacher-always-on versus Student-always-off fork."""
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_adaptive_rerank_instruction_formal_fork as base
from scape.adapters.components import minus_mask
from scape.common.sha256sums import write_sha256sums
from scape.rendering.dual_view import DualViewRenderer

COMPONENT = "adaptive_rerank_instruction"
SOURCE_COMPONENT = COMPONENT
ARTIFACT_PREFIX = "ADAPTIVE_RERANK_ALWAYS_ON_OFF"
SHARD_PREFIX = "ADAPTIVE_ALWAYS_ON_OFF"
RUNNER_NAME = "adaptive_rerank_always_on_off_128"
SCHEMA_VERSION = "adaptive_rerank_always_on_off_v1"
SEEDS = (2214, 2215, 2216, 2217)
HORIZONS = (4, 8)
SOURCE = REPO / "outputs/0820_adaptive_rerank_instruction_128_cohorts"
OUT_DEFAULT = REPO / "outputs/0821_adaptive_rerank_instruction_always_on_off_128"
MODEL_DEFAULT = Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
BCP_DEFAULT = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_action(action: Mapping[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def masks() -> tuple[dict[str, bool], dict[str, bool]]:
    student = minus_mask(COMPONENT)
    teacher = dict(student)
    teacher[COMPONENT] = True
    if student[COMPONENT] or not teacher[COMPONENT]:
        raise RuntimeError("failed to construct explicit adaptive-rerank masks")
    return student, teacher


def policy_action(
    state: base.LiveState,
    scorer: base.HFContinuationScorer,
    renderer: DualViewRenderer,
    mask: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    snapshot = state.snapshot()
    view = renderer.render_fn(snapshot, mask)
    distribution = scorer.distribution(view, state)
    return distribution["decoded"], distribution, snapshot.content_hash()


def source_rows(seed: int, k: int) -> list[dict[str, Any]]:
    return load_jsonl(SOURCE / f"seed{seed}" / "shards" / f"{SOURCE_COMPONENT}_K{k}.jsonl")


def reconstruct_states(
    *,
    seed: int,
    k: int,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    searcher: Any,
    scorer: base.HFContinuationScorer,
    renderer: DualViewRenderer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = source_rows(seed, k)
    if len(source) != 32:
        raise RuntimeError(f"seed {seed} K{k}: expected 32 source rows, found {len(source)}")
    source_other = source_rows(seed, 8 if k == 4 else 4)
    identity = lambda row: (str(row["query_id"]), int(row["turn_id"]), str(row["snapshot_hash"]))
    if [identity(row) for row in source] != [identity(row) for row in source_other]:
        raise RuntimeError(f"seed {seed}: source K4/K8 ordered frozen states differ")

    manifest = json.loads(
        (SOURCE / f"seed{seed}" / "manifests" / "UTILITY_LIVE256.json").read_text(encoding="utf-8")
    )
    targets = {(str(row["query_id"]), int(row["turn_id"])): row for row in source}
    if len(targets) != len(source):
        raise RuntimeError(f"seed {seed} K{k}: duplicate query/turn target")

    # Reconstruct the frozen source trajectory with the source shard's
    # disabled component. The target component mask is only used after the
    # frozen snapshot has been recovered.
    source_student_mask = minus_mask(SOURCE_COMPONENT)
    found: dict[tuple[str, int], dict[str, Any]] = {}
    source_action_matches = 0
    snapshot_matches = 0
    for qid_value in manifest["query_ids"]:
        qid = str(qid_value)
        state = base.LiveState(
            qid=qid,
            query=queries[qid],
            gold=qrels[qid],
            searcher=searcher,
            component=SOURCE_COMPONENT,
            branch_seed=f"collect:{SOURCE_COMPONENT}:{qid}",
        )
        for _ in range(8):
            key = (qid, int(state.step))
            action_s, _, _ = policy_action(state, scorer, renderer, source_student_mask)
            action_to_execute = action_s
            if key in targets:
                snapshot = state.snapshot()
                source_row = targets[key]
                snapshot_matches += int(snapshot.content_hash() == str(source_row["snapshot_hash"]))
                source_action_matches += int(canonical_action(action_s) == canonical_action(source_row["a_S"]))
                found[key] = {
                    "source": source_row,
                    "snapshot": snapshot.to_dict(),
                    "snapshot_hash": snapshot.content_hash(),
                }
                # Advance with the action frozen in the source shard. This keeps
                # later target states tied to the original cohort trajectory even
                # if the current model/runtime changes greedy decoding slightly.
                action_to_execute = source_row["a_S"]
            state.execute(action_to_execute)
            if len(found) == len(targets):
                break
        if len(found) == len(targets):
            break

    ordered = []
    for row in source:
        key = (str(row["query_id"]), int(row["turn_id"]))
        if key not in found:
            raise RuntimeError(f"seed {seed} K{k}: missing reconstructed state {key}")
        ordered.append(found[key])
    audit = {
        "seed": seed,
        "K": k,
        "source_rows": len(source),
        "reconstructed_rows": len(ordered),
        "source_k4_k8_ordered_identity_match": True,
        "ordered_snapshot_hash_matches": snapshot_matches,
        "source_student_action_matches": source_action_matches,
    }
    if snapshot_matches != len(source):
        raise RuntimeError(f"frozen reconstruction audit failed: {json.dumps(audit, sort_keys=True)}")
    return ordered, audit


def run_branch(
    start: base.LiveState,
    *,
    k: int,
    scorer: base.HFContinuationScorer,
    renderer: DualViewRenderer,
    label: str,
    mask: Mapping[str, bool],
) -> tuple[base.LiveState, list[dict[str, Any]]]:
    state = start.clone(label)
    trace = []
    for index in range(k):
        action, distribution, snapshot_hash = policy_action(state, scorer, renderer, mask)
        state.execute(action)
        trace.append({
            "branch": label,
            "decision_index": index,
            "phase": "first_action" if index == 0 else f"continue_{index}",
            "component_enabled": bool(mask[COMPONENT]),
            "mask": dict(mask),
            "action": action,
            "top_prob": max(distribution["tool_name_probs"].values()),
            "snapshot_hash": snapshot_hash,
            "metrics": state.metrics(),
        })
    return state, trace


def run_cell(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "shards").mkdir(exist_ok=True)
    (args.out_dir / "audits").mkdir(exist_ok=True)
    queries = base._load_queries(args.browsecomp_root / "topics-qrels/queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels/qrel_evidence.txt")
    searcher, search_backend = base.build_searcher(args.index_path, args.corpus_path)
    scorer = base.HFContinuationScorer(
        args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens
    )
    renderer = DualViewRenderer()
    states, reconstruction = reconstruct_states(
        seed=args.seed,
        k=args.K,
        queries=queries,
        qrels=qrels,
        searcher=searcher,
        scorer=scorer,
        renderer=renderer,
    )
    student_mask, teacher_mask = masks()
    output = args.out_dir / "shards" / f"{SHARD_PREFIX}_K{args.K}_SEED{args.seed}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(states):
            source = item["source"]
            qid = str(source["query_id"])
            start = base.state_from_snapshot(item["snapshot"], queries[qid], qrels[qid], searcher, COMPONENT)
            common_initial_hash = start.snapshot().content_hash()
            student, trace_s = run_branch(
                start, k=args.K, scorer=scorer, renderer=renderer, label="S", mask=student_mask
            )
            teacher, trace_t = run_branch(
                start, k=args.K, scorer=scorer, renderer=renderer, label="T", mask=teacher_mask
            )
            metrics_s = student.metrics()
            metrics_t = teacher.metrics()
            action_s = trace_s[0]["action"]
            action_t = trace_t[0]["action"]
            row = {
                "schema_version": SCHEMA_VERSION,
                "component": COMPONENT,
                "protocol": "Teacher-always-on_vs_Student-always-off",
                "contract": "Teacher adaptive_rerank_instruction ON at every one of K decisions; Student OFF at every one of K decisions; first action included in K",
                "seed": args.seed,
                "K": args.K,
                "state_id": source["state_id"],
                "query_id": qid,
                "turn_id": int(source["turn_id"]),
                "snapshot_hash": source["snapshot_hash"],
                "reconstructed_snapshot_hash": item["snapshot_hash"],
                "source_component": SOURCE_COMPONENT,
                "branch_S_initial_state_hash": common_initial_hash,
                "branch_T_initial_state_hash": common_initial_hash,
                "a_S": action_s,
                "a_T": action_t,
                "first_action_disagreement": int(canonical_action(action_t) != canonical_action(action_s)),
                "branch_S_metrics": metrics_s,
                "branch_T_metrics": metrics_t,
                "tool_cost_delta": metrics_t["tool_search_cost"] - metrics_s["tool_search_cost"],
                "utility_delta": metrics_t["objective_utility"] - metrics_s["objective_utility"],
                "branch_S_trace": trace_s,
                "branch_T_trace": trace_t,
                "teacher_component_enabled_all_steps": all(step["component_enabled"] for step in trace_t),
                "student_component_disabled_all_steps": all(not step["component_enabled"] for step in trace_s),
                "forced_action_included_in_K": True,
                "full_harness_takeover": False,
                "search_backend": search_backend,
                "runner": RUNNER_NAME,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 8 == 0:
                print(json.dumps({"seed": args.seed, "K": args.K, "finished": index + 1}), flush=True)

    reconstruction["output"] = str(output)
    reconstruction["output_rows"] = len(states)
    reconstruction["source_snapshot_sha256"] = hashlib.sha256(
        "\n".join(item["snapshot_hash"] for item in states).encode()
    ).hexdigest()
    audit_path = args.out_dir / "audits" / f"RECONSTRUCTION_K{args.K}_SEED{args.seed}.json"
    audit_path.write_text(json.dumps(reconstruction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reconstruction, indent=2, ensure_ascii=False), flush=True)
    return 0


def metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.mean(values),
        "positive": sum(value > 1e-12 for value in values),
        "negative": sum(value < -1e-12 for value in values),
        "zero": sum(abs(value) <= 1e-12 for value in values),
    }


def aggregate(args: argparse.Namespace) -> int:
    all_rows = []
    reconstruction_audits = []
    for k in HORIZONS:
        for seed in SEEDS:
            path = args.out_dir / "shards" / f"{SHARD_PREFIX}_K{k}_SEED{seed}.jsonl"
            rows = load_jsonl(path)
            if len(rows) != 32:
                raise RuntimeError(f"{path}: expected 32 rows, found {len(rows)}")
            all_rows.extend(rows)
            reconstruction_audits.append(json.loads(
                (args.out_dir / "audits" / f"RECONSTRUCTION_K{k}_SEED{seed}.json").read_text(encoding="utf-8")
            ))

    summaries = []
    for k in HORIZONS:
        rows = [row for row in all_rows if int(row["K"]) == k]
        disagreement = metric_stats(rows, "first_action_disagreement")
        cost = metric_stats(rows, "tool_cost_delta")
        utility = metric_stats(rows, "utility_delta")
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
            "initial_branch_hash_mismatch": sum(row["branch_S_initial_state_hash"] != row["branch_T_initial_state_hash"] for row in rows),
            "frozen_reconstruction_mismatch": sum(row["snapshot_hash"] != row["reconstructed_snapshot_hash"] for row in rows),
            "teacher_mask_failure": sum(not row["teacher_component_enabled_all_steps"] for row in rows),
            "student_mask_failure": sum(not row["student_component_disabled_all_steps"] for row in rows),
            "horizon_action_count_failure": sum(len(row["branch_T_trace"]) != k or len(row["branch_S_trace"]) != k for row in rows),
            "full_harness_takeover": sum(bool(row["full_harness_takeover"]) for row in rows),
            "per_seed": {
                str(seed): {
                    "n": sum(int(row["seed"]) == seed for row in rows),
                    "first_action_disagreement_rate": statistics.mean(float(row["first_action_disagreement"]) for row in rows if int(row["seed"]) == seed),
                    "tool_cost_delta_mean": statistics.mean(float(row["tool_cost_delta"]) for row in rows if int(row["seed"]) == seed),
                    "utility_delta_mean": statistics.mean(float(row["utility_delta"]) for row in rows if int(row["seed"]) == seed),
                }
                for seed in SEEDS
            },
        })

    identities = {
        k: {
            seed: [
                (str(row["query_id"]), int(row["turn_id"]), str(row["snapshot_hash"]))
                for row in all_rows if int(row["K"]) == k and int(row["seed"]) == seed
            ]
            for seed in SEEDS
        }
        for k in HORIZONS
    }
    ordered_k4_k8_match = {str(seed): identities[4][seed] == identities[8][seed] for seed in SEEDS}
    audit_failure = any(
        summary["n_paired_states"] != 128
        or summary["initial_branch_hash_mismatch"]
        or summary["frozen_reconstruction_mismatch"]
        or summary["teacher_mask_failure"]
        or summary["student_mask_failure"]
        or summary["horizon_action_count_failure"]
        or summary["full_harness_takeover"]
        for summary in summaries
    ) or not all(ordered_k4_k8_match.values())
    if audit_failure:
        raise RuntimeError("aggregate audit failed")

    per_state = args.out_dir / f"{ARTIFACT_PREFIX}_PER_STATE.jsonl"
    with per_state.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "status": "completed_formal",
        "component": COMPONENT,
        "protocol": "Teacher-always-on_vs_Student-always-off",
        "contract": "Explicit ON/OFF rendering masks at every decision; exactly K actions including the first action",
        "seeds": list(SEEDS),
        "states_per_seed": 32,
        "paired_states_per_K": 128,
        "horizons": list(HORIZONS),
        "model": str(args.model),
        "source": str(SOURCE),
        "summaries": summaries,
        "audit": {
            "ordered_k4_k8_match": ordered_k4_k8_match,
            "reconstruction_audits": reconstruction_audits,
            "full_harness_takeover": False,
        },
    }
    summary_path = args.out_dir / f"{ARTIFACT_PREFIX}_SUMMARY.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {COMPONENT} always-on/off gain",
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
    (args.out_dir / f"{ARTIFACT_PREFIX}_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    args = parser.parse_args()
    os.environ.setdefault("JAVA_HOME", "/opt/scape-jdk21")
    if args.mode == "cell":
        if args.seed is None or args.K is None:
            parser.error("--seed and --K are required in cell mode")
        return run_cell(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
