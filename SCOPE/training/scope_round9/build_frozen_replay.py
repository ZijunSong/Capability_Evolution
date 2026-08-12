#!/usr/bin/env python3
"""Build frozen replay datasets for Round 9 with reconstructed live candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers import AutoTokenizer

from training.scope.rollback_effective_input import build_rollback_effective_input

OUT_DATA = _REPO / "artifacts/datasets/scope_round9/frozen_replay"
R8_VALID = _REPO / "artifacts/datasets/scope_round8/rollback_sdi/valid.jsonl"
R8_COLLECTION = _REPO / "outputs/scope_round8/rollback_collection"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
R8_PHASE3 = _REPO / "outputs/scope_round8/phase3_closed_loop"

WAVE_A_VARIANTS = [
    "base_agent_core",
    "rollback_o7_seed42",
    "rollback_o7_seed43",
    "rollback_o7_seed44",
    "rollback_prompt_hint_distill",
    "rollback_trajectory_imitation",
    "rollback_correct_only",
    "rollback_soft_replan_only",
]

_CKPT_TURN_RE = re.compile(r"^ckpt_(\d+)_")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_collection_index() -> dict[tuple[str, int], dict]:
    index: dict[tuple[str, int], dict] = {}
    for mode in ("natural", "injected"):
        root = R8_COLLECTION / mode
        if not root.exists():
            continue
        for shard in sorted(root.iterdir()):
            if not shard.is_dir():
                continue
            for row in load_jsonl(shard / "rollback_events.jsonl"):
                key = (str(row.get("query_id", "")), int(row.get("turn_id", 0)))
                if key[0] and key not in index:
                    index[key] = row
    return index


def _parse_ckpt_turn(cid: str | None) -> int:
    if not cid:
        return -1
    m = _CKPT_TURN_RE.match(str(cid))
    return int(m.group(1)) if m else -1


def _load_phase3_events(variant: str) -> list[dict]:
    agg = R8_PHASE3 / variant / "_agg" / "rollback_events.jsonl"
    if agg.exists():
        return load_jsonl(agg)
    events: list[dict] = []
    for i in range(4):
        events.extend(load_jsonl(R8_PHASE3 / variant / f"shard{i}" / "rollback_events.jsonl"))
    return events


def reconstruct_candidates_for_query(events: list[dict]) -> dict[int, list[dict]]:
    """Rebuild candidate checkpoints per turn from episode telemetry.

    Round 8 Phase 3 events omit available_checkpoints. Shadow gold IDs frequently
    share turn_id with the decision turn, so candidates include turn_id <= t.
    """
    by_turn_id: dict[int, str] = {}
    for e in events:
        for key in ("shadow_checkpoint_id", "predicted_checkpoint_id"):
            cid = e.get(key)
            tt = _parse_ckpt_turn(cid)
            if tt >= 0 and tt not in by_turn_id:
                by_turn_id[tt] = str(cid)

    per_turn: dict[int, list[dict]] = {}
    for e in events:
        t = int(e.get("turn_id", 0))
        gold = e.get("shadow_checkpoint_id")
        gt = _parse_ckpt_turn(gold)
        max_turn = t  # include same-turn checkpoints created at turn start
        cands: list[dict] = []
        seen: set[str] = set()
        for tt in range(max_turn + 1):
            cid = by_turn_id.get(tt)
            if cid is None and gt == tt and gold:
                cid = str(gold)
            if cid is None:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            cands.append(
                {
                    "checkpoint_id": cid,
                    "turn_id": tt,
                    "n_curated": 0,
                    "n_pool": 0,
                    "n_verified": 0,
                    "remaining_recovery_budget": "?",
                }
            )
        if gold and str(gold) not in seen and gt >= 0 and gt <= t:
            cands.append(
                {
                    "checkpoint_id": str(gold),
                    "turn_id": gt,
                    "n_curated": 0,
                    "n_pool": 0,
                    "n_verified": 0,
                    "remaining_recovery_budget": "?",
                }
            )
        # Guarantee a non-empty candidate set so ROLLBACK predictions never
        # become invalid solely due to missing Phase-3 checkpoint telemetry.
        if not cands:
            qid = str(events[0].get("query_id", "unknown"))
            cands.append(
                {
                    "checkpoint_id": f"ckpt_init_{qid}",
                    "turn_id": 0,
                    "n_curated": 0,
                    "n_pool": 0,
                    "n_verified": 0,
                    "remaining_recovery_budget": "?",
                }
            )
        per_turn[t] = cands
    return per_turn


def extract_live_events(variant: str, collection_index: dict[tuple[str, int], dict]) -> list[dict]:
    events = _load_phase3_events(variant)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_q[str(ev.get("query_id", ""))].append(ev)

    samples: list[dict] = []
    for qid, q_events in by_q.items():
        q_events = sorted(q_events, key=lambda e: int(e.get("turn_id", 0)))
        cand_by_turn = reconstruct_candidates_for_query(q_events)
        for ev in q_events:
            turn = int(ev.get("turn_id", 0))
            key = (qid, turn)
            base = collection_index.get(key, {})
            ds = dict(base.get("decision_state") or {})
            # Never reuse collection checkpoint IDs (different rollout UUIDs).
            ds["available_checkpoints"] = cand_by_turn.get(turn, [])
            ds["turn_id"] = turn
            ds["task_id"] = ds.get("task_id") or qid
            if not ds.get("rendered_context"):
                ds["rendered_context"] = (
                    f"query_id={qid} turn={turn} "
                    f"shadow_op={ev.get('shadow_operation')} "
                    f"student_op={ev.get('student_operation')}"
                )
            op = ev.get("shadow_operation") or "CONTINUE"
            ck = ev.get("shadow_checkpoint_id")
            samples.append(
                {
                    "query_id": qid,
                    "event_id": f"live:{variant}:{qid}:{turn}",
                    "decision_state": ds,
                    "student_state_text": base.get("student_state_text")
                    or ds.get("rendered_context")
                    or json.dumps(ds, ensure_ascii=False),
                    "target_action": {"operation": op, "checkpoint_id": ck},
                    "operation": op,
                    "source_mode": "phase3_live",
                }
            )
    return samples


def serialize_rows(
    samples: list[dict],
    tokenizer,
    *,
    state_source: str,
    hint: str = "",
) -> list[dict]:
    rows = []
    for idx, sample in enumerate(samples):
        rec = build_rollback_effective_input(
            sample, tokenizer, state_source=state_source, hint=hint
        )
        d = rec.to_dict()
        d["event_id"] = sample.get("event_id") or f"{state_source}:{d['query_id']}:{d['turn']}:{idx}"
        rows.append(d)
    return rows


def build_report(datasets: dict[str, list[dict]]) -> dict:
    report: dict = {"datasets": {}}
    for name, rows in datasets.items():
        ops = Counter(r["gold_operation"] for r in rows)
        cand_counts = Counter(len(r.get("candidate_list") or []) for r in rows)
        rb = [r for r in rows if r.get("gold_operation") == "ROLLBACK_TO"]
        coverage = (
            sum(1 for r in rb if r.get("gold_in_candidates")) / max(len(rb), 1) if rb else 1.0
        )
        trunc = sum(1 for r in rows if r.get("truncated")) / max(len(rows), 1)
        lengths = [r.get("token_length_after", 0) for r in rows]
        prompt_hashes = [r.get("prompt_sha256") for r in rows]
        hash_collisions = len(prompt_hashes) - len(set(prompt_hashes))
        label_by_hash: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            label_by_hash[r.get("prompt_sha256", "")].add(r.get("gold_operation", ""))
        label_collisions = sum(1 for labels in label_by_hash.values() if len(labels) > 1)
        report["datasets"][name] = {
            "n_rows": len(rows),
            "operation_distribution": dict(ops),
            "candidate_count_distribution": {str(k): v for k, v in sorted(cand_counts.items())},
            "gold_candidate_coverage": coverage,
            "rollback_n": len(rb),
            "truncation_rate": trunc,
            "mean_token_length": sum(lengths) / max(len(lengths), 1),
            "prompt_hash_collisions": hash_collisions,
            "effective_input_label_collisions": label_collisions,
        }
    return report


def write_report_md(report: dict, path: Path) -> None:
    lines = ["# Frozen replay dataset report", ""]
    for name, stats in report.get("datasets", {}).items():
        lines.append(f"## {name}")
        for key, val in stats.items():
            lines.append(f"- {key}: {val}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DATA)
    p.add_argument("--model-path", default=BASE_MODEL)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    datasets: dict[str, list[dict]] = {}
    collection_index = build_collection_index()

    offline = load_jsonl(R8_VALID)
    for i, row in enumerate(offline):
        row.setdefault("event_id", f"offline_valid:{row.get('query_id')}:{i}")
    datasets["offline_valid"] = serialize_rows(
        offline, tokenizer, state_source="offline_valid"
    )
    write_jsonl(args.output_dir / "offline_valid.jsonl", datasets["offline_valid"])

    base_live_samples = extract_live_events("base_agent_core", collection_index)
    datasets["base_live"] = serialize_rows(
        base_live_samples, tokenizer, state_source="base_live"
    )
    write_jsonl(args.output_dir / "base_live.jsonl", datasets["base_live"])

    self_dir = args.output_dir / "self_live"
    self_dir.mkdir(parents=True, exist_ok=True)
    for variant in WAVE_A_VARIANTS:
        samples = extract_live_events(variant, collection_index)
        rows = serialize_rows(samples, tokenizer, state_source="self_live")
        datasets[f"self_live/{variant}"] = rows
        write_jsonl(self_dir / f"{variant}.jsonl", rows)

    report = build_report(datasets)
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_report_md(report, args.output_dir / "dataset_report.md")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
