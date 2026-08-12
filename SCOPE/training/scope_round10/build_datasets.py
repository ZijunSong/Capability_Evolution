#!/usr/bin/env python3
"""Barrier 4: Build support-aligned binary operation datasets D0-D4."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round10.common import (
    DATA,
    R8_DATA,
    R9_DATA,
    binary_operation,
    dataset_gate,
    enrich_from_frozen,
    file_sha256,
    load_jsonl,
    query_ids,
    write_json,
    write_jsonl,
)
from training.scope_round9.build_hier_sdi_dataset import enrich_sample

DATASET_ROOT = DATA / "binary_datasets"
SPLIT_DIR = DATA / "live_split"


def map_offline(rows: list[dict], *, state_source: str = "offline") -> list[dict]:
    out = []
    for row in rows:
        sample = enrich_sample(row, include_summary=True)
        if not sample:
            continue
        op = binary_operation(sample)
        if op is None:
            continue
        sample["state_source"] = state_source
        sample["gold_operation"] = op
        sample["operation"] = op
        ta = dict(sample.get("target_action") or {})
        ta["operation"] = op
        sample["target_action"] = ta
        out.append(sample)
    return out


def subsample_mixed(live: list[dict], offline: list[dict], rng: random.Random) -> list[dict]:
    n = min(len(live), len(offline))
    if n == 0:
        return live + offline
    live_s = rng.sample(live, n)
    off_s = rng.sample(offline, n)
    return live_s + off_s


def build_mixed_splits(
    live_train: list[dict],
    live_valid: list[dict],
    offline_train: list[dict],
    offline_valid: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    lt_q, lv_q = query_ids(live_train), query_ids(live_valid)
    off_train = [r for r in offline_train if str(r.get("query_id")) not in lv_q]
    off_valid = [r for r in offline_valid if str(r.get("query_id")) not in lt_q]
    return subsample_mixed(live_train, off_train, rng), subsample_mixed(live_valid, off_valid, rng)


def build_hard_continue(live_train: list[dict], replay_path: Path) -> list[dict]:
    """High-confidence rollback mis-predictions on CONTINUE gold."""
    if not replay_path.exists():
        return []
    replay = {r.get("event_id"): r for r in load_jsonl(replay_path)}
    hard = []
    seen = set()
    for row in live_train:
        eid = row.get("event_id")
        if binary_operation(row) != "CONTINUE":
            continue
        rr = replay.get(eid)
        if not rr:
            continue
        if rr.get("pred_operation") == "ROLLBACK_TO" and rr.get("gold_operation") == "CONTINUE":
            if eid not in seen:
                hard.append(row)
                seen.add(eid)
    return hard


def build_hard_rollback(live_train: list[dict], replay_path: Path) -> list[dict]:
    if not replay_path.exists():
        return []
    replay = {r.get("event_id"): r for r in load_jsonl(replay_path)}
    hard = []
    seen = set()
    for row in live_train:
        eid = row.get("event_id")
        if binary_operation(row) != "ROLLBACK_TO":
            continue
        rr = replay.get(eid)
        if not rr:
            continue
        if rr.get("pred_operation") == "CONTINUE" and rr.get("gold_operation") == "ROLLBACK_TO":
            if eid not in seen:
                hard.append(row)
                seen.add(eid)
    return hard


def add_source_token(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        row = dict(r)
        ds = dict(row.get("decision_state") or {})
        src = row.get("state_source", "unknown")
        ds["state_source_token"] = f"[state_source={src}]"
        row["decision_state"] = ds
        out.append(row)
    return out


def write_dataset(name: str, train: list[dict], valid: list[dict], test: list[dict]) -> dict:
    # Ensure query-level disjoint splits
    tq, vq = query_ids(train), query_ids(valid)
    test = [r for r in test if str(r.get("query_id")) not in tq and str(r.get("query_id")) not in vq]
    ddir = DATASET_ROOT / name
    write_jsonl(ddir / "train.jsonl", train)
    write_jsonl(ddir / "valid.jsonl", valid)
    write_jsonl(ddir / "test.jsonl", test)
    gate = dataset_gate(train, valid, test)
    gate["dataset_sha256"] = {
        "train": file_sha256(ddir / "train.jsonl"),
        "valid": file_sha256(ddir / "valid.jsonl"),
    }
    write_json(ddir / "DATASET_GATE.json", gate)
    return gate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    live_train = load_jsonl(SPLIT_DIR / "live_train.jsonl")
    live_valid = load_jsonl(SPLIT_DIR / "live_valid.jsonl")
    live_test = load_jsonl(SPLIT_DIR / "live_test.jsonl")
    offline_train = map_offline(load_jsonl(R8_DATA / "rollback_sdi/train.jsonl"))
    offline_valid = map_offline(load_jsonl(R8_DATA / "rollback_sdi/valid.jsonl"))

    offline_q = query_ids(offline_train) | query_ids(offline_valid)
    live_test_disjoint = [r for r in live_test if str(r.get("query_id")) not in offline_q]
    live_q = query_ids(live_train) | query_ids(live_valid)
    offline_test_disjoint = [r for r in live_test_disjoint if str(r.get("query_id")) not in live_q]

    replay = (
        _REPO / "outputs/scope_round9/wave_a/rollback_o7_seed42/base_live/hf_replay.jsonl"
    )
    hard_c = build_hard_continue(live_train, replay)
    hard_r = build_hard_rollback(live_train, replay)

    gates = {}
    gates["D0_offline_only"] = write_dataset("D0_offline_only", offline_train, offline_valid, live_test_disjoint)
    gates["D1_live_only"] = write_dataset("D1_live_only", live_train, live_valid, live_test)
    d2_train, d2_valid = build_mixed_splits(
        live_train, live_valid, offline_train, offline_valid, rng
    )
    gates["D2_mixed_aligned"] = write_dataset("D2_mixed_aligned", d2_train, d2_valid, live_test_disjoint)
    d3_train = d2_train + [r for r in hard_c if r not in d2_train]
    gates["D3_mixed_hard_continue"] = write_dataset("D3_mixed_hard_continue", d3_train, d2_valid, live_test_disjoint)
    gates["D4_source_token"] = write_dataset(
        "D4_source_token",
        add_source_token(d2_train),
        add_source_token(d2_valid),
        live_test_disjoint,
    )

    write_json(DATASET_ROOT / "ALL_DATASET_GATES.json", gates)
    for name, g in gates.items():
        print(f"{name}: pass={g['gate_pass']} train={g['n_train']} dist={g['class_distribution_train']}")
        if not g["gate_pass"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
