#!/usr/bin/env python3
"""Barrier3: build fresh on-policy operation SDI (natural prior, no upsample)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from training.scope_round11.stage1_views import build_stage1_view

_REPO = Path(__file__).resolve().parents[2]
RAW = _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"
OUT = _REPO / "artifacts/datasets/scope_round13/operation_sdi"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
MAX_LEN = 1536


def load_raw(split: str) -> list[dict]:
    rows: list[dict] = []
    root = RAW / split
    for p in sorted(root.glob("*/rollback_events.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_row(r: dict, tokenizer=None) -> dict:
    gold = str(r.get("gold_operation") or (r.get("target_action") or {}).get("operation") or "CONTINUE")
    ck = r.get("gold_checkpoint_id") or (r.get("target_action") or {}).get("checkpoint_id")
    out = {
        "event_id": r.get("event_id"),
        "query_id": r.get("query_id"),
        "capability_id": "rollback",
        "decision_state": r.get("decision_state") or {},
        "student_state_text": r.get("student_state_text") or "",
        "gold_operation": gold,
        "operation": gold,
        "gold_checkpoint_global_id": ck,
        "gold_checkpoint_id": ck,
        "target_action": {"operation": gold, "checkpoint_id": ck},
        "student_operation": r.get("student_operation"),
        "student_margin": r.get("student_margin"),
        "student_scores": r.get("student_scores"),
        "A0_prompt_hash": r.get("A0_prompt_hash"),
        "candidate_ids": r.get("candidate_ids"),
        "student_visible_features": r.get("student_visible_features"),
        "route": r.get("shadow_route") or r.get("route"),
        "shadow_reason_code": r.get("shadow_reason_code"),
        "include_candidate_summary": True,
    }
    if tokenizer is not None:
        view = build_stage1_view(out, tokenizer, "A0", max_length=MAX_LEN)
        out["effective_input_text"] = view.effective_input_text
        out["stage1_text"] = view.effective_input_text
        out["prompt_sha256"] = view.prompt_sha256
        out["truncated"] = view.truncated
    return out


def annotate_hard(train: list[dict]) -> float:
    abs_m = [abs(float(r["student_margin"])) for r in train if r.get("student_margin") is not None]
    q25 = float(np.percentile(abs_m, 25)) if abs_m else 0.0
    for r in train:
        m = r.get("student_margin")
        pred = r.get("student_operation")
        gold = r.get("gold_operation")
        hard = False
        if pred is not None and pred != gold:
            hard = True
        if m is not None and abs(float(m)) <= q25:
            hard = True
        r["is_hard_event"] = hard
        r["hard_margin_q25"] = q25
    return q25


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-test", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    train = [normalize_row(r, tok) for r in load_raw("train")]
    valid = [normalize_row(r, tok) for r in load_raw("valid")]
    # Never mix valid into train
    train_q = {r["query_id"] for r in train}
    valid_q = {r["query_id"] for r in valid}
    assert not (train_q & valid_q), "train/valid query overlap"

    q25 = annotate_hard(train)
    for r in valid:
        r["is_hard_event"] = False  # unused in training

    train_path = OUT / "train.jsonl"
    valid_path = OUT / "valid.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with valid_path.open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gate = {
        "n_train": len(train),
        "n_valid": len(valid),
        "train_queries": len(train_q),
        "valid_queries": len(valid_q),
        "train_gold_rollback_prior": sum(1 for r in train if r["gold_operation"] == "ROLLBACK_TO")
        / max(len(train), 1),
        "valid_gold_rollback_prior": sum(1 for r in valid if r["gold_operation"] == "ROLLBACK_TO")
        / max(len(valid), 1),
        "hard_margin_q25": q25,
        "n_hard_train": sum(1 for r in train if r.get("is_hard_event")),
        "no_global_upsample": True,
        "natural_prior": True,
        "has_effective_input_text": True,
        "stage1_view": "A0",
        "max_length": MAX_LEN,
    }

    test_dones = list((RAW / "test").glob("*/DONE")) if (RAW / "test").exists() else []
    # Manifest uses 5 shards; only seal test.jsonl when complete (or forced).
    test_ready = args.with_test or len(test_dones) >= 5
    if test_ready and test_dones:
        test_raw = load_raw("test")
        test = [normalize_row(r, tok) for r in test_raw]
        test_q = {r["query_id"] for r in test}
        assert not (test_q & train_q) and not (test_q & valid_q), "test overlap"
        for r in test:
            r["is_hard_event"] = False
        with (OUT / "test.jsonl").open("w", encoding="utf-8") as f:
            for r in test:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gate["n_test"] = len(test)
        gate["test_queries"] = len(test_q)
    else:
        gate["n_test"] = 0
        gate["test_queries"] = 0
        gate["test_pending_shards"] = 5 - len(test_dones)

    (OUT / "DATASET_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
