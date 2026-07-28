"""Tests for bare rollout (no Harness)."""

from __future__ import annotations

import json
from pathlib import Path

from training.opd._policy_backend import MockRolloutBackend
from training.opd.bare_rollout import (
    bare_messages,
    load_completed_query_ids,
    run_bare_rollout,
)
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.rollout_worker import QueryRecord


def test_bare_messages_is_query_only():
    msgs = bare_messages("Who won the Nobel Prize?")
    assert msgs == [{"role": "user", "content": "Who won the Nobel Prize?"}]


def test_run_bare_rollout_mock(tmp_path: Path):
    records = [QueryRecord(query_id="q1", query="Test question?")]
    out = tmp_path / "bare.jsonl"
    trajectories = run_bare_rollout(
        MockRolloutBackend(), records, output_jsonl=out, resume=False
    )
    assert len(trajectories) == 1
    assert trajectories[0].mode == "bare"
    assert load_completed_query_ids(out) == {"q1"}


def test_load_browsecomp_from_fixture_jsonl(tmp_path: Path):
    path = tmp_path / "browsecomp_plus_decrypted.jsonl"
    rows = [
        {"query_id": "1", "query": "Q1"},
        {"query_id": "2", "query": "Q2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records = load_browsecomp_full_queries(
        split="all", download_if_missing=False, answers_path=path
    )
    assert len(records) == 2
