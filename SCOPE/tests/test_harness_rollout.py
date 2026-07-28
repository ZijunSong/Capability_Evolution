"""Tests for harness rollout helpers."""

from __future__ import annotations

import json
from pathlib import Path

from training.opd.harness_rollout import load_completed_query_ids, save_harness_manifest


def test_load_completed_query_ids(tmp_path: Path):
    path = tmp_path / "harness_rollouts.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "recall": 0.1}) + "\n"
        + json.dumps({"query_id": "q2", "recall": 0.0}) + "\n",
        encoding="utf-8",
    )
    assert load_completed_query_ids(path) == {"q1", "q2"}


def test_save_harness_manifest(tmp_path: Path):
    jsonl = tmp_path / "harness_rollouts.jsonl"
    jsonl.write_text(json.dumps({"query_id": "q1"}) + "\n", encoding="utf-8")
    path = save_harness_manifest(tmp_path, manifest={"model_path": "test"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mode"] == "harness"
    assert data["n_episodes"] == 1
    assert data["model_path"] == "test"
