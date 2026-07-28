"""Tests for BrowseComp rollout query resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.opd.rollout_worker import (
    QueryRecord,
    RolloutConfig,
    build_mock_transitions_from_queries,
    load_query_records_from_json,
    resolve_query_ids,
)


def test_resolve_query_ids_train_split():
    fake_dataset = MagicMock()
    fake_dataset.get_all_query_ids.return_value = ["q1", "q2", "q3", "q4", "q5"]

    with patch("training.opd.rollout_worker.get_dataset", return_value=fake_dataset):
        ids = resolve_query_ids(
            RolloutConfig(dataset="browsecompplus", split="train", limit=2, seed=7)
        )
    assert len(ids) == 2
    assert all(qid in {"q1", "q2", "q3", "q4", "q5"} for qid in ids)


def test_resolve_explicit_query_ids_filters_unknown():
    fake_dataset = MagicMock()
    fake_dataset.get_test_query_ids.return_value = ["a", "b", "c"]

    with patch("training.opd.rollout_worker.get_dataset", return_value=fake_dataset):
        ids = resolve_query_ids(
            RolloutConfig(
                dataset="browsecompplus",
                split="test",
                limit=10,
                query_ids=["a", "missing"],
            )
        )
    assert ids == ["a"]


def test_load_query_records_from_json_fixture():
    path = Path(__file__).parent / "fixtures" / "browsecomp_sample_queries.json"
    records = load_query_records_from_json(path)
    assert len(records) == 3
    assert records[0].query_id == "browsecomp_sample_1"


def test_mock_transitions_from_query_records():
    records = [
        QueryRecord(query_id="q1", query="Who discovered penicillin?"),
    ]
    transitions = build_mock_transitions_from_queries(records)
    assert len(transitions) == 1
    assert transitions[0].query_id == "q1"
    assert transitions[0].metadata["mode"] == "mock_browsecomp"
