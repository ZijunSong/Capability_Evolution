from __future__ import annotations

import json

import pytest

from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow, fingerprint
from trim.training.sft_ablation.shards import merge_shards, write_shard


def _row(row_id: str, query: str) -> SFTBatchRow:
    profile = PROFILES["teacher_sft"]
    return SFTBatchRow(
        row_id=row_id,
        experiment="teacher_sft",
        query_id=query,
        episode_id=query,
        decision_id=row_id,
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[True],
        weight=1.0,
        policy_version="p",
        prompt_owner=profile.prompt_owner,
        target_source=profile.supervision_provider,
    )


def test_node_count_does_not_change_the_merged_set(tmp_path):
    identity = {"experiment": "teacher_sft", "loss": "action_ce_v1"}
    rows = [_row("b", "q2"), _row("a", "q1")]
    one = tmp_path / "one"
    write_shard(one, rows, identity=identity, round_id=0, policy_digest="d", query_ids=["q1", "q2"])
    left, right = tmp_path / "l", tmp_path / "r"
    write_shard(left, [rows[1]], identity=identity, round_id=0, policy_digest="d", query_ids=["q1"])
    write_shard(right, [rows[0]], identity=identity, round_id=0, policy_digest="d", query_ids=["q2"])
    merged_one = merge_shards([one], identity=identity, round_id=0, policy_digest="d", expected_query_ids=["q1", "q2"])
    merged_two = merge_shards([right, left], identity=identity, round_id=0, policy_digest="d", expected_query_ids=["q2", "q1"])
    assert [row.row_id for row in merged_one] == [row.row_id for row in merged_two] == ["a", "b"]


def test_empty_shard_is_complete_and_bad_shards_are_refused(tmp_path):
    identity = {"experiment": "teacher_sft", "loss": "action_ce_v1"}
    empty = tmp_path / "empty"
    write_shard(empty, [], identity=identity, round_id=3, policy_digest="d", query_ids=[])
    assert merge_shards([empty], identity=identity, round_id=3, policy_digest="d", expected_query_ids=[]) == []
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ValueError, match="DONE"):
        merge_shards([missing], identity=identity, round_id=3, policy_digest="d", expected_query_ids=[])
    stale = tmp_path / "stale"
    write_shard(stale, [], identity=identity, round_id=3, policy_digest="old", query_ids=[])
    with pytest.raises(ValueError, match="digest"):
        merge_shards([stale], identity=identity, round_id=3, policy_digest="d", expected_query_ids=[])
    dup = tmp_path / "dup"
    write_shard(dup, [_row("a", "q")], identity=identity, round_id=3, policy_digest="d", query_ids=["q", "q"])
    with pytest.raises(ValueError, match="duplicate query"):
        merge_shards([dup], identity=identity, round_id=3, policy_digest="d", expected_query_ids=["q", "q"])
    tampered = tmp_path / "tamper"
    write_shard(tampered, [_row("a", "q")], identity=identity, round_id=3, policy_digest="d", query_ids=["q"])
    (tampered / "rows.jsonl").write_text(json.dumps({"row_id": "changed"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        merge_shards([tampered], identity=identity, round_id=3, policy_digest="d", expected_query_ids=["q"])
    assert fingerprint({"experiment": "teacher_sft"}) != fingerprint({"experiment": "student_onpolicy_sft"})
