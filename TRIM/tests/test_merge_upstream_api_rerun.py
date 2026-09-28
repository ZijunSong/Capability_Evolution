from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MERGE = Path(__file__).resolve().parents[1] / "scripts" / "merge_upstream_api_rerun.py"
_SPEC = importlib.util.spec_from_file_location("merge_upstream_api_rerun", _MERGE)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
merge_rerun = _MOD.merge_rerun


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_merge_rerun_replaces_infra_rows_and_marks_formal(tmp_path: Path):
    original = tmp_path / "original"
    rerun = tmp_path / "rerun"
    api = original / "upstream_api"
    shard = api / "shards" / "rank0"
    _write_jsonl(
        api / "PER_QUERY.jsonl",
        [
            {
                "query_id": "keep-me",
                "finish_reason": "explicit_end_search",
                "recall": 1.0,
                "precision": 1.0,
                "f1": 1.0,
                "trajectory_recall": 1.0,
                "final_answer_recall": 1.0,
            },
            {
                "query_id": "bad-1",
                "finish_reason": "infra_error",
                "infra_error": "TypeError('ellipsis')",
                "format_error": 0.0,
            },
        ],
    )
    _write_jsonl(
        api / "TURNS.jsonl",
        [
            {"query_id": "keep-me", "turn": 0},
            {"query_id": "bad-1", "turn": 0, "pending_failed_attempt": {"x": 1}},
        ],
    )
    _write_jsonl(
        shard / "PER_QUERY.jsonl",
        [
            {"query_id": "bad-1", "finish_reason": "infra_error", "infra_error": "TypeError('ellipsis')"},
        ],
    )
    (shard / "DONE.json").write_text(
        json.dumps({"ok": False, "n_infra_error": 1, "partial": True, "infra_clean": False}) + "\n",
        encoding="utf-8",
    )
    (api / "SUMMARY.json").write_text(
        json.dumps({"eval_profile": "upstream_core_local_bm25", "eval_replicas": 20, "eval_mode": "harness"})
        + "\n",
        encoding="utf-8",
    )
    (original / "FOUR_CELL_OFFICIAL_SUMMARY.json").write_text(
        json.dumps(
            {
                "component": "all",
                "pool": {"pool_contract": "transfer_hotpotqa", "query_count": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        rerun / "upstream_api" / "PER_QUERY.jsonl",
        [
            {
                "query_id": "bad-1",
                "finish_reason": "explicit_end_search",
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "trajectory_recall": 1.0,
                "final_answer_recall": 0.0,
                "format_error": 0.0,
            }
        ],
    )
    _write_jsonl(
        rerun / "upstream_api" / "TURNS.jsonl",
        [{"query_id": "bad-1", "turn": 0, "stage": "attempt_result"}],
    )

    report = merge_rerun(original, rerun, ["bad-1"])
    assert report["formal_eligible"] is True
    assert report["claim_status"] == "FORMAL_ELIGIBLE"
    merged = [json.loads(line) for line in (api / "PER_QUERY.jsonl").read_text().splitlines() if line.strip()]
    assert [row["query_id"] for row in merged] == ["keep-me", "bad-1"]
    assert merged[1]["finish_reason"] == "explicit_end_search"
    assert "infra_error" not in merged[1]
    turns = [json.loads(line) for line in (api / "TURNS.jsonl").read_text().splitlines() if line.strip()]
    assert [row["query_id"] for row in turns] == ["keep-me", "bad-1"]
    official = json.loads((original / "FOUR_CELL_OFFICIAL_SUMMARY.json").read_text())
    assert official["formal_eligible"] is True
    shard_done = json.loads((shard / "DONE.json").read_text())
    assert shard_done["n_infra_error"] == 0
    assert (api / "_pre_ellipsis_rerun" / "PER_QUERY.jsonl").is_file()


def test_merge_rerun_rejects_lingering_infra(tmp_path: Path):
    original = tmp_path / "original"
    rerun = tmp_path / "rerun"
    api = original / "upstream_api"
    _write_jsonl(
        api / "PER_QUERY.jsonl",
        [{"query_id": "bad-1", "finish_reason": "infra_error"}],
    )
    _write_jsonl(
        rerun / "upstream_api" / "PER_QUERY.jsonl",
        [{"query_id": "bad-1", "finish_reason": "infra_error", "infra_error": "still broken"}],
    )
    try:
        merge_rerun(original, rerun, ["bad-1"])
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "still infra_error" in str(exc)
    assert raised
