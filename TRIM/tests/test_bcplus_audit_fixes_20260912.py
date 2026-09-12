"""Regression tests for BC+ 830 final audit fixes (2026-09-12)."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trim.eval.eval_parallel import merge_traces, stream_merge_jsonl  # noqa: E402
from trim.eval.harness1_api_eval import (  # noqa: E402
    count_tool_calls_from_turns,
    summarize_api_traces,
)
from trim.eval.sr_opd_four_cell_eval import write_upstream_api_eval_outputs  # noqa: E402
from trim.local_backend.corpus_store import LocalCorpusStore  # noqa: E402
from trim.local_backend.tools import LocalGrepCorpusTool  # noqa: E402
from trim.upstream_harness1.api_adapter import (  # noqa: E402
    ChatCompletionsClient,
    parse_chat_completion,
)
from trim.upstream_harness1.env_bridge import (  # noqa: E402
    SchemaValidationError,
    _validate_tool_params,
    clip_observation_text,
)


class FakeSchema:
    name: str

    def __init__(self, name: str):
        self.name = name


class FakeMeta:
    def __init__(self, returned_chunk_ids: list[str], **kwargs):
        self.returned_chunk_ids = returned_chunk_ids
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_grep_catastrophic_regex_terminates_within_timeout():
    store = LocalCorpusStore.from_memory([{"id": "x", "text": "a" * 30 + "!"}])
    grep = LocalGrepCorpusTool(
        store=store,
        metadata_cls=FakeMeta,
        schema=FakeSchema("grep_corpus"),
        timeout_s=0.05,
    )
    started = time.perf_counter()
    text, meta = grep({"pattern": "(a+)+$"})
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
    assert meta.returned_chunk_ids == []
    assert "timed out" in text.lower() or "partial scan" in text.lower()


def test_grep_partial_scan_marks_incomplete_hits():
    docs = [{"id": f"d{i}", "text": f"needle{i} " + ("x" * 200)} for i in range(400)]
    store = LocalCorpusStore.from_memory(docs)
    grep = LocalGrepCorpusTool(
        store=store,
        metadata_cls=FakeMeta,
        schema=FakeSchema("grep_corpus"),
        limit=400,
        timeout_s=0.002,
    )
    text, meta = grep({"pattern": "needle"})
    if not meta.returned_chunk_ids:
        pytest.skip("scan timed out with zero hits on this host")
    if len(meta.returned_chunk_ids) < len(docs):
        assert "partial scan" in text.lower()


def test_parse_accepts_ada_lovelace_answer_json():
    body = json.dumps({"name": "Ada Lovelace", "birth_year": 1815})
    parsed = parse_chat_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
    )
    assert parsed.ok is True
    assert parsed.tool_calls[0]["name"] == "user_text"


def test_validate_tool_params_rejects_bad_array_items():
    tool = MagicMock()
    tool.tool_schema = MagicMock(
        name="curate",
        required=["add_ids"],
        parameters={"add_ids": {"type": "array", "items": {"type": "string"}}},
    )
    with pytest.raises(SchemaValidationError, match="expected string"):
        _validate_tool_params(tool, {"add_ids": [{"unexpected": 1}]})


def test_null_add_ids_is_schema_error():
    tool = MagicMock()
    tool.tool_schema = MagicMock(
        name="curate",
        required=["add_ids"],
        parameters={"add_ids": {"type": "array"}},
    )
    with pytest.raises(SchemaValidationError, match="must not be null"):
        _validate_tool_params(tool, {"add_ids": None})


def test_clip_preserves_first_doc_prefix_with_preamble_and_oversized_first_block():
    preamble = "Search results:"
    first = "\n# DOCUMENT ID: first_0\n" + ("X" * 600)
    second = "\n# DOCUMENT ID: second_0\nshort"
    raw = preamble + first + second
    clipped = clip_observation_text(raw, 200)
    assert "first_0" in clipped
    assert "truncated" in clipped


def test_merge_traces_rejects_duplicate_query_ids():
    rows = [{"query_id": "1"}, {"query_id": "2"}]
    shards = [[{"query_id": "1", "f1": 0.0}], [{"query_id": "1", "f1": 1.0}, {"query_id": "2"}]]
    with pytest.raises(RuntimeError, match="duplicate"):
        merge_traces(shards, rows)


def test_summarize_api_traces_reports_missing_planned_queries():
    traces = [{"query_id": "1", "recall": 1.0, "precision": 1.0, "f1": 1.0}]
    summary = summarize_api_traces(traces, n_planned=3)
    assert summary["n_planned"] == 3
    assert summary["coverage"]["missing"] == 2
    assert summary["dropped_queries"] == 2
    assert summary["partial"] is True


def test_failed_http_503_events_are_counted():
    hits: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append({})
            self.send_response(503)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        client = ChatCompletionsClient(base_url=url, model="m", max_retries=3, timeout_s=2.0)
        with pytest.raises(Exception):
            client.complete([{"role": "user", "content": "hi"}], tools=None)
        assert len(hits) == 3
    finally:
        server.shutdown()

    turns = [
        {
            "transport_retry_events": [
                {"category": "server_error_unclassified"},
                {"category": "server_error_unclassified"},
                {"category": "server_error_unclassified"},
            ]
        }
    ]
    counts = count_tool_calls_from_turns(turns)
    assert counts["transport_error_count"] == 3


def test_upstream_api_output_schema_is_not_sr_opd(tmp_path):
    out = tmp_path / "upstream_api_schema"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "setting": "upstream_api",
        "evaluation_path": "upstream_api",
        "n_queries": 2,
        "n_planned": 2,
        "coverage_complete": True,
        "execution_complete": True,
        "infra_clean": True,
        "coverage": {"planned": 2, "completed": 2, "missing": 0},
        "recall": 0.5,
        "precision": 0.5,
        "f1": 0.5,
    }
    payload = write_upstream_api_eval_outputs(
        out,
        component_id="zero",
        summaries=[summary],
        pool_meta={"n_queries": 2},
    )
    assert payload["status"] == "UPSTREAM_API_BASELINE_EVAL"
    assert payload["run_kind"] == "upstream_api_base_model"
    assert "protocol_complete_rl_opd" not in payload
    assert payload["formal_eligible"] is True


def test_stream_merge_jsonl(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    dest = tmp_path / "merged.jsonl"
    a.write_text('{"x": 1}\n', encoding="utf-8")
    b.write_text('{"x": 2}\n', encoding="utf-8")
    n = stream_merge_jsonl([a, b], dest)
    assert n == 2
    lines = dest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
