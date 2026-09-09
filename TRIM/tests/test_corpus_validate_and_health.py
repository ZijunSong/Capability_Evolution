"""Corpus/index alignment and tool-health aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.eval.tool_health import build_tool_health_payload, merge_tool_health, write_tool_health
from trim.local_backend.auxiliary import LocalVerifierClient, OpenAIChatShim, _extract_verifier_text
from trim.local_backend.corpus_store import LocalCorpusStore
from trim.local_backend.corpus_validate import validate_corpus_against_index


def test_validate_corpus_against_index_requires_full_coverage():
    store = LocalCorpusStore.from_memory([{"id": "59931", "text": "alpha"}, {"id": "69324", "text": "beta"}])
    with pytest.raises(RuntimeError, match="corpus/index mismatch"):
        validate_corpus_against_index(store=store, index_num_docs=100195)


def test_validate_corpus_against_index_accepts_matching_store():
    docs = [{"id": str(i), "text": f"doc {i}"} for i in range(3)]
    docs.extend([{"id": "59931", "text": "probe"}, {"id": "69324", "text": "probe"}, {"id": "44797", "text": "probe"}])
    store = LocalCorpusStore.from_memory(docs)
    report = validate_corpus_against_index(store=store, index_num_docs=len(store.documents))
    assert report["ok"] is True


def test_extract_verifier_text_from_reasoning_tail():
    text = _extract_verifier_text(None, "Long reasoning...\nno. document does not support the claim.")
    assert text is not None
    assert text.lower().startswith("no")


def test_tool_health_merge_sums_counters(tmp_path: Path):
    a = build_tool_health_payload({"read_success": 2, "grep_success": 1}, worker_rank=0)
    b = build_tool_health_payload({"read_success": 3, "grep_no_results": 4}, worker_rank=1)
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    write_tool_health(pa, a)
    write_tool_health(pb, b)
    merged = merge_tool_health([pa, pb])
    assert merged["counters"]["read_success"] == 5
    assert merged["counters"]["grep_success"] == 1
    assert merged["counters"]["grep_no_results"] == 4


def test_local_verifier_records_empty_content(tmp_path: Path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = {"choices": [{"message": {"content": None, "reasoning_content": ""}, "finish_reason": "length"}]}
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        log: dict = {}
        client = LocalVerifierClient(base_url=url, model="v", require_loopback=True, capability_log=log)
        shim = OpenAIChatShim(client)
        with pytest.raises(RuntimeError, match="empty final content"):
            shim.create(
                model="v",
                messages=[{"role": "user", "content": "CLAIM: x\n\nDOCUMENT:\ny"}],
                max_tokens=80,
            )
        assert log["verify_requests"] == 1
        assert log["verify_empty_content"] == 1
    finally:
        server.shutdown()
