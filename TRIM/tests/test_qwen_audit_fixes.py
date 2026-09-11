"""Regression tests for Qwen bcplus_test_50 audit fixes (2026-09-11)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "external" / "harness-1"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from harness.tools import SearchCorpusToolCallMetadata  # noqa: E402
from harness.ultra_core import WorkingMemory, parse_doc_ids_from_observation  # noqa: E402
from trim.eval.tool_health import merge_tool_health  # noqa: E402
from trim.local_backend.auxiliary import LocalVerifierClient  # noqa: E402
from trim.local_backend.format_obs import format_search_observation  # noqa: E402
from trim.upstream_harness1.api_adapter import parse_chat_completion  # noqa: E402
from trim.upstream_harness1.env_bridge import clip_observation_text  # noqa: E402


def test_review_docs_accepts_chunk_id_and_doc_id():
    wm = WorkingMemory(query="audit test")
    wm.doc_store["80896"] = {"full_text": "full body", "snippet": "full body"}
    wm.pool_id_set.add("80896")
    wm.pool_ids.append("80896")

    by_doc = wm.review_docs(["80896"])
    by_chunk = wm.review_docs(["80896_0"])

    assert "full body" in by_doc
    assert "full body" in by_chunk
    assert "(not found in memory)" not in by_chunk


def test_curate_importance_only_retag():
    wm = WorkingMemory(query="audit test")
    wm.curated_ids = ["6316"]
    wm.curated_importance = {"6316": "high"}
    wm.pool_id_set.add("6316")

    # Enable importance tagging for this test instance path via module flags.
    import harness.ultra_core as uc

    old = uc.V8D_IMPORTANCE_TAGGING
    uc.V8D_IMPORTANCE_TAGGING = True
    try:
        result = wm.curate(
            add_ids=[],
            remove_ids=[],
            importance={"6316_0": "very_high"},
        )
    finally:
        uc.V8D_IMPORTANCE_TAGGING = old

    assert wm.curated_importance["6316"] == "very_high"
    assert "6316[very_high]" in result


def test_parse_hermes_tool_call_wrapper_in_content():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '<tool_call>\n{"name": "review_docs", '
                            '"arguments": {"doc_ids": ["74795_0"]}}\n'
                        )
                    },
                }
            ]
        }
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_verifier_retries_on_length_with_empty_content():
    client = LocalVerifierClient(
        base_url="http://127.0.0.1:8050/v1",
        model="harness-1-verifier",
        require_loopback=False,
    )
    calls: list[int | None] = []

    def fake_complete(messages, tools, max_tokens, timeout_s=None, temperature=None):
        calls.append(max_tokens)
        if len(calls) == 1:
            return {
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {},
            }
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
            "usage": {},
        }

    client._http = MagicMock()
    client._http.complete = fake_complete

    out = client.chat(messages=[{"role": "user", "content": "claim?"}], max_tokens=512)
    assert calls == [512, 1024]
    assert out.choices[0].message.content == "yes"
    assert client.capability_log.get("verify_length_retries") == 1


def test_search_observation_keeps_multiple_document_ids_within_budget():
    ids = [f"doc{i}_0" for i in range(10)]
    docs = [f"TITLE {i}\n" + ("x" * 5000) for i in range(10)]
    text = format_search_observation(ids, docs, display_limit=10, total_char_budget=12000)
    visible_ids = parse_doc_ids_from_observation(text)
    assert len(visible_ids) >= 5
    assert all(f"doc{i}_0" in visible_ids for i in range(min(5, len(visible_ids))))
    assert "display_limit=10" not in text


def test_search_metadata_carries_full_doc_texts_separate_from_display():
    meta = SearchCorpusToolCallMetadata(
        returned_chunk_ids=["71608_0"],
        doc_texts={"71608_0": "FULL BODY " * 10000},
    )
    display = format_search_observation(["71608_0"], ["SHORT DISPLAY"])
    assert len(display) < len(meta.doc_texts["71608_0"])
    assert meta.doc_texts["71608_0"].startswith("FULL BODY")


def test_add_to_pool_dedup_still_stores_reviewable_text():
    wm = WorkingMemory(query="dedup test")
    import harness.ultra_core as uc

    old = uc.V8D_CONTENT_DEDUP
    uc.V8D_CONTENT_DEDUP = True
    try:
        wm.content_dedup = uc.ContentDedupTracker()
        first = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
        second = first + " tiny variation"
        wm.add_to_pool(["100_0"], {"100_0": first})
        wm.add_to_pool(["65829_0"], {"65829_0": second})
        assert "65829" in wm.doc_store or "65829_0" in wm.doc_store
        review = wm.review_docs(["65829_0"])
        assert "(not found in memory)" not in review
    finally:
        uc.V8D_CONTENT_DEDUP = old


def test_review_docs_reports_omitted_by_limit():
    wm = WorkingMemory(query="review limit")
    for i in range(6):
        wm.doc_store[f"d{i}"] = {"full_text": f"body {i}", "snippet": f"body {i}"}
    out = wm.review_docs([f"d{i}" for i in range(6)])
    assert "omitted_by_limit=1" in out
    assert out.count("# DOCUMENT ID:") == 5


def test_clip_observation_preserves_document_boundaries():
    blocks = []
    for i in range(8):
        blocks.append(f"\n# DOCUMENT ID: doc{i}_0\n" + ("word " * 400))
    text = "".join(blocks)
    clipped = clip_observation_text(text, 6000)
    visible = parse_doc_ids_from_observation(clipped)
    assert len(visible) >= 2
    assert "more document(s) not shown" in clipped


def test_merge_tool_health_sums_verify_length_retries(tmp_path):
    def write_shard(name: str, retries: int) -> Path:
        path = tmp_path / name
        path.write_text(
            '{"counters": {"verify_length_retries": '
            + str(retries)
            + '}, "capability_log": {"verify_length_retries": '
            + str(retries)
            + "}}\n",
            encoding="utf-8",
        )
        return path

    merged = merge_tool_health([write_shard("a.json", 1), write_shard("b.json", 3)])
    assert merged["counters"]["verify_length_retries"] == 4
    assert merged["capability_log"]["verify_length_retries"] == 4
