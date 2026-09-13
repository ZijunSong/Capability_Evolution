"""Regression tests for Harness-G 4-cell audit fixes (2026-09-13)."""

from __future__ import annotations

import time

from trim.adapters.harness_profiles import zero_mask_for
from trim.eval.eval_parallel import summarize_merged_traces
from trim.eval.harness1_metrics import EpisodeTiming
from trim.eval.harness_g_env import (
    _entities_from_sentences,
    _init_visible,
    _lookup_sids,
    _sentences_from_store,
    _sync_curated,
    execute_tool,
    new_state,
)
from trim.eval.harness_g_runtime import parse_harness_g_action
from trim.eval.sr_opd_four_cell_eval import legal_rate
from trim.training.parse_rollout_action import parse_generated_action


def _mask():
    return zero_mask_for("Harness-G")


def test_parse_rejects_analysis_json_example():
    mask = _mask()
    text = (
        "Analysis: we could emit to=answer {\"reason\": \"done\"} as an example, "
        "but the actual call is missing."
    )
    action, ok = parse_generated_action(text, None, enc=None, harness_mask=mask, finish_reason="length")
    assert ok is False
    assert action["name"] == "truncated"


def test_parse_rejects_menu_id_embedded_in_prose():
    mask = _mask()
    action_map = {"A0": {"type": "SELECT", "sid": "d1:s0", "name": "select"}}
    text = "Looking at A0 in the menu, I think we should select it."
    action, ok = parse_generated_action(text, None, enc=None, harness_mask=mask, action_map=action_map)
    assert ok is False


def test_parse_accepts_strict_menu_id_only():
    action_map = {"A0": {"type": "SELECT", "sid": "18128:s3", "name": "select"}}
    action, ok = parse_harness_g_action("A0", action_map=action_map)
    assert ok is True
    assert action["arguments"]["sid"] == "18128:s3"


def test_init_uses_numeric_sid_order_and_doc_coverage():
    store = {
        "d1": {"id": "d1", "text": "s0. s1. s2. s3. s4. s5. s6. s7. s8. s9. s10."},
        "d2": {"id": "d2", "text": "Evidence sentence here."},
    }
    st = new_state("query", store, harness_mask=_mask())
    visible = _init_visible(st, searcher=None, search_k=10)
    assert visible[0] == "d1:s0"
    assert any(sid.startswith("d2:") for sid in visible)


def test_lookup_prioritizes_new_retrieval_docs():
    store = {
        "d1": {"id": "d1", "text": " ".join(f"old sentence {i}." for i in range(12))},
        "new": {"id": "new", "text": "Fresh evidence about the entity."},
    }
    st = new_state("query", store, harness_mask=_mask())
    st["entities"] = {
        "e:test": {
            "eid": "e:test",
            "surface": "Test",
            "sids": [f"d1:s{i}" for i in range(6)],
            "synonyms": [],
        }
    }
    visible = _lookup_sids(st, "e:test", new_doc_order=["new"])
    assert any(sid.startswith("new:") for sid in visible)


def test_observed_docids_are_cumulative():
    store = {
        "a": {"id": "a", "text": "Doc A sentence."},
        "b": {"id": "b", "text": "Doc B sentence."},
        "c": {"id": "c", "text": "Doc C sentence."},
    }
    st = new_state("query", store, harness_mask=_mask())
    st["visible_sids"] = ["a:s0"]
    _sync_curated(st)
    assert st["observed_docids"] == ["a"]
    st["visible_sids"] = ["b:s0"]
    _sync_curated(st)
    assert set(st["observed_docids"]) == {"a", "b"}


def test_entities_do_not_merge_by_first_token():
    sentences = _sentences_from_store(
        {
            "d1": {"id": "d1", "text": "John Smith wrote the report."},
            "d2": {"id": "d2", "text": "John Williams reviewed it."},
        }
    )
    entities = _entities_from_sentences(sentences)
    assert entities["e:john_smith"]["synonyms"] == []
    assert entities["e:john_williams"]["synonyms"] == []


def test_failed_lookup_is_recorded_in_history():
    st = new_state(
        "Who?",
        {"d1": {"id": "d1", "text": "Alice Smith visited Paris."}},
        harness_mask=_mask(),
    )
    st2, obs, ok = execute_tool(st, "lookup", {"eid": "e:missing_entity"})
    assert ok is False
    assert "eid_not_found" in obs
    assert "ERROR" in obs
    assert "Harness-G Working Memory" not in obs


def test_harness_g_legal_rate_counts_runtime_tools():
    names = ["init", "select", "lookup", "answer"]
    assert legal_rate(names, harness_g=True) == 1.0


def test_summarize_merged_traces_keeps_harness_g():
    rows = [{"query_id": "q1", "official_split": "test"}]
    traces = [{"query_id": "q1", "tool_names": ["init", "select"], "n_tool_calls": 2}]
    summary = summarize_merged_traces(
        traces,
        rows,
        leak_count=0,
        primary_split="test",
        harness_g=True,
    )
    assert summary["eval_harness"] == "Harness-G"
    assert summary["legal_action_rate"] == 1.0


def test_episode_timing_freezes_on_finish():
    timing = EpisodeTiming()
    time.sleep(0.01)
    timing.mark_finished()
    snap1 = timing.snapshot()["e2e_sec"]
    time.sleep(0.02)
    snap2 = timing.snapshot()["e2e_sec"]
    assert abs(snap1 - snap2) < 0.005
