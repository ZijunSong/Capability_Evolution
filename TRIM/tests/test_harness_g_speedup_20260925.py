"""Regression for the 2026-09-25 Harness-G speed and contract fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.adapters.components import full_mask, zero_mask
from trim.eval.harness1_metrics import summarize_quality_and_timing, trace_fields
from trim.eval.harness_g_contract import is_formal_harness_g_eval
from trim.eval.harness_g_env import (
    _frontier_from_observed_docs,
    execute_tool,
    new_state,
    record_nonexecution_failure,
)
from trim.eval.harness_g_graph import (
    GraphFullScanRefused,
    begin_scan_bucket,
    build_graph_from_documents,
    drain_scan_bucket,
    save_graph_index,
    write_graph_metadata_sidecar,
    read_graph_metadata_sidecar,
)
from trim.eval.harness_g_official import validate_official_run, write_partial_summary

_TRIM = Path(__file__).resolve().parents[1]


def _corpus_graph():
    docs = {
        "d1": {"text": "Alice Smith met Bob Jones in Paris. The treaty was signed in 1842."},
        "d2": {"text": "Carol Adams founded the company. Alice Smith later joined the board."},
        "d3": {"text": "Alpha signal one is here. Alpha signal two follows."},
        "d4": {"text": "Alpha signal three is here. Alpha signal four follows."},
    }
    return build_graph_from_documents(docs, scope="corpus")


def _state_for(graph, docs: list[str], *, selected: list[str] | None = None):
    return {
        "graph": graph,
        "query": "Alice Smith Bob Jones",
        "sentences": graph.sentences,
        "selected_sids": list(selected or []),
        "candidate_docids": list(docs),
        "graph_fingerprint": graph.content_fingerprint(),
    }


def test_formal_eval_covers_official_test166_and_test50():
    assert is_formal_harness_g_eval(harness="Harness-G", benchmark="bcplus_test_166")
    assert is_formal_harness_g_eval(harness="Harness-G", benchmark="bcplus_test_50")
    assert not is_formal_harness_g_eval(harness="Harness-G", benchmark="bcplus_test_166", smoke=True)
    assert not is_formal_harness_g_eval(harness="Harness-1", benchmark="bcplus_test_166")


def test_bridge_returns_a_real_target_and_cached_copy():
    graph = _corpus_graph()
    alice = next(eid for eid, rec in graph.entities.items() if "alice" in str(rec.get("surface") or "").lower())
    first = graph.propose_bridge_entities([alice], "Bob Jones Paris", [], topm=5)
    assert first, "bridge produced no target; the NameError path would also look empty"
    assert any(row["target_eid"] != alice for row in first)
    begin_scan_bucket()
    second = graph.propose_bridge_entities([alice], "Bob Jones Paris", [], topm=5)
    scans = drain_scan_bucket()
    assert scans["bridge"]["result_cache_hit"] >= 1
    assert second == first
    second[0]["score"] = -1
    third = graph.propose_bridge_entities([alice], "Bob Jones Paris", [], topm=5)
    assert third[0]["score"] != -1


def test_nonexecution_and_infrastructure_do_not_count_as_success(monkeypatch):
    st = new_state("Who is Alice Smith?", {"d1": {"text": "Alice Smith visited Paris."}}, harness_mask=zero_mask("Harness-G"))
    failed, _obs, ok = record_nonexecution_failure(
        st,
        "select",
        {"sid": "missing"},
        code="parse_failed",
        msg="could not parse",
    )
    assert ok is False
    assert failed.get("selected_sids") == st.get("selected_sids")
    hist = failed.get("tool_history") or []
    assert hist and hist[-1].get("execution_ok") is False

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk down")

    monkeypatch.setattr("trim.eval.harness_g_env._execute_tool_dispatch", _boom)
    infra, _obs, infra_ok = execute_tool(st, "init", {})
    assert infra_ok is False
    events = infra.get("turn_events") or []
    assert events[-1].get("error_class") == "infrastructure"
    assert infra.get("selected_sids") == st.get("selected_sids")


def test_trace_fields_keep_stage_timers_and_do_not_invent_missing_ones():
    exported = trace_fields(
        {"prompt_sec": 1, "execute_sec": 2, "parse_sec": 3, "menu_sec": 4, "harness_sec": 5}
    )
    assert exported["prompt_sec"] == 1
    assert exported["execute_sec"] == 2
    assert exported["parse_sec"] == 3
    assert exported["menu_sec"] == 4
    assert exported["harness_sec"] == 5
    assert "frontier_sec" not in exported
    summary = summarize_quality_and_timing(
        [
            {
                "e2e_sec": 3,
                "model_sec": 5,
                "harness_sec": 1,
                "model_sec_kind": "allocated_share",
                "generate_batches": [{"batch_id": "b1", "wall_sec": 10, "n_requests": 2}],
            },
            {
                "e2e_sec": 4,
                "model_sec": 5,
                "harness_sec": 1,
                "generate_batches": [{"batch_id": "b1", "wall_sec": 10, "n_requests": 2}],
            },
        ]
    )
    assert summary["sum_generate_batch_wall_sec"] == 10
    assert summary["sum_model_sec"] == 10
    assert "mean_frontier_sec" not in summary


def test_frontier_incremental_matches_full_rescan():
    graph = _corpus_graph()
    fresh_both = _frontier_from_observed_docs(_state_for(graph, ["d1", "d2"]), cap=8)
    growing = _state_for(graph, ["d1"])
    first = _frontier_from_observed_docs(growing, cap=8)
    assert growing["_frontier_last_new_docs"] == 1
    growing["candidate_docids"] = ["d1"]
    again = _frontier_from_observed_docs(growing, cap=8)
    assert growing["_frontier_last_new_docs"] == 0
    assert again == first
    growing["candidate_docids"] = ["d1", "d2"]
    expanded = _frontier_from_observed_docs(growing, cap=8)
    assert growing["_frontier_last_new_docs"] == 1
    assert expanded == fresh_both
    growing["candidate_docids"] = ["d1"]
    shrunk = _frontier_from_observed_docs(growing, cap=8)
    assert shrunk == first


def test_lookup_keeps_cross_document_rotation():
    graph = _corpus_graph()
    rows = graph.lookup_entity(
        "missing-eid",
        "alpha signal",
        topk=2,
        extra_sids=list(graph.sentences),
    )
    assert len({row["doc_id"] for row in rows}) == 2
    begin_scan_bucket()
    graph.lookup_entity(
        "missing-eid",
        "alpha signal",
        topk=2,
        extra_sids=[row["sid"] for row in rows],
    )
    scans = drain_scan_bucket()
    assert scans.get("lookup_rank", {}).get("token_cache_hit", 0) >= 1


def test_empty_corpus_retrieval_does_not_scan_every_sentence():
    graph = _corpus_graph()
    graph._refuse_full_scan = True
    calls = {"n": 0}

    def _boom(_query):
        calls["n"] += 1
        raise AssertionError("full scan")

    graph._rank_all = _boom
    with pytest.raises(GraphFullScanRefused):
        graph.rank_init_sids("zzzz-no-such-token", topk=3, doc_order=None, hybrid=False)
    assert calls["n"] == 0
    rows = graph.hybrid_initial_retrieve("zzzz-no-such-token", topk=3, doc_order=[])
    assert calls["n"] == 0
    assert isinstance(rows, list)


def test_official_summary_rejects_partial_and_shard_substitute(tmp_path: Path):
    planned = [str(i) for i in range(100)]
    run = tmp_path / "partial66"
    shard = run / "shards" / "harness_rank0"
    shard.mkdir(parents=True)
    (run / "LAUNCH.json").write_text(
        json.dumps(
            {
                "graph_index_path": "/tmp/g.pkl",
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
                "graph_required": True,
                "planned_query_ids": planned,
                "n_expected": 100,
                "benchmark": "bcplus_test_166",
            }
        )
    )
    (shard / "PER_QUERY.jsonl").write_text(json.dumps({"query_id": "0", "recall": 1}) + "\n")
    refused = validate_official_run(run)
    assert refused["ok"] is False
    assert any("substitute" in item for item in refused["failures"])

    rows = []
    for qid in planned[:66]:
        rows.append(
            {
                "query_id": qid,
                "gold_docids": ["t"],
                "selected_docids": [],
                "observed_docids": [],
                "recall": 0.0,
                "trajectory_recall": 0.0,
                "graph_enabled": True,
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
            }
        )
    (run / "PER_QUERY.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    verdict = validate_official_run(run)
    assert verdict["ok"] is False
    partial = write_partial_summary(run, verdict)
    assert partial["official"] is False
    assert partial["completed"] == 66
    assert partial["planned"] == 100
    assert partial["missing"] == 34
    assert not (run / "FOUR_CELL_OFFICIAL_SUMMARY.json").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "LAUNCH.json").write_text(
        json.dumps(
            {
                "graph_index_path": "/tmp/g.pkl",
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
                "planned_query_ids": ["a", "b"],
                "n_expected": 2,
            }
        )
    )
    (empty / "PER_QUERY.jsonl").write_text("")
    empty_verdict = validate_official_run(empty)
    assert empty_verdict["ok"] is False
    empty_partial = write_partial_summary(empty, empty_verdict)
    assert empty_partial["completed"] == 0
    assert empty_partial["planned"] == 2


def test_query_results_resume_skips_only_matching_completed(tmp_path: Path):
    from trim.eval.eval_shard_worker import _load_completed_queries, _persist_query

    identity = {
        "contract_sha256": "code",
        "graph_fingerprint": "graph",
        "model_path": "/m",
        "component": "all",
        "seed": 1,
        "max_turns": 4,
        "max_new_tokens": 8,
    }
    other = dict(identity)
    other["contract_sha256"] = "other"
    kept = _persist_query(tmp_path, {"query_id": "q1", "recall": 0.2}, identity)
    _persist_query(tmp_path, {"query_id": "q2", "recall": 0.0}, other)
    failed = tmp_path / "query_results" / "q3.json"
    failed.write_text(
        json.dumps({"status": "infra_failed", "query_id": "q3", "contract": identity, "trace": {"query_id": "q3"}})
    )
    loaded = _load_completed_queries(tmp_path, identity)
    assert set(loaded) == {"q1"}
    assert loaded["q1"]["recall"] == kept["recall"]


def test_graph_sidecar_must_match_file_identity(tmp_path: Path):
    graph = _corpus_graph()
    path = tmp_path / "graph.pkl"
    save_graph_index(graph, path)
    meta = graph.metadata()
    write_graph_metadata_sidecar(path, meta)
    loaded = read_graph_metadata_sidecar(path)
    assert loaded is not None
    assert loaded["graph_fingerprint"] == graph.content_fingerprint()
    assert loaded["graph_metadata_source"] == "sidecar"
    path.write_bytes(path.read_bytes() + b" ")
    assert read_graph_metadata_sidecar(path) is None


def test_launch_script_exposes_speed_controls_and_does_not_wipe_outputs():
    text = (_TRIM / "scripts" / "run_bcplus_test100_harness_g_gpu036_eval.sh").read_text(encoding="utf-8")
    assert "FAIL_FAST" in text
    assert "ENV_WORKERS" in text
    assert "EVAL_CHUNK_SIZE" in text
    assert "RESUME" in text
    assert 'rm -rf "${out}"' not in text
    assert "refuse to overwrite" in text


def test_stage_timers_do_not_change_frontier_order():
    graph = _corpus_graph()
    left = _frontier_from_observed_docs(_state_for(graph, ["d1", "d2"]), cap=8)
    right = _frontier_from_observed_docs(_state_for(graph, ["d1", "d2"]), cap=8)
    assert left == right
    st = new_state(
        "Alice Smith",
        {},
        harness_mask=full_mask("Harness-G"),
        graph_index=graph,
    )
    st["candidate_docids"] = ["d1", "d2"]
    st, _obs, ok = execute_tool(st, "init", {})
    assert ok is True
    assert "init_retrieve_sec" in (st.get("_stage_sec") or {})
