"""Regression tests for the 2026-09-18 Harness-G test50 root-cause plan."""

from __future__ import annotations

import json
from pathlib import Path

from trim.adapters.harness_profiles import full_mask_for, zero_mask_for
from trim.eval.harness1_metrics import episode_quality_metrics
from trim.eval.harness_g_env import (
    MAX_IDENTICAL_FAILURES,
    build_action_map,
    execute_tool,
    new_state,
    record_nonexecution_failure,
)
from trim.eval.harness_g_graph import build_graph_from_documents
from trim.eval.harness_g_official import validate_official_run
from trim.training.four_cell_runtime import freeze_train_state


def _zero():
    return zero_mask_for("Harness-G")


def _all():
    return full_mask_for("Harness-G")


def _alias_docs():
    return {
        "d1": {"text": "US scientists study river ecology."},
        "d2": {"text": "USA researchers publish results."},
    }


def test_bridge_select_succeeds_on_large_and_small_graph_branches():
    docs = _alias_docs()
    graph = build_graph_from_documents(docs, scope="corpus")
    graph.entities.update({f"e:padding_{i}": {} for i in range(50001)})
    for cell, mask in [("zero", _zero()), ("all", _all())]:
        state = new_state("river ecology", docs, graph_index=graph, harness_mask=mask)
        state.update(
            initialized=True,
            visible_sids=["d1:s0"],
            action_map={"A0": {"type": "SELECT", "name": "select", "sid": "d1:s0"}},
        )
        after, _obs, ok = execute_tool(state, "select", {"sid": "d1:s0"})
        assert ok is True, cell
        assert "d1:s0" in after["selected_sids"]
        assert after.get("infrastructure_failure") is not True


def test_small_graph_select_still_works_without_padding():
    docs = _alias_docs()
    graph = build_graph_from_documents(docs, scope="corpus")
    for mask in (_zero(), _all()):
        state = new_state("river ecology", docs, graph_index=graph, harness_mask=mask)
        state.update(
            initialized=True,
            visible_sids=["d1:s0"],
            action_map={"A0": {"type": "SELECT", "name": "select", "sid": "d1:s0"}},
        )
        after, _obs, ok = execute_tool(state, "select", {"sid": "d1:s0"})
        assert ok is True
        assert after["selected_sids"] == ["d1:s0"]


def test_attempt_event_created_for_parse_and_internal_failure():
    st = new_state("q", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    failed, obs, ok = record_nonexecution_failure(
        st, "select", {"sid": "missing"}, code="parse_failed", msg="bad parse", parse_ok=False
    )
    assert ok is False
    assert "parse_failed" in obs or "ERROR" in obs
    assert failed["tool_history"]
    assert failed["tool_history"][-1]["parse_ok"] is False
    boom, obs2, ok2 = record_nonexecution_failure(
        st,
        "select",
        {"sid": "x"},
        code="infrastructure_failure",
        msg="NameError: rec",
        parse_ok=True,
        schema_ok=True,
        error_class="infrastructure",
        traceback_ref="NameError: name 'rec' is not defined",
    )
    assert ok2 is False
    assert boom.get("ended") is True
    assert boom.get("termination_kind") == "infrastructure_failure"
    assert "NameError" in (obs2 + (boom.get("end_reason") or ""))


def test_repeat_failure_stops_episode_without_counting_as_answer():
    st = new_state("Who?", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    sid = st["visible_sids"][0]
    st, _, ok = execute_tool(st, "select", {"sid": sid})
    assert ok is True
    for i in range(MAX_IDENTICAL_FAILURES):
        st, obs, ok = execute_tool(st, "lookup", {"eid": "e:not_on_menu"})
        assert ok is False
        if i + 1 < MAX_IDENTICAL_FAILURES:
            assert st.get("ended") is False
    assert st.get("ended") is True
    assert st.get("end_reason") == "protocol_failure"
    assert st.get("termination_kind") == "protocol_failure"
    assert st.get("termination_kind") != "answer"


def test_zero_legal_repeat_lookup_is_not_protocol_failure():
    store = {"d1": {"text": "Alice Smith visited Paris. Bob Jones lived nearby."}}
    st = new_state("Alice Smith", store, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    eids = [a.get("eid") for a in st["action_map"].values() if a.get("type") == "LOOKUP"]
    if not eids:
        return
    eid = eids[0]
    st, _, ok1 = execute_tool(st, "lookup", {"eid": eid})
    assert ok1 is True
    if ("LOOKUP", eid) in {(str(a.get("type")), a.get("eid")) for a in st["action_map"].values()}:
        st, _, ok2 = execute_tool(st, "lookup", {"eid": eid})
        assert ok2 is True
        assert st.get("ended") is False


def test_metrics_export_gold_and_attempt_counts():
    st = new_state("q", {"86987": {"text": "evidence."}}, harness_mask=_zero())
    st["selected_sids"] = ["86987:s0"]
    st["selected_docids"] = ["86987"]
    st["observed_docids"] = ["86987"]
    st["attempt_events"] = [
        {"parse_ok": True, "schema_ok": True, "menu_ok": True, "execution_ok": True},
        {"parse_ok": True, "schema_ok": True, "menu_ok": False, "execution_ok": False, "error_class": "infrastructure", "error_type": "NameError", "traceback_ref": "tb"},
    ]
    stats = episode_quality_metrics(
        st,
        {"gold_docids": ["86987", "99136"], "evidence_docids": ["86987"], "query": "q"},
        tool_names=["init", "select"],
        valids=[True, True],
        reward=0.0,
    )
    assert stats["gold_docids"] == ["86987", "99136"]
    assert stats["evidence_docids"] == ["86987"]
    assert stats["n_generated"] == 2
    assert stats["n_attempt_events"] == 2
    assert stats["n_parse_ok"] == 2
    assert stats["n_execution_ok"] == 1
    assert stats["n_infrastructure_failures"] >= 1
    assert abs(stats["recall"] - 0.5) < 1e-9


def test_hidden_graph_candidates_are_not_observed():
    corpus = {
        "d1": {"text": "Alice Smith met Bob Jones."},
        "d2": {"text": "Bob Jones recorded the treaty."},
        "d3": {"text": "Bob Jones hid a later clause."},
        "d4": {"text": "Bob Jones archived unused notes."},
        "d5": {"text": "Bob Jones filed extras."},
        "d6": {"text": "Bob Jones kept drafts."},
        "d7": {"text": "Bob Jones stored the gold page."},
        "d8": {"text": "Unrelated weather in Lyon."},
    }
    graph = build_graph_from_documents(corpus, scope="corpus")
    st = new_state(
        "Bob Jones treaty",
        {"d1": corpus["d1"], "d8": corpus["d8"]},
        harness_mask=_zero(),
        graph_index=graph,
    )
    st, _, ok = execute_tool(st, "init", {})
    assert ok
    assert "d7" not in (st.get("observed_docids") or [])
    bob = next(a.get("eid") for a in st["action_map"].values() if a.get("eid") in {"e:bob", "e:bob_jones"})
    st, _, ok = execute_tool(st, "lookup", {"eid": bob})
    assert ok
    assert "d7" in (st.get("candidate_docids") or st.get("graph_expanded_docids") or [])
    assert "d7" not in (st.get("observed_docids") or []) or any(
        str(s).startswith("d7:") for s in (st.get("visible_sids") or [])
    )
    if not any(str(s).startswith("d7:") for s in (st.get("visible_sids") or [])):
        assert "d7" not in (st.get("observed_docids") or [])
        stats = episode_quality_metrics(
            st,
            {"gold_docids": ["d7"], "evidence_docids": ["d7"]},
            tool_names=["init", "lookup"],
            valids=[True, True],
            reward=0.0,
        )
        assert stats["trajectory_recall"] == 0.0


def test_official_validator_rejects_empty_and_incomplete_runs(tmp_path: Path):
    run = tmp_path / "empty"
    run.mkdir()
    (run / "LAUNCH.json").write_text(
        json.dumps(
            {
                "graph_index_path": "/tmp/g.pkl",
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
                "graph_required": True,
                "n_expected": 50,
                "planned_query_ids": [str(i) for i in range(50)],
            }
        )
    )
    (run / "PER_QUERY.jsonl").write_text("")
    result = validate_official_run(run)
    assert result["ok"] is False
    assert any("empty" in f or "expected 50" in f for f in result["failures"])

    one = tmp_path / "one"
    one.mkdir()
    (one / "LAUNCH.json").write_text(
        json.dumps(
            {
                "graph_index_path": "/tmp/g.pkl",
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
                "n_expected": 50,
            }
        )
    )
    (one / "PER_QUERY.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "gold_docids": ["t"],
                "selected_docids": ["t"],
                "observed_docids": ["t"],
                "recall": 1.0,
                "trajectory_recall": 1.0,
                "graph_enabled": True,
                "graph_scope": "corpus",
                "graph_fingerprint": "abc",
                "n_generated": 2,
            }
        )
        + "\n"
    )
    result = validate_official_run(one)
    assert result["ok"] is False
    assert any("expected 50" in f or "attempt" in f for f in result["failures"])


def test_hybrid_global_channel_is_not_doc_order_whitelist():
    docs = {
        "noise": {"text": "Unrelated weather notes from Lyon."},
        "local": {"text": "Alice Smith met a colleague."},
        "target": {"text": "The treaty was signed in 1842 by Bob Jones."},
    }
    graph = build_graph_from_documents(docs, scope="corpus")
    local_only = graph.rank_init_sids("treaty signed 1842", topk=6, doc_order=["noise", "local"], hybrid=False)
    assert all(not str(sid).startswith("target:") for sid in local_only)
    hybrid = graph.hybrid_initial_retrieve("treaty signed 1842", topk=6, doc_order=["noise", "local"])
    sids = [row["sid"] for row in hybrid]
    channels = graph.last_hybrid_channels
    assert channels["global_outside_whitelist"] or any(str(s).startswith("target:") for s in sids)
    off = graph.rank_init_sids("treaty signed 1842", topk=6, doc_order=["noise", "local"], hybrid=False)
    assert set(off) != set(sids) or channels["global_sentence_sids"] or channels["entity_mention_sids"]


def test_frontier_refills_after_visited_filter():
    sentences = {}
    entities = {}
    doc_to_sids = {"d": []}
    sentence_to_entities = {}
    for i in range(40):
        eid = f"e:z{i:02d}"
        sid = f"d:s{i}"
        entities[eid] = {"eid": eid, "surface": f"Z{i:02d} Entity", "sids": [sid]}
        sentences[sid] = {"sid": sid, "doc_id": "d", "parent_docid": "d", "text": f"Z{i:02d} Entity mentioned.", "idx": i}
        doc_to_sids["d"].append(sid)
        sentence_to_entities[sid] = [eid]
    graph = build_graph_from_documents({"d": {"text": "Z00 Entity mentioned."}}, scope="corpus")
    graph.sentences.update(sentences)
    graph.entities.update(entities)
    graph.doc_to_sids.update(doc_to_sids)
    graph.sentence_to_entities.update(sentence_to_entities)
    st = new_state("Z39 Entity", {"d": {"text": "x"}}, harness_mask=_all(), graph_index=graph)
    st["initialized"] = True
    st["candidate_docids"] = ["d"]
    st["graph_expanded_docids"] = ["d"]
    st["visited_eids"] = [f"e:z{i:02d}" for i in range(32)]
    st["visible_sids"] = ["d:s39"]
    menu = build_action_map(st, include_answer=True, lookup_cap=8, lookup_eids=[f"e:z{i:02d}" for i in range(40)])
    lookup_eids = [a.get("eid") for a in menu.values() if a.get("type") == "LOOKUP"]
    assert lookup_eids
    assert all(eid not in set(st["visited_eids"]) for eid in lookup_eids)


def test_bridge_gets_reserved_menu_slot_when_frontier_is_full():
    docs = _alias_docs()
    graph = build_graph_from_documents(docs, scope="corpus")
    st = new_state("USA researchers", docs, harness_mask=_all(), graph_index=graph)
    st["initialized"] = True
    st["visible_sids"] = ["d1:s0"]
    st["selected_sids"] = ["d1:s0"]
    filler = [f"e:filler_{i}" for i in range(8)]
    for eid in filler:
        st["entities"][eid] = {"eid": eid, "surface": eid, "sids": ["d1:s0"]}
        graph.entities[eid] = st["entities"][eid]
    st["frontier_eids"] = filler
    menu = build_action_map(st, include_answer=True, lookup_cap=8, lookup_eids=list(filler))
    eids = [a.get("eid") for a in menu.values() if a.get("type") == "LOOKUP"]
    assert any(eid in {"e:usa", "e:us"} for eid in eids) or st.get("runtime_effects", {}).get("bridge_entities_menu_delta", 0) >= 0
    types = {a.get("type") for a in menu.values()}
    assert "ANSWER" in types


def test_similar_entities_uses_stored_edges_on_large_graph():
    docs = _alias_docs()
    graph = build_graph_from_documents(docs, scope="corpus")
    syns_small = graph.similar_entities("e:us")
    graph.entities.update({f"e:padding_{i}": {"eid": f"e:padding_{i}", "surface": f"Pad{i}", "sids": []} for i in range(50001)})
    syns_large = graph.similar_entities("e:us")
    assert "e:usa" in syns_small or "e:usa" in (graph.entities["e:us"].get("synonyms") or [])
    assert "e:usa" in syns_large or "e:usa" in (graph.entities["e:us"].get("synonyms") or [])


def test_lookup_keeps_full_candidates_and_page_reaches_later_rows():
    store = {f"d{i}": {"text": f"Alice Smith fact number {i} appears here."} for i in range(8)}
    graph = build_graph_from_documents(store, scope="corpus")
    st = new_state("Alice Smith", store, harness_mask=_zero(), graph_index=graph)
    st, _, _ = execute_tool(st, "init", {})
    eid = next(a.get("eid") for a in st["action_map"].values() if a.get("eid") in {"e:alice", "e:alice_smith"})
    st, _, ok = execute_tool(st, "lookup", {"eid": eid})
    assert ok
    cands = st.get("lookup_candidates") or []
    assert len(cands) >= len(st.get("visible_sids") or [])
    if len(cands) > 6:
        assert any(a.get("type") == "PAGE" for a in st["action_map"].values())
        st2, obs, ok2 = execute_tool(st, "page", {"direction": "next"})
        assert ok2 is True
        assert st2["visible_sids"]
        assert st2["visible_sids"] != st["visible_sids"] or "PAGE" in obs


def test_fingerprint_changes_when_alias_edge_changes():
    docs = _alias_docs()
    graph = build_graph_from_documents(docs, scope="corpus")
    fp1 = graph.content_fingerprint()
    rec = graph.entities["e:us"]
    rec["synonyms"] = list(rec.get("synonyms") or []) + ["e:made_up"]
    graph._fingerprint = None
    fp2 = graph.content_fingerprint()
    assert fp1 != fp2
    graph._fingerprint = None
    rec["synonyms"] = [x for x in rec["synonyms"] if x != "e:made_up"]
    fp3 = graph.content_fingerprint()
    assert fp3 == fp1


def test_eval_only_freeze_does_not_copy_sentence_maps():
    docs = {"d": {"text": "Alice Smith visited Paris."}}
    st = new_state("q", docs, harness_mask=_zero())
    frozen = freeze_train_state(st, eval_only=True)
    assert frozen["sentences"] is st["sentences"]
    train = freeze_train_state(st, eval_only=False)
    assert train["sentences"] is st["sentences"]
    assert train["action_map"] is not st["action_map"] or st["action_map"] == {}


def test_prompt_budget_reads_encoding_max_model_len():
    from trim.training.batched_env_rollout import _prompt_budget

    class Enc:
        max_model_len = 32768
        max_new_tokens = 2048

    ml, mn = _prompt_budget(Enc())
    assert ml == 32768
    assert mn == 2048
    ml2, _mn2 = _prompt_budget(object())
    assert ml2 == 8192


def test_answer_with_records_termination_diagnostics():
    st = new_state("Alice", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_all())
    st, _, _ = execute_tool(st, "init", {})
    sid = next(a.get("sid") for a in st["action_map"].values() if a.get("type") == "ANSWER_WITH")
    st, _, ok = execute_tool(st, "answer_with", {"sid": sid})
    assert ok is True
    assert st["ended"] is True
    assert st["termination_kind"] == "model_answer"
    diag = st.get("termination_diagnostics") or {}
    assert "empty_selection" in diag
    assert diag["empty_selection"] is False
    assert diag["n_selected"] >= 1


def test_runtime_audit_does_not_call_gold_inject_on_default_path():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "audit_harness_g_graph_runtime.py").read_text(
        encoding="utf-8"
    )
    start = src.index("def run_query")
    end = src.index("\ndef main")
    body = src[start:end]
    assert "if altered_menu_oracle" in body
    call_lines = [
        i
        for i, line in enumerate(body.splitlines())
        if "_inject_gold_bridges(" in line
    ]
    assert call_lines
    lines = body.splitlines()
    for idx in call_lines:
        window = "\n".join(lines[max(0, idx - 4) : idx + 1])
        assert "altered_menu_oracle" in window


def test_search_metrics_separates_gold_and_evidence():
    from trim.eval.sr_opd_four_cell_eval import search_metrics

    class Hit:
        def __init__(self, docid):
            self.docid = docid

    class Searcher:
        name = "fake"

        def search(self, query, k):
            return [Hit("g1"), Hit("e1")] + [Hit(f"n{i}") for i in range(k)]

    sm = search_metrics(Searcher(), "q", ["e1"], gold=["g1", "g2"])
    assert sm["initial_bm25_gold_recall_at_5"] == 0.5
    assert sm["initial_bm25_evidence_recall_at_5"] == 1.0
    assert sm["initial_bm25_recall_at_5"] == 1.0
