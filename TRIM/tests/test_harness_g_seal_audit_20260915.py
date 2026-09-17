"""Regression fixtures from the 2026-09-15 Harness-G seal audit."""

from __future__ import annotations

from copy import deepcopy

from trim.adapters.harness_profiles import full_mask_for, zero_mask_for
from trim.eval.contract_fingerprint import collect_contract_fingerprint, fingerprints_compatible
from trim.eval.harness_g_env import (
    MAX_IDENTICAL_FAILURES,
    allowed_menu_pairs,
    execute_tool,
    new_state,
    wm_text,
)
from trim.eval.harness_g_graph import build_graph_from_documents, mixquery_text
from trim.eval.harness_g_runtime import parse_harness_g_action
from trim.eval.harness1_metrics import episode_quality_metrics
from trim.training.parse_rollout_action import parse_generated_action


def _zero():
    return zero_mask_for("Harness-G")


def _all():
    return full_mask_for("Harness-G")


def test_q1034_analysis_example_is_rejected():
    mask = _zero()
    text = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "Here is an example of finishing:\n"
        "to=functions.answer<|channel|>commentary<|message|>{\"reason\":\"done\"}<|call|>"
    )
    action, ok = parse_generated_action(text, None, enc=None, harness_mask=mask, finish_reason="stop")
    assert ok is False
    st = new_state("q", {"d": {"text": "Alice visited Paris."}}, harness_mask=mask)
    st, _, _ = execute_tool(st, "init", {})
    ended = bool(st.get("ended"))
    st2, obs, ok2 = execute_tool(st, action.get("name"), action.get("arguments") or {})
    assert ok2 is False
    assert st2.get("ended") == ended
    assert "ERROR" in obs


def test_invalid_json_init_is_rejected():
    text = (
        "<|start|>assistant to=functions.init<|channel|>commentary"
        "<|message|>invalid JSON<|call|>"
    )
    action, ok = parse_harness_g_action(text)
    assert ok is False
    assert action["name"] != "init"


def test_multiple_calls_are_rejected():
    text = (
        "<|start|>assistant to=functions.init<|channel|>commentary<|message|>{}<|call|>"
        "<|start|>assistant to=functions.select<|channel|>commentary"
        '<|message|>{"sid":"d:s0"}<|call|>'
    )
    action, ok = parse_harness_g_action(text)
    assert ok is False


def test_return_end_token_is_accepted_explicitly():
    text = (
        "<|start|>assistant to=functions.init<|channel|>commentary"
        "<|message|>{}<|return|>"
    )
    action, ok = parse_harness_g_action(text)
    assert ok is True
    assert action["name"] == "init"


def test_answer_with_select_then_same_sid_fails():
    mask = _all()
    st = new_state("Who is Alice?", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=mask)
    st, _, _ = execute_tool(st, "init", {})
    sid = next(a["sid"] for a in st["action_map"].values() if a.get("type") == "SELECT")
    st, _, ok = execute_tool(st, "select", {"sid": sid})
    assert ok is True
    st2, obs, ok2 = execute_tool(st, "answer_with", {"sid": sid})
    assert ok2 is False
    assert not st2.get("ended")
    assert "target_not_in_menu" in obs
    assert "ANSWER" in obs or "answer" in obs.lower()


def test_answer_with_atomically_selects_unselected_visible():
    mask = _all()
    st = new_state("Who is Alice?", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=mask)
    st, _, _ = execute_tool(st, "init", {})
    sid = next(a["sid"] for a in st["action_map"].values() if a.get("type") == "ANSWER_WITH")
    st2, _, ok = execute_tool(st, "answer_with", {"sid": sid})
    assert ok is True
    assert st2["ended"] is True
    assert sid in st2["selected_sids"]


def test_answer_not_in_post_init_menu():
    st = new_state("Who is Alice?", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    assert ("ANSWER", None) not in allowed_menu_pairs(st)
    st2, obs, ok = execute_tool(st, "answer", {"reason": "empty"})
    assert ok is False
    assert st2.get("ended") is False
    assert "target_not_in_menu" in obs


def test_every_menu_action_executes_on_state_copy():
    st = new_state(
        "Alice Smith Paris",
        {"d": {"text": "Alice Smith visited Paris. Bob Jones lived nearby."}},
        harness_mask=_all(),
    )
    st, _, _ = execute_tool(st, "init", {})
    for aid, action in list(st["action_map"].items()):
        clone = deepcopy({k: v for k, v in st.items() if k != "graph"})
        clone["graph"] = st["graph"]
        name = str(action.get("name") or action.get("type") or "").lower()
        args = {}
        if action.get("sid"):
            args["sid"] = action["sid"]
        if action.get("eid"):
            args["eid"] = action["eid"]
        _st2, _obs, ok = execute_tool(clone, name, args)
        assert ok is True, (aid, action, _obs)


def test_init_then_lookup_new_entity_without_forced_select():
    store = {
        "p1": {"text": "Alice met Bob Jones in Lyon."},
        "p2": {"text": "Bob Jones later published the secret number forty two."},
    }
    st = new_state("What number did Bob publish?", store, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    lookup_eids = [a.get("eid") for a in st["action_map"].values() if a.get("type") == "LOOKUP"]
    assert "e:alice" in lookup_eids or "e:bob_jones" in lookup_eids
    target = "e:alice" if "e:alice" in lookup_eids else lookup_eids[0]
    st, _, ok = execute_tool(st, "lookup", {"eid": target})
    assert ok is True
    next_eids = [a.get("eid") for a in st["action_map"].values() if a.get("type") == "LOOKUP"]
    assert next_eids, "post-lookup menu must offer entities from newly visible sentences"


def test_selection_order_preserved_for_mixquery():
    store = {
        "z": {"text": "Zulu unique_zulu_token."},
        "a": {"text": "Alpha unique_alpha_token."},
        "m": {"text": "Mike unique_mike_token."},
    }
    st = new_state("question tokens here", store, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    from trim.eval.harness_g_env import _lookup_mixquery, build_action_map

    st["visible_sids"] = ["z:s0", "a:s0", "m:s0"]
    st["action_map"] = build_action_map(st, include_answer=True)
    for sid in ("z:s0", "a:s0", "m:s0"):
        st, _, ok = execute_tool(st, "select", {"sid": sid})
        assert ok is True
        st["visible_sids"] = ["z:s0", "a:s0", "m:s0"]
        st["action_map"] = build_action_map(st, include_answer=True)
    assert st["selected_sids"] == ["z:s0", "a:s0", "m:s0"]
    query = _lookup_mixquery(st)
    assert query.index("unique_zulu_token") < query.index("unique_alpha_token") < query.index("unique_mike_token")


def test_mixquery_keeps_bridge_evidence_on_long_questions():
    question = " ".join(["question"] * 91)
    evidence = ["alpha bridge token uniquezzz", "mike bridge token uniquexxx"]
    meta = mixquery_text(question, evidence, question_word_budget=256, evidence_word_budget=64)
    assert "uniquezzz" in meta["query"]
    assert meta["question_words_used"] == 91
    assert meta["evidence_words_used"] > 0
    meta208 = mixquery_text(" ".join(["question"] * 208), evidence)
    assert "uniquezzz" in meta208["query"]


def test_lookup_ranks_later_relevant_sentence():
    store = {
        "fresh1": {
            "text": "Preface one. Preface two. Preface three. The decisive fact is forty two."
        },
        "fresh2": {"text": "The decisive fact is forty two."},
        "old": {"text": " ".join(f"Alice mentioned in filler {i}." for i in range(8))},
    }
    st = new_state("decisive fact forty two", store, harness_mask=_all())
    from trim.eval.harness_g_env import _lookup_sids

    visible = _lookup_sids(st, "e:alice", new_doc_ids=["fresh1", "fresh2"])
    texts = " ".join(str((st["sentences"].get(sid) or {}).get("text") or "") for sid in visible)
    assert "forty two" in texts


def test_full_sentence_is_shown_past_old_240_limit():
    fact = "THE_SECRET_ANSWER_IS_PARIS"
    prefix = "A" * 250
    st = new_state("Where?", {"d": {"text": prefix + " " + fact + "."}}, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    wm = wm_text(st)
    assert fact in wm
    sid = st["visible_sids"][0]
    st2, obs, ok = execute_tool(st, "select", {"sid": sid})
    assert ok is True
    assert fact in obs


def test_corpus_graph_reaches_support_doc_outside_initial_store():
    corpus = {
        "noise": {"text": "Unrelated weather notes from Lyon."},
        "bridge": {"text": "Alice met Bob Jones at the archive."},
        "target": {"text": "Bob Jones recorded that the treaty was signed in 1842."},
    }
    graph = build_graph_from_documents(corpus, scope="corpus")
    st = new_state(
        "When was the treaty signed?",
        {"noise": corpus["noise"], "bridge": corpus["bridge"]},
        harness_mask=_zero(),
        graph_index=graph,
    )
    assert st["graph_scope"] == "corpus"
    st, _, _ = execute_tool(st, "init", {})
    if "e:alice" in {a.get("eid") for a in st["action_map"].values()}:
        st, _, _ = execute_tool(st, "lookup", {"eid": "e:alice"})
    bob = next((a.get("eid") for a in st["action_map"].values() if a.get("eid") == "e:bob_jones"), None)
    if bob is None:
        # INIT may already expose Bob from the bridge page.
        bob = next((a.get("eid") for a in st["action_map"].values() if str(a.get("eid") or "").startswith("e:bob")), None)
    assert bob, st["action_map"]
    st, _, ok = execute_tool(st, "lookup", {"eid": bob})
    assert ok is True
    texts = " ".join(str((st["sentences"].get(sid) or {}).get("text") or "") for sid in st["visible_sids"])
    assert "1842" in texts or any(str(s).startswith("target:") for s in st["visible_sids"])


def test_uk_us_and_stopword_entities():
    graph = build_graph_from_documents(
        {
            "d": {
                "text": "The UK and US signed with NASA. For From Growing Article Year skipped. Alice Smith stayed."
            }
        }
    )
    eids = set(graph.entities)
    assert "e:uk" in eids
    assert "e:us" in eids
    assert "e:nasa" in eids
    assert "e:alice_smith" in eids
    assert "e:the" not in eids
    assert "e:for" not in eids
    assert "e:from" not in eids
    assert "e:article" not in eids


def test_synonyms_expand_when_enabled():
    store = {
        "d1": {"text": "The US signed the treaty."},
        "d2": {"text": "The United States later ratified it in 1842."},
    }
    on = new_state("When did the United States ratify?", store, harness_mask=_all())
    off = new_state("When did the United States ratify?", store, harness_mask=_zero())
    from trim.eval.harness_g_env import _lookup_sids

    vis_on = _lookup_sids(on, "e:us")
    vis_off = _lookup_sids(off, "e:us")
    on_docs = {str(s).split(":")[0] for s in vis_on}
    off_docs = {str(s).split(":")[0] for s in vis_off}
    assert "d2" in on_docs or vis_on != vis_off


def test_repeat_failure_caps_same_invalid_action():
    st = new_state("Who?", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_zero())
    st, _, _ = execute_tool(st, "init", {})
    sid = st["visible_sids"][0]
    st, _, ok = execute_tool(st, "select", {"sid": sid})
    assert ok is True
    last_obs = ""
    for _ in range(MAX_IDENTICAL_FAILURES):
        st, last_obs, ok = execute_tool(st, "select", {"sid": sid})
        assert ok is False
        assert st.get("ended") is False
    assert "protocol_failure" in last_obs


def test_component_lexical_hint_is_not_named_snc_in_wm():
    st = new_state("Alice", {"d": {"text": "Alice Smith visited Paris."}}, harness_mask=_all())
    st["visible_sids"] = list(st["sentences"])[:1]
    from trim.eval.harness_g_env import build_action_map

    menu = build_action_map(st, include_answer=True)
    selects = [a for a in menu.values() if a.get("type") == "SELECT"]
    assert selects and "lexical_frontier_hint" in selects[0]
    wm = wm_text({**st, "action_map": menu, "initialized": True})
    assert "lexical_hint=" in wm
    assert " snc=" not in wm


def test_gold_document_recall_field_and_g_select_stats():
    st = new_state("q", {"86987": {"text": "evidence."}}, harness_mask=_zero())
    st["selected_sids"] = ["86987:s0"]
    st["selected_docids"] = ["86987"]
    st["curated"] = {"86987": {"id": "86987", "text": "evidence."}}
    st["tool_history"] = [{"name": "select", "parse_ok": True, "schema_ok": True, "target_ok": True, "execution_ok": True}]
    stats = episode_quality_metrics(
        st,
        {"evidence_docids": ["86987"], "gold_docids": ["86987"]},
        tool_names=["init", "select", "truncated"],
        valids=[True, True, False],
        reward=0.0,
    )
    assert stats["final_answer_recall"] == 1.0
    assert stats["successful_select"] == 1.0
    assert stats["used_curate"] == 0.0
    assert "truncated" not in {n for n in ["init", "select"] if n == "truncated"}
    assert stats["tool_diversity"] == 2.0


def test_contract_fingerprint_stable_for_same_files():
    a = collect_contract_fingerprint(sampling={"max_turns": 40, "max_new_tokens": 2048})
    b = collect_contract_fingerprint(sampling={"max_turns": 40, "max_new_tokens": 2048})
    ok, reason = fingerprints_compatible(a, b)
    assert ok, reason
    c = collect_contract_fingerprint(sampling={"max_turns": 40, "max_new_tokens": 4096})
    ok2, _reason = fingerprints_compatible(a, c)
    assert ok2 is False


def test_graph_index_roundtrip_injected_into_eval_new_state(tmp_path):
    from trim.eval.harness_g_graph import load_graph_index, save_graph_index
    from trim.training.upstream_train_env import new_state_fn

    corpus = {
        "noise": {"text": "Unrelated weather notes from Lyon."},
        "bridge": {"text": "Alice met Bob Jones at the archive."},
        "target": {"text": "Bob Jones recorded that the treaty was signed in 1842."},
    }
    graph = build_graph_from_documents(corpus, scope="corpus")
    path = tmp_path / "corpus_graph.pkl"
    save_graph_index(graph, path)
    loaded = load_graph_index(path)
    assert loaded.scope == "corpus"
    assert loaded.content_fingerprint() == graph.content_fingerprint()
    fn = new_state_fn(
        train_env="local_legacy",
        harness_mask=_zero(),
        is_harness_g=True,
        graph_index=loaded,
    )
    st = fn("When was the treaty signed?", {"noise": corpus["noise"], "bridge": corpus["bridge"]}, "q1")
    assert st["graph_scope"] == "corpus"
    assert any(str(sid).startswith("target:") for sid in st["sentences"])
    a = collect_contract_fingerprint(graph_fingerprint=graph.content_fingerprint(), graph_scope="corpus")
    b = collect_contract_fingerprint(graph_fingerprint=loaded.content_fingerprint(), graph_scope="corpus")
    ok, reason = fingerprints_compatible(a, b)
    assert ok, reason
    c = collect_contract_fingerprint(graph_fingerprint="other", graph_scope="episode_doc_store")
    ok2, _reason = fingerprints_compatible(a, c)
    assert ok2 is False
