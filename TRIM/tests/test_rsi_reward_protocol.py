from trim.training.reward_protocol import score_terminal


def test_recall_only_has_no_shaping_and_is_stable():
    episode = {"selected_doc_ids": ["a"], "max_turns": 8}
    labels = {"gold_doc_ids": ["a", "b"]}
    first = score_terminal("final-answer-recall", episode, labels)
    second = score_terminal("final-answer-recall", episode, labels)
    assert first == second
    assert first["total"] == 0.5
    assert first["shaping"] == 0.0
    assert first["legal"] == 0.0
    assert first["penalty"] == 0.0


def test_composite_protocol_is_traceable_and_stable():
    episode = {
        "state": {"curated": {"a": {"id": "a"}}, "ended": True},
        "actions": [{"name": "curate", "arguments": {"add_ids": ["a"]}}],
        "valids": [True],
        "max_turns": 40,
        "termination_kind": "end_search",
    }
    labels = {"row": {"query": "q", "gold_docids": ["a"], "evidence_docids": ["a"]}}
    first = score_terminal("harness1-composite", episode, labels)
    second = score_terminal("harness1-composite", episode, labels)
    assert first == second
    assert first["protocol"] == "harness1_v3_defaults_on_trim_metrics_v1"
    assert "total" in first
