from trim.eval.harness1_metrics import episode_quality_metrics


def _metrics(selected, observed, gold, evidence):
    state = {
        "selected_docids": list(selected),
        "curated": {did: {} for did in selected},
        "observed_docids": list(observed),
        "pool": {did: {} for did in observed},
        "ended": True,
        "invalid_tools": 0,
        "tool_history": [],
    }
    row = {"gold_docids": list(gold), "evidence_docids": list(evidence), "query": "q"}
    return episode_quality_metrics(state, row, tool_names=["init", "select"], valids=[True, True], reward=0.0)


def test_q1115_evidence_hit_is_not_official_recall():
    gold = {"17478", "27358", "33780", "85950"}
    evidence = {"17478", "27358", "33780", "44721", "49570", "84397", "85950"}
    selected = {
        "61352", "45601", "922", "76368", "17190", "61786", "97031", "44721",
        "83935", "98342", "51985", "42351", "66458", "26126",
    }
    metrics = _metrics(selected, selected, gold, evidence)
    assert metrics["recall"] == 0.0
    assert metrics["gold_recall"] == 0.0
    assert metrics["evidence_recall"] == 1 / 7
    assert metrics["trajectory_recall"] == 0.0
    assert metrics["n_gold"] == 4
    assert metrics["n_gold_selected"] == 0
    assert metrics["n_evidence_selected"] == 1


def test_q1127_gold_hit_sanity():
    gold = {"86987", "99136"}
    evidence = {"86987", "99136"}
    selected = {"86987", "88633", "42247"}
    metrics = _metrics(selected, selected, gold, evidence)
    assert metrics["recall"] == 0.5
    assert metrics["gold_recall"] == 0.5
    assert metrics["evidence_recall"] == 0.5
    assert metrics["n_gold_selected"] == 1


def test_q153_empty_selection_and_single_gold_denominator():
    gold = {"41740"}
    evidence = {"19197", "41740", "57988"}
    empty = _metrics(set(), set(), gold, evidence)
    assert empty["recall"] == 0.0
    hit = _metrics({"41740"}, {"41740"}, gold, evidence)
    assert hit["recall"] == 1.0
    assert hit["gold_recall"] == 1.0
    assert hit["evidence_recall"] == 1 / 3
    assert hit["n_gold"] == 1
    assert hit["n_evidence"] == 3
