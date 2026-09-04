from trim.eval.harness1_metrics import (
    episode_quality_metrics,
    f1_score,
    f_beta_score,
    format_summary_table,
    set_precision,
    set_recall,
    summarize_quality_and_timing,
)
from trim.eval.sr_opd_four_cell_eval import summarize_traces


def test_set_recall_precision_and_f_scores():
    relevant = {"e1", "e2"}
    curated = {"e1", "noise"}
    assert set_recall(curated, relevant) == 0.5
    assert set_precision(curated, relevant) == 0.5
    assert abs(f1_score(0.5, 0.5) - 0.5) < 1e-9
    assert abs(f_beta_score(1.0, 0.5, beta=2.0) - (5 * 1.0 * 0.5) / (4 * 1.0 + 0.5)) < 1e-9


def test_episode_quality_metrics_match_harness1_document_formulas():
    state = {
        "curated": {"e1": {}, "g1": {}},
        "pool": {"e1": {}, "e2": {}, "noise": {}},
        "n_tool_calls": 4,
        "n_search_calls": 2,
        "invalid_tools": 0,
        "ended": True,
    }
    row = {
        "query_id": "q1",
        "query": "who",
        "evidence_docids": ["e1", "e2"],
        "gold_docids": ["g1"],
    }
    metrics = episode_quality_metrics(
        state,
        row,
        tool_names=["search_corpus", "curate", "read_document", "end_search"],
        valids=[True, True, True, True],
        reward=0.8,
        max_turns=6,
        timing={"e2e_sec": 1.5, "model_sec": 1.0, "harness_sec": 0.4, "elapsed_s": 1.5},
        actions=[{"name": "search_corpus", "arguments": {"query": "who"}}],
    )
    assert metrics["recall"] == 0.5
    assert metrics["trajectory_recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["final_answer_recall"] == 1.0
    assert metrics["trajectory_fa_recall"] == 1.0
    assert metrics["final_answer_found"] == 1.0
    assert metrics["n_curated"] == 2
    assert metrics["n_pool"] == 3
    assert metrics["num_turns"] == 4
    assert metrics["total_curate_calls"] == 1
    assert metrics["tool_diversity"] == 4
    assert metrics["error"] is False
    assert metrics["e2e_sec"] == 1.5
    assert metrics["model_sec"] == 1.0
    assert metrics["harness_sec"] == 0.4


def test_summarize_traces_includes_harness1_and_timing():
    traces = [
        {
            "query_id": "q1",
            "tool_names": ["search_corpus", "curate"],
            "recall": 0.5,
            "precision": 1.0,
            "f1": 2 / 3,
            "f_beta": 0.7,
            "trajectory_recall": 1.0,
            "final_answer_recall": 0.0,
            "trajectory_fa_recall": 0.0,
            "final_answer_found": 0.0,
            "reward": 0.2,
            "n_curated": 1,
            "n_pool": 3,
            "num_turns": 2,
            "tool_diversity": 2,
            "total_curate_calls": 1,
            "curate_rate": 0.5,
            "used_curate": 1.0,
            "no_error": 1.0,
            "error": False,
            "max_turns_reached": 0.0,
            "n_tool_calls": 2,
            "n_search_calls": 1,
            "evidence_recall_at_5": 0.5,
            "evidence_recall_at_100": 1.0,
            "e2e_sec": 2.0,
            "model_sec": 1.5,
            "harness_sec": 0.4,
        },
        {
            "query_id": "q2",
            "tool_names": ["search_corpus"],
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "f_beta": 0.0,
            "trajectory_recall": 0.5,
            "final_answer_recall": 0.0,
            "trajectory_fa_recall": 0.0,
            "final_answer_found": 0.0,
            "reward": 0.0,
            "n_curated": 0,
            "n_pool": 2,
            "num_turns": 1,
            "tool_diversity": 1,
            "total_curate_calls": 0,
            "curate_rate": 0.0,
            "used_curate": 0.0,
            "no_error": 1.0,
            "error": False,
            "max_turns_reached": 0.0,
            "n_tool_calls": 1,
            "n_search_calls": 1,
            "evidence_recall_at_5": 0.0,
            "evidence_recall_at_100": 0.0,
            "e2e_sec": 1.0,
            "model_sec": 0.8,
            "harness_sec": 0.2,
        },
    ]
    summary = summarize_traces(traces, setting="closed_loop", retrieval_name="bm25")
    extra = summarize_quality_and_timing(traces)
    assert summary["recall"] == 0.25
    assert summary["trajectory_recall"] == 0.75
    assert summary["precision"] == 0.5
    assert summary["legal_action_rate"] == 1.0
    assert summary["test_evidence_recall_at_5"] == 0.25
    assert abs(summary["mean_e2e_sec"] - 1.5) < 1e-9
    assert abs(summary["mean_model_sec"] - 1.15) < 1e-9
    assert abs(summary["mean_harness_sec"] - 0.3) < 1e-9
    assert summary["error_rate"] == 0.0
    assert extra["p50_e2e_sec"] in {1.0, 2.0}
    table = format_summary_table("eval", summary)
    assert "Curated Recall" in table
    assert "Model Time" in table
    assert "Harness Time" in table
    assert "E2E Time" in table
