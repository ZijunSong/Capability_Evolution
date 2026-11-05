"""Calibration math (§11) and stratified selection (§9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.probes.probe_c_calibrate import (
    C1Row,
    C3Row,
    calibrate,
    cost_accounting,
    load_c3_rows,
    prediction_scatter,
    selection_confusion,
    spearman,
    strategy_comparison,
)
from trim.probes.candidate_selector_v3 import CandidateRow, select_candidates_v3


def test_spearman_known_values():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [1, 2, 3]) != "N/A"  # ties handled
    assert spearman([1], [2]) == "N/A"
    assert spearman([1, 1, 1], [1, 2, 3]) == "N/A"  # zero variance


def test_prediction_scatter_pairs_only_finished_c3():
    c1 = [
        C1Row("a", 1, 8, r_d=0.1, r_n=0.2),
        C1Row("b", 1, 8, r_d=0.05, r_n=0.1),
        C1Row("c", 1, 8, r_d=0.0, r_n=0.0),
    ]
    c3 = [
        C3Row("a", g_50=0.2, e_50=0.1, verified=True),
        C3Row("b", g_50=0.1, e_50=0.05, verified=True),
        C3Row("c", g_50="N/A", e_50="N/A", verified=False, manifest_missing=["cost"]),
    ]
    out = prediction_scatter(c1, c3)
    assert out["n_paired"] == 3
    assert out["spearman"]["r_n_vs_e_50"] == pytest.approx(1.0)


def test_missing_c3_not_filled_negative():
    c1 = [C1Row("a", 1, 8, status="PROMOTE"), C1Row("b", 1, 8, status="DEPRIORITIZE")]
    c3 = [C3Row("a", g_50=0.2, e_50=0.1, verified=True, beneficial=True)]
    out = selection_confusion(c1, c3)
    dep = out["by_c1_status"]["DEPRIORITIZE"]
    assert dep["c3_missing"] == ["b"]
    assert dep["n_beneficial_of_finished"] == 0
    assert dep["beneficial_rate_of_finished"] == "N/A"
    pro = out["by_c1_status"]["PROMOTE"]
    assert pro["n_beneficial_of_finished"] == 1


def test_calibrate_warns_without_filtered_candidate():
    c1 = [C1Row("a", 1, 8, status="PROMOTE")]
    c3 = [C3Row("a", g_50=0.2, e_50=0.1, verified=True, beneficial=True)]
    result = calibrate(c1_rows=c1, c3_rows=c3, require_filtered_candidate=True)
    assert any("filtered-out" in w for w in result["warnings"])


def test_c3_missing_manifest_marked_unverified(tmp_path: Path):
    p = tmp_path / "c3.jsonl"
    p.write_text(
        json.dumps({"component_id": "a", "g_50": 0.1, "manifest": {"model_id": "m"}})
        + "\n",
        encoding="utf-8",
    )
    rows = load_c3_rows(p)
    assert rows[0].verified is False
    assert rows[0].label == "legacy/unverified"
    assert "harness_pair" in rows[0].manifest_missing


def test_strategy_comparison_budget_and_regret():
    c3 = [
        C3Row("a", g_50=0.3, verified=True),
        C3Row("b", g_50=0.1, verified=True),
        C3Row("c", g_50=0.2, verified=True),
    ]
    out = strategy_comparison(
        {"A_only": ["b"], "A_B_C": ["a"]},
        c3_rows=c3,
        total_budget_gpu_hours=10.0,
        costs={"a": 5.0, "b": 1.0, "c": 50.0},
    )
    assert out["strategies"]["A_B_C"]["mean_quality_of_measured"] == pytest.approx(0.3)
    assert out["strategies"]["A_only"]["regret_vs_best"] == pytest.approx(0.2)
    assert out["strategies"]["A_B_C"]["within_total_budget"] is True
    # The unselected component is still accounted.
    assert out["unselected_but_accounted"] == ["c"]


def test_cost_accounting_totals():
    ledger = [
        {"stage": "collect", "gpu_hours": 2.0},
        {"stage": "cache", "component_id": "a", "gpu_hours": 1.0, "n_teacher_calls": 10},
        {"stage": "screen", "component_id": "a", "gpu_hours": 3.0},
        {"stage": "confirm", "component_id": "a", "gpu_hours": 4.0, "n_task_calls": 2},
        {"stage": "calibrate_c3", "component_id": "x", "gpu_hours": 5.0},
    ]
    out = cost_accounting(ledger)
    assert out["C_new_total_gpu_hours"] == pytest.approx(15.0)
    assert out["teacher_calls"] == 10
    assert out["task_calls"] == 2


def test_selector_v3_stratified_and_missing_not_zero():
    rows = [
        CandidateRow("good", c_status="PROMOTE", r_n=0.2, r_d=0.1, cost_gpu_hours=1.0),
        CandidateRow("border", c_status="BORDERLINE", r_n=0.1, r_d=0.05, cost_gpu_hours=1.0),
        CandidateRow("no_c", c_status="NOT_RUN", r_n="N/A", r_d="N/A", cost_gpu_hours=1.0),
        CandidateRow("bad", valid=False, c_status="INVALID"),
        CandidateRow(
            "token_budget_marker",
            c_status="DEPRIORITIZE",
            r_n=-0.5,
            r_d=-0.5,
            cost_gpu_hours=1.0,
        ),
    ]
    out = select_candidates_v3(rows, top_k=4, force_runtime_anchors=True)
    selected = out["selected"]
    # Runtime anchors keep placement; then PROMOTE before BORDERLINE before no-evidence.
    assert selected[0] == "token_budget_marker"
    assert selected.index("good") < selected.index("border")
    assert selected.index("border") < selected.index("no_c")
    assert "bad" not in selected
    bad_rec = next(r for r in out["records"] if r["component_id"] == "bad")
    assert bad_rec["selected"] is False

    # Missing C evidence never sorts as a big win.
    rows2 = [
        CandidateRow("missing", c_status="PROMOTE", r_n="N/A", r_d="N/A", cost_gpu_hours=1.0),
        CandidateRow("real", c_status="PROMOTE", r_n=0.01, r_d=0.0, cost_gpu_hours=9.0),
    ]
    out2 = select_candidates_v3(rows2, force_runtime_anchors=False)
    assert out2["selected"] == ["real", "missing"]
