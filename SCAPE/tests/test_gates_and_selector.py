from __future__ import annotations

from scape.eval.retirement import evaluate_gate_s
from scape.probes.candidate_selector import select_candidates
from scape.probes.learnability import LearnabilityCurve, evaluate_gate_l


def test_candidate_selector_excludes_runtime_anchors():
    rows = [
        {
            "component_id": "evidence_graph",
            "contribution": 0.05,
            "influence_above_null": 0.2,
            "runtime_cost": 1.0,
            "quality_positive": True,
        },
        {
            "component_id": "importance_tagging",
            "contribution": 0.04,
            "influence_above_null": 0.15,
            "runtime_cost": 1.2,
            "quality_positive": True,
        },
        {
            "component_id": "content_dedup",
            "contribution": 0.1,
            "influence_above_null": 0.3,
            "runtime_cost": 0.5,
            "quality_positive": True,
        },
        {
            "component_id": "token_budget_marker",
            "contribution": 0.01,
            "influence_above_null": 0.01,
            "runtime_cost": 0.1,
            "quality_positive": True,
        },
    ]
    out = select_candidates(rows, top_k=2)
    assert out["n_selected"] == 2
    selected_ids = {c["component_id"] for c in out["candidates"].values()}
    assert "content_dedup" not in selected_ids
    assert "token_budget_marker" not in selected_ids
    assert "evidence_graph" in selected_ids


def test_gate_l_and_gate_s():
    curves = [
        LearnabilityCurve(
            component_id="evidence_graph",
            seed=42,
            d_pre=1.0,
            d_post_by_n={512: 0.7, 2000: 0.5, 8000: 0.4},
            invalid_tool_rate_pre=0.02,
            invalid_tool_rate_post_by_n={512: 0.02, 2000: 0.01, 8000: 0.01},
        ),
        LearnabilityCurve(
            component_id="evidence_graph",
            seed=43,
            d_pre=1.1,
            d_post_by_n={512: 0.8, 2000: 0.55, 8000: 0.45},
            invalid_tool_rate_pre=0.03,
            invalid_tool_rate_post_by_n={512: 0.03, 2000: 0.02, 8000: 0.02},
        ),
    ]
    gl = evaluate_gate_l(curves)
    assert gl["pass"] is True

    grid = {
        "S0": {"quality": 0.40, "cost": 10.0},
        "S1": {"quality": 0.30, "cost": 7.0},
        "S2": {"quality": 0.41, "cost": 7.0},
        "S3": {"quality": 0.42, "cost": 10.0},
    }
    gs = evaluate_gate_s(grid)
    assert gs["pass"] is True
    assert gs["can_claim_retired"] is True
