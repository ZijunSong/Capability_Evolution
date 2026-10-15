import json
from pathlib import Path

from trim.evolution.operators import Proposal
from trim.evolution.proposer import ScriptedProposer
from trim.evolution.search import evolve_bank
from tests.rsi_support import write_bank


def _proposal(root: Path, kind: str, digest: str) -> Proposal:
    return Proposal(
        proposal_id=digest,
        proposal_type=kind,
        bank_dir=str(root),
        content_hash=digest,
        hypothesis={
            "failure_type": "bad_query",
            "trajectory_id": "traj-1",
            "mechanism": "repair a visible query",
            "expected_fix": "higher recall",
            "rollback": "drop the component",
        },
        adaptation={"source_component": "evo.query_repair", "route": "native_anchor"},
        examples={"positive": {}, "negative": {}, "missing_evidence": {}, "invalid_args": {}},
        touched_components=("evo.query_repair",),
        edits=1,
    )


def test_search_covers_gain_regression_cost_and_budget(tmp_path: Path):
    banks = {}
    for name in ("gain", "flat", "regress", "cost", "prune"):
        banks[name] = write_bank(tmp_path / name, component_ids=["evo.query_repair"])
    proposals = [
        _proposal(banks["gain"], "add", "gain"),
        _proposal(banks["flat"], "edit", "flat"),
        _proposal(banks["regress"], "edit", "regress"),
        _proposal(banks["cost"], "edit", "cost"),
        _proposal(banks["prune"], "prune", "prune"),
    ]
    scores = {
        "gain": {"quality": 0.30, "delta_h": 0.10, "retention_ok": True, "coverage": 0.4, "cost": {"total": 2}},
        "flat": {"quality": 0.20, "delta_h": 0.0, "retention_ok": True, "coverage": 0.2, "cost": {"total": 2}},
        "regress": {"quality": 0.05, "delta_h": -0.15, "retention_ok": False, "coverage": 0.2, "cost": {"total": 2}},
        "cost": {"quality": 0.30, "delta_h": 0.10, "retention_ok": True, "coverage": 0.2, "cost": {"total": 99}, "protocol_ok": False},
        "prune": {"quality": 0.30, "delta_h": 0.10, "retention_ok": True, "coverage": 0.5, "cost": {"total": 1}},
    }

    def score(proposal: Proposal) -> dict:
        payload = {
            "per_question": {"q": 1},
            "protocol_ok": True,
            "projection_complete": True,
            "used_unavailable_info": False,
            "distillable": True,
            "active_components": 1,
            "trigger_stats": {},
        }
        payload.update(scores[proposal.content_hash])
        return payload

    broken = Proposal(
        proposal_id="broken",
        proposal_type="add",
        bank_dir=str(banks["gain"]),
        content_hash="broken",
        hypothesis={},
        adaptation={},
        examples={},
        edits=1,
    )
    result = evolve_bank(
        proposer=ScriptedProposer([broken, *proposals]),
        score_fn=score,
        baseline_quality=0.20,
        min_generations=3,
        max_generations=4,
        max_candidate_evaluations=8,
        max_invalid_proposals=4,
        plateau_patience=3,
        epsilon_select=0.005,
        min_observed_gain=0.02,
        delta_sufficient=0.50,
    )
    assert result["invalid"] >= 1
    assert result["evaluated"] >= 4
    assert result["archive_distillable"] >= 1
    assert result["decision"] == "CONFIRM_TEACHER"
    json.dumps(result)
