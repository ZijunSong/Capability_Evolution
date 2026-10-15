import importlib.util
from pathlib import Path

import pytest

from trim.evolution.campaign import search_until_trim
from trim.evolution.materialize import parse_proposal_text, write_candidate_bank
from trim.evolution.operators import Proposal
from trim.evolution.proposer import ScriptedProposer


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_harness_evolve.py"
    spec = importlib.util.spec_from_file_location("run_harness_evolve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_32_gpu_self_evolution_and_stops_before_trim():
    mod = _load_script()
    assert mod.DEFAULT_NNODES * mod.DEFAULT_REPLICAS_PER_NODE == 32
    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "run_train" not in text
    assert "started_trim" in text


def test_search_can_become_ready_for_trim_without_launching_it(tmp_path: Path):
    payload = {
        "failure_type": "redundant_search",
        "trajectory_id": "traj-1",
        "mechanism": "rewrite the next search from unused visible words",
        "expected_fix": "fewer repeated searches",
        "rollback": "drop the component",
        "guidance": "After a search, use only words from the visible question that were not in the previous search.",
    }
    first = write_candidate_bank(tmp_path / "bank-a", payload, proposal_type="add")
    second = write_candidate_bank(tmp_path / "bank-b", payload, proposal_type="edit")

    def score(proposal: Proposal) -> dict:
        del proposal
        return {
            "quality": 0.20,
            "delta_h": 0.06,
            "per_question": {"n": 4},
            "cost": {"total": 1.0},
            "retention_ok": True,
            "protocol_ok": True,
            "projection_complete": True,
            "used_unavailable_info": False,
            "coverage": 0.5,
            "distillable": True,
            "active_components": 1,
            "trigger_stats": {},
        }

    result = search_until_trim(
        proposer=ScriptedProposer([first, second]),
        score_fn=score,
        baseline_quality=0.10,
        min_generations=1,
        max_generations=4,
    )
    assert result["ready_for_trim"] is True
    assert result["decision"] == "CONFIRM_TEACHER"
    assert result["started_trim"] is False
    assert result["executed_training"] is False
    assert result["next_stage"] == "trim"


def test_bad_proposer_json_is_rejected():
    with pytest.raises(ValueError):
        parse_proposal_text("no object here")


def test_component_and_train_flags_are_rejected():
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod._parse(
            [
                "--model_name",
                "m",
                "--zero-rollout-dir",
                "/tmp/zero",
                "--out",
                "/tmp/out",
                "--train_method",
                "trim",
            ]
        )
