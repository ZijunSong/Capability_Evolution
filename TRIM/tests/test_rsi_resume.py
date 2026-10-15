import pytest

from trim.rsi.controller import Campaign, CampaignError, run_campaign
from trim.rsi.splits import QuestionRecord, SplitError, assign_roles, consume_confirmation
from trim.rsi.types import PolicyArtifact
from trim.rsi.base_contract import build_base_contract


def _policy(digest: str, parent: str = "") -> PolicyArtifact:
    return PolicyArtifact(
        schema_version="trim_policy_artifact_v1",
        policy_id=digest,
        base_model_ref="base",
        base_model_sha256="basehash",
        export_dir=f"/tmp/{digest}",
        deployment="lora",
        tokenizer_ref="tok",
        chat_template_sha256="chat",
        digest=digest,
        round_id="r000",
        parent_digest=parent,
    )


def test_confirmation_panel_is_consumed_once():
    records = [
        QuestionRecord("q1", "c1"),
        QuestionRecord("q2", "c2"),
    ]
    manifest = assign_roles(records, roles={"confirm_h_r000": ["q1"], "work": ["q2"]})
    consume_confirmation(manifest, role="confirm_h_r000", round_id="r000", candidate_id="bank-a", policy_digest="p")
    consume_confirmation(manifest, role="confirm_h_r000", round_id="r000", candidate_id="bank-a", policy_digest="p")
    with pytest.raises(SplitError, match="second candidate"):
        consume_confirmation(
            manifest, role="confirm_h_r000", round_id="r000", candidate_id="bank-b", policy_digest="p"
        )


def test_resume_does_not_rerun_a_finished_job_and_rejects_a_changed_bank(tmp_path):
    calls = {"n": 0}

    def confirm(policy, bank, panel):
        calls["n"] += 1
        return {"pass_all_required_gates": False}

    from trim.rsi.controller import CampaignHooks

    records = [
        QuestionRecord("w", "cw"),
        QuestionRecord("s", "cs"),
        QuestionRecord("d", "cd"),
        QuestionRecord("h", "ch"),
        QuestionRecord("g", "cg"),
    ]
    splits = assign_roles(
        records,
        roles={
            "work": ["w"],
            "harness_select": ["s"],
            "weight_dev": ["d"],
            "confirm_h_r000": ["h"],
            "confirm_w_r000": ["g"],
        },
    )
    config = {
        "campaign_id": "c",
        "max_rounds": 1,
        "data": {"teacher_confirm_prefix": "confirm_h_r", "weight_confirm_prefix": "confirm_w_r"},
        "code_commit": "abc",
    }
    policy = _policy("theta")
    contract = build_base_contract(code_commit="abc", reward_protocol="final-answer-recall", metric_protocol="final_answer_recall")
    hooks = CampaignHooks(
        evaluate_zero=lambda policy, panel: {"quality": 0.1},
        evolve_bank=lambda policy, baseline, panel: {
            "has_eligible_champion": True,
            "champion": {"bank_sha256": "bank1"},
            "stop_reason": "CONFIRM_TEACHER",
        },
        confirm_teacher=confirm,
        launch_train=lambda *args: {"ok": False},
        select_checkpoint=lambda *args: _policy("child", parent="theta"),
        confirm_four_grid=lambda *args: {"promotable": False},
    )
    first = run_campaign(
        config, root=tmp_path / "run", policy=policy, base_contract=contract, splits=splits, hooks=hooks
    )
    assert first["status"] == "STOP_CONFIRMATION_FAILED"
    assert calls["n"] == 1
    second = run_campaign(
        config,
        root=tmp_path / "run",
        policy=policy,
        base_contract=contract,
        splits=splits,
        hooks=hooks,
        resume=True,
    )
    assert second["status"] == "STOP_CONFIRMATION_FAILED"
    assert calls["n"] == 1
    campaign = Campaign(tmp_path / "run", config, policy)
    campaign.state = __import__("json").loads((tmp_path / "run" / "STATE.json").read_text(encoding="utf-8"))
    with pytest.raises(CampaignError):
        campaign.job_result("r000:confirm_h:bank1", {"policy": "theta", "bank": "OTHER", "panel": ["h"]})
