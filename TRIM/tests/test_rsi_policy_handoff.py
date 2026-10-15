from trim.rsi.controller import Campaign, CampaignError
from trim.rsi.types import PolicyArtifact


def _policy(digest: str, parent: str, round_id: str = "r000") -> PolicyArtifact:
    return PolicyArtifact(
        schema_version="trim_policy_artifact_v1",
        policy_id=digest,
        base_model_ref="base",
        base_model_sha256="basehash",
        export_dir=f"/models/{digest}",
        deployment="lora",
        tokenizer_ref="tok",
        chat_template_sha256="chat",
        digest=digest,
        round_id=round_id,
        parent_digest=parent,
    )


def test_promoted_policy_is_the_child_not_the_base(tmp_path):
    current = _policy("theta-r", parent="basehash")
    campaign = Campaign(tmp_path, {"campaign_id": "c", "max_rounds": 2}, current)
    child = _policy("theta-r1", parent="theta-r", round_id="r000")
    campaign.promote(child, {"status": "PROMOTED_PARTIAL_INTERNALIZATION", "promotable": True})
    assert campaign.policy.digest == "theta-r1"
    assert campaign.policy.digest != campaign.policy.base_model_sha256
    assert campaign.policy.parent_digest == "theta-r"


def test_base_fallback_is_rejected(tmp_path):
    current = _policy("theta-r", parent="basehash")
    campaign = Campaign(tmp_path, {"campaign_id": "c", "max_rounds": 2}, current)
    fallen = _policy("basehash", parent="someone-else")
    try:
        campaign.promote(fallen, {"promotable": True})
    except CampaignError as exc:
        assert "base" in str(exc) or "parent" in str(exc)
    else:
        raise AssertionError("expected rejection")
