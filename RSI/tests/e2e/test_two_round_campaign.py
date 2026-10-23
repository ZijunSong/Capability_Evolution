"""E2E acceptance (toy scale, synthetic-marked):

- two-round closed loop with real training updates and promotions
- test_round_handoff: round 2 Actor/Teacher/proposer use theta_1
- test_resume_idempotence: interruption + resume never double-steps or
  double-consumes confirms

Toy runs are state-machine acceptance only (task book M5): performance claims
require real models/environments.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.campaign.controller import CampaignController
from rsi.campaign.registry import PolicyRegistry
from rsi.contracts.action import Finish, ToolCall
from rsi.evolution.proposer import FixtureProposer
from rsi.harness.loader import load_empty_bank

from helpers import HybridPolicy, toy_finalizer_patch  # noqa: E402

pytestmark = pytest.mark.e2e

_OBS = re.compile(r"\[\d+\] observation: (.*)")


def reader_plan(view_text: str, step: int):
    obs = _OBS.findall(view_text)[1:]
    n = len(obs)
    if n == 0:
        return ToolCall(name="read_file", arguments={"path": "a.txt"})
    if n == 1:
        return ToolCall(name="read_file", arguments={"path": "b.txt"})
    return Finish(answer=None)


def _tiny_cfg(tmp_path, campaign_id: str, max_rounds: int = 2):
    from rsi.config import load_config

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "toy_smoke.yaml"
    cfg = load_config(cfg_path)
    cfg.raw = dict(cfg.raw)
    cfg.raw["campaign_id"] = campaign_id
    cfg.raw["max_rounds"] = max_rounds
    cfg.raw["execution"] = {**cfg.section("execution"), "runs_root": str(tmp_path / "runs")}
    cfg.raw["evolution"] = {
        **cfg.section("evolution"),
        "max_generations": 1,
        "candidates_per_generation": 1,
        "max_candidate_evaluations": 2,
        "plateau_patience": 1,
        "dev_panel_per_domain": 2,
    }
    cfg.raw["training"] = {**cfg.section("training"), "max_optimizer_steps_per_round": 1, "batch_episodes": 2}
    cfg.raw["evaluation"] = {**cfg.section("evaluation"), "confirm_panel_per_domain": 2}
    return cfg


def _make_controller(cfg, run_dir: Path, n_patches: int = 4):
    empty = load_empty_bank()
    patch = toy_finalizer_patch(empty.bank_digest)
    return CampaignController(
        cfg=cfg,
        run_dir=run_dir,
        proposer_factory=lambda: FixtureProposer([patch] * n_patches),
        policy_factory=lambda: HybridPolicy(reader_plan),
    )


def test_two_round_closed_loop(tmp_path):
    cfg = _tiny_cfg(tmp_path, "e2e_two_round")
    run_dir = tmp_path / "runs" / "e2e_two_round"
    report = _make_controller(cfg, run_dir).run()
    assert report["state"] == "DONE"
    assert report["n_successful_promotions"] == 2
    assert report["real_training_executed"]
    assert report["shared_multidomain_evaluated"]
    assert report["synthetic_results"]  # toy -> refused from formal merge
    assert report["formal_eligible"] is False

    # frozen identity artifacts exist
    for name in ("campaign.json", "resolved_config.json", "state.json", "jobs.jsonl", "budget_ledger.jsonl"):
        assert (run_dir / name).exists()
    assert (run_dir / "final" / "FROZEN").is_file()
    assert (run_dir / "final" / "final_test.json").is_file()
    for r in ("r000", "r001"):
        for stage in ("baseline", "evolution", "confirm_harness", "probe", "training", "four_cells", "promotion_decision"):
            assert (run_dir / "rounds" / r / f"{stage}.json").exists()


def test_round_handoff(tmp_path):
    """After promotion, round 2 stages run under theta_1 and cold caches."""
    cfg = _tiny_cfg(tmp_path, "e2e_handoff")
    run_dir = tmp_path / "runs" / "e2e_handoff"
    report = _make_controller(cfg, run_dir).run()
    assert report["state"] == "DONE"

    registry = PolicyRegistry(run_dir / "policies")
    current = registry.current_policy()
    assert current is not None and current.kind == "trained"

    r0 = json.loads((run_dir / "rounds" / "r000" / "baseline.json").read_text())
    r1 = json.loads((run_dir / "rounds" / "r001" / "baseline.json").read_text())
    # round-2 baseline identity is the policy promoted at end of round 1,
    # not theta_0
    assert r1["policy_digest"] != r0["policy_digest"]

    # lineage: current (round-2 trained) descends from the policy round 2 ran under
    entry = json.loads((run_dir / "policies" / f"{current.policy_id}.json").read_text())
    assert entry["lineage"]["parent_policy_id"] == r1["policy_digest"]

    # score caches are keyed per JobKey: no cross-policy reuse
    cache_dir = run_dir / "score_cache"
    payloads = [json.loads(p.read_text()) for p in cache_dir.glob("*.json")]
    assert all("policy_digest" in p for p in payloads)

    # final report machine-readable levels
    assert report["n_successful_promotions"] == 2
    assert report["stopped_reason"] is None


class _Boom(Exception):
    pass


def test_resume_idempotence(tmp_path):
    """Interrupt before COMMIT_PROMOTION, resume: one optimizer step total per
    round, one confirm consumption per key, promotion happens once."""
    cfg = _tiny_cfg(tmp_path, "e2e_resume", max_rounds=1)
    run_dir = tmp_path / "runs" / "e2e_resume"
    ctrl = _make_controller(cfg, run_dir, n_patches=2)

    original = ctrl._stage_commit_promotion
    calls = {"n": 0}

    def exploding(registry):
        calls["n"] += 1
        raise _Boom("interrupt before promotion commit")

    ctrl._stage_commit_promotion = exploding  # type: ignore[method-assign]
    with pytest.raises(_Boom):
        ctrl.run()

    # resume via the documented entry point
    ctrl2 = CampaignController.resume(run_dir)
    empty = load_empty_bank()
    patch = toy_finalizer_patch(empty.bank_digest)
    ctrl2.proposer_factory = lambda: FixtureProposer([patch] * 2)
    ctrl2.policy_factory = lambda: HybridPolicy(reader_plan)
    report = ctrl2.run()
    assert report["state"] == "DONE"
    assert calls["n"] == 1  # interrupted exactly once

    # confirm panels consumed exactly once per key (same-stage reentry idempotent)
    ledger = json.loads((run_dir / "confirm_consumption.json").read_text())
    attempts = {k: v.get("attempts", 0) for k, v in ledger["consumed"].items()}
    assert all(a == 1 for a in attempts.values()), attempts

    # training ran without duplicating optimizer steps
    train = json.loads((run_dir / "rounds" / "r000" / "training.json").read_text())
    assert train["global_steps"] == 1
