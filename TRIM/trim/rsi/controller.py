"""Outer campaign state machine. Training itself stays in run_train_multinode.py."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from trim.rsi.base_contract import assert_base_contract
from trim.rsi.canonical import sha256_json
from trim.rsi.splits import SplitError, consume_confirmation
from trim.rsi.types import BudgetLedgerEntry, PolicyArtifact


class CampaignError(RuntimeError):
    pass


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class CampaignHooks:
    evaluate_zero: Callable[[PolicyArtifact, dict[str, Any]], dict[str, Any]]
    evolve_bank: Callable[[PolicyArtifact, dict[str, Any], dict[str, Any]], dict[str, Any]]
    confirm_teacher: Callable[[PolicyArtifact, dict[str, Any], dict[str, Any]], dict[str, Any]]
    launch_train: Callable[[PolicyArtifact, dict[str, Any], dict[str, Any]], dict[str, Any]]
    select_checkpoint: Callable[[dict[str, Any], dict[str, Any]], PolicyArtifact]
    confirm_four_grid: Callable[[PolicyArtifact, PolicyArtifact, dict[str, Any], dict[str, Any]], dict[str, Any]]


class Campaign:
    def __init__(self, root: Path, config: dict[str, Any], policy: PolicyArtifact) -> None:
        self.root = root
        self.config = config
        self.policy = policy
        self.state = {
            "campaign_id": config["campaign_id"],
            "status": "PREFLIGHT",
            "round_index": 0,
            "jobs": {},
            "code_commit": config.get("code_commit") or "",
            "config_sha256": sha256_json(config),
        }

    def pending_rounds(self) -> list[str]:
        done = int(self.state.get("round_index") or 0)
        return [f"r{index:03d}" for index in range(done, int(self.config["max_rounds"]))]

    def save(self) -> None:
        atomic_write_json(self.root / "STATE.json", self.state)
        atomic_write_json(self.root / "policies" / "current.json", self.policy.to_dict())

    def append_budget(self, entry: BudgetLedgerEntry) -> None:
        path = self.root / "BUDGET_LEDGER.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def stop(self, reason: str, *, round_id: str) -> dict[str, Any]:
        self.state["status"] = reason
        self.state["stop_round"] = round_id
        self.save()
        report = {
            "campaign_id": self.config["campaign_id"],
            "status": reason,
            "round_id": round_id,
            "policy_digest": self.policy.digest,
            "updated": False,
            "executed_training": False,
        }
        atomic_write_json(self.root / "FINAL_REPORT.json", report)
        return report

    def promote(self, candidate: PolicyArtifact, report: dict[str, Any]) -> PolicyArtifact:
        if candidate.digest == self.policy.base_model_sha256 and candidate.parent_digest != self.policy.digest:
            raise CampaignError("refusing to promote a checkpoint that fell back to the original base")
        if candidate.parent_digest != self.policy.digest:
            raise CampaignError("candidate parent digest is not the current policy")
        self.policy = candidate
        self.state["round_index"] = int(self.state.get("round_index") or 0) + 1
        self.state["status"] = report.get("status") or "PROMOTED"
        self.save()
        atomic_write_json(self.root / "rounds" / candidate.round_id / "PROMOTION.json", report)
        return candidate

    def lock_job(self, job_id: str, payload: dict[str, Any]) -> None:
        jobs = self.state.setdefault("jobs", {})
        prior = jobs.get(job_id)
        if prior and prior.get("payload_sha256") != sha256_json(payload):
            raise CampaignError(f"job {job_id} changed identity; refusing to reuse its result")
        if prior and prior.get("state") == "done":
            return
        jobs[job_id] = {
            "state": "running",
            "payload_sha256": sha256_json(payload),
            "payload": payload,
        }
        self.save()

    def finish_job(self, job_id: str, result: dict[str, Any]) -> None:
        job = self.state["jobs"][job_id]
        job["state"] = "done"
        job["result"] = result
        self.save()

    def job_result(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        prior = self.state.get("jobs", {}).get(job_id)
        if not prior:
            return None
        if prior.get("payload_sha256") != sha256_json(payload):
            raise CampaignError(f"job {job_id} does not match the frozen payload")
        if prior.get("state") == "done":
            return dict(prior.get("result") or {})
        return None


def run_campaign(
    config: dict[str, Any],
    *,
    root: Path,
    policy: PolicyArtifact,
    base_contract: dict[str, Any],
    splits: Any,
    hooks: CampaignHooks,
    resume: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    assert_base_contract(base_contract)
    if resume and (root / "STATE.json").is_file():
        campaign = Campaign(root, config, policy)
        campaign.state = _read_json(root / "STATE.json")
        if campaign.state.get("config_sha256") != sha256_json(config):
            raise CampaignError("resume config hash does not match STATE.json")
        current = _read_json(root / "policies" / "current.json")
        campaign.policy = PolicyArtifact.from_dict(current)
    else:
        campaign = Campaign(root, config, policy)
        atomic_write_json(root / "CAMPAIGN.json", {"campaign_id": config["campaign_id"], "config": config})
        atomic_write_json(root / "BASE_CONTRACT.json", base_contract)
        campaign.save()

    for round_id in list(campaign.pending_rounds()):
        h_role = f"{config['data']['teacher_confirm_prefix']}{round_id[-3:]}"
        w_role = f"{config['data']['weight_confirm_prefix']}{round_id[-3:]}"
        try:
            panel = {
                "round_id": round_id,
                "work": list(splits.ids("work")),
                "select": list(splits.ids("harness_select")),
                "weight_dev": list(splits.ids("weight_dev")),
                "confirm_h": list(splits.ids(h_role)),
                "confirm_w": list(splits.ids(w_role)),
                "dataset_manifest_hash": splits.dataset_manifest_hash,
            }
        except SplitError as exc:
            return campaign.stop(str(exc), round_id=round_id)
        round_dir = root / "rounds" / round_id
        atomic_write_json(round_dir / "PANEL.json", panel)
        baseline = hooks.evaluate_zero(campaign.policy, panel)
        atomic_write_json(round_dir / "ZERO_BASELINE.json", baseline)
        search = hooks.evolve_bank(campaign.policy, baseline, panel)
        atomic_write_json(round_dir / "SEARCH.json", search)
        if not search.get("has_eligible_champion"):
            return campaign.stop(str(search.get("stop_reason") or "STOP_NO_TEACHER_GAIN"), round_id=round_id)
        bank = dict(search["champion"])
        atomic_write_json(round_dir / "CHAMPION.json", bank)
        consume_confirmation(
            splits,
            role=h_role,
            round_id=round_id,
            candidate_id=str(bank.get("bank_sha256")),
            policy_digest=campaign.policy.digest,
        )
        atomic_write_json(root / "SPLITS.json", splits.to_dict())
        h_payload = {
            "policy": campaign.policy.digest,
            "bank": bank.get("bank_sha256"),
            "panel": panel["confirm_h"],
        }
        h_job = f"{round_id}:confirm_h:{bank.get('bank_sha256')}"
        cached = campaign.job_result(h_job, h_payload)
        if cached is None:
            campaign.lock_job(h_job, h_payload)
            h_report = hooks.confirm_teacher(campaign.policy, bank, panel)
            campaign.finish_job(h_job, h_report)
        else:
            h_report = cached
        atomic_write_json(round_dir / "TEACHER_CONFIRM.json", h_report)
        if not h_report.get("pass_all_required_gates"):
            return campaign.stop("STOP_CONFIRMATION_FAILED", round_id=round_id)
        train_payload = {
            "policy": campaign.policy.digest,
            "bank": bank.get("bank_sha256"),
            "work": panel["work"],
            "round_id": round_id,
        }
        train_job = f"{round_id}:train:{bank.get('bank_sha256')}:{campaign.policy.digest}"
        cached_train = campaign.job_result(train_job, train_payload)
        if cached_train is None:
            campaign.lock_job(train_job, train_payload)
            train_result = hooks.launch_train(campaign.policy, bank, panel)
            campaign.finish_job(train_job, train_result)
        else:
            train_result = cached_train
        atomic_write_json(round_dir / "TRAIN.json", train_result)
        if not train_result.get("ok"):
            return campaign.stop("STOP_TRAIN_FAILED", round_id=round_id)
        candidate = hooks.select_checkpoint(train_result, panel)
        if candidate.parent_digest != campaign.policy.digest:
            raise CampaignError("weight-dev checkpoint is not a child of the frozen round policy")
        consume_confirmation(
            splits,
            role=w_role,
            round_id=round_id,
            candidate_id=candidate.digest,
            policy_digest=campaign.policy.digest,
        )
        atomic_write_json(root / "SPLITS.json", splits.to_dict())
        grid = hooks.confirm_four_grid(campaign.policy, candidate, bank, panel)
        atomic_write_json(round_dir / "FOUR_GRID.json", grid)
        if not grid.get("promotable"):
            return campaign.stop("STOP_WEIGHT_PROMOTION_FAILED", round_id=round_id)
        campaign.promote(candidate, grid)
    campaign.state["status"] = "DONE"
    campaign.save()
    report = {
        "campaign_id": config["campaign_id"],
        "status": "DONE",
        "policy_digest": campaign.policy.digest,
        "updated": campaign.policy.digest != policy.digest,
        "executed_training": True,
    }
    atomic_write_json(root / "FINAL_REPORT.json", report)
    return report
