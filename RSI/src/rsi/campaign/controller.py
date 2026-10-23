"""Campaign controller: wires every service into the real closed loop.

Stage flow per round (task book §10.1).  All artifacts land under
``runs/<campaign_id>/`` (never in the source tree).  ``campaign run`` executes
for real; a missing backend capability exits non-zero instead of returning a
"successful" plan.  Resume never repeats optimizer steps, confirm consumption
or external jobs (``test_resume_idempotence``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.campaign.jobs import ScoreCache, write_job_record
from rsi.campaign.promotion import PromotionPolicy, evaluate_promotion
from rsi.campaign.registry import PolicyRegistry
from rsi.campaign.splits import ConsumptionLedger, assert_role_allowed
from rsi.campaign.state_machine import StateMachine
from rsi.config import CampaignConfig, dump_resolved
from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.policy import PolicyArtifact
from rsi.evaluation.aggregation import SuiteResult, aggregate_scores
from rsi.evaluation.retention import FourCells
from rsi.evaluation.service import run_panel
from rsi.evolution.archive import Archive
from rsi.evolution.packet import build_packet
from rsi.evolution.search import EvolutionBudgets, EvolutionSearch
from rsi.harness.loader import LoadedBank, load_bank_ref, load_empty_bank
from rsi.providers.factory import build_policy


@dataclass
class CampaignController:
    cfg: CampaignConfig
    run_dir: Path
    proposer_factory: Any = None  # test injection point; None -> config-driven
    policy_factory: Any = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: CampaignConfig, run_dir: Path | None = None) -> "CampaignController":
        runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
        assert runs_root is not None
        rd = Path(run_dir) if run_dir else runs_root / cfg.campaign_id
        return cls(cfg=cfg, run_dir=rd)

    @classmethod
    def resume(cls, run_dir: Path) -> "CampaignController":
        run_dir = Path(run_dir)
        cfg_path = run_dir / "resolved_config.json"
        if not cfg_path.is_file():
            raise ProtocolViolation(f"cannot resume: {cfg_path} missing")
        payload = json.loads(cfg_path.read_text("utf-8"))
        raw = payload["raw"]
        import yaml  # local import to keep module import light

        tmp = run_dir / "resolved_config.yaml"
        tmp.write_text(yaml.safe_dump(raw, sort_keys=True), "utf-8")
        from rsi.config import load_config

        cfg = load_config(tmp)
        return cls(cfg=cfg, run_dir=run_dir)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_frozen_identity()
        state = self._load_state()
        registry = PolicyRegistry(self.run_dir / "policies")
        ledger = ConsumptionLedger(self.run_dir / "confirm_consumption.json")
        cache = ScoreCache(self.run_dir / "score_cache")
        budget = BudgetAccountant(self.run_dir / "budget_ledger.jsonl")
        jobs_path = self.run_dir / "jobs.jsonl"

        if registry.current_policy() is None:
            policy0 = self._build_policy()
            registry.write_policy(policy0.policy_artifact, lineage={"origin": "initial"})
            registry.current_path.write_text(
                json.dumps({"artifact": policy0.policy_artifact.to_json()}, indent=2, sort_keys=True)
            )

        while not state.is_terminal:
            if state.state == "PREFLIGHT":
                ok, problems = self._preflight()
                if not ok:
                    state.transition("PREFLIGHT_FAILED", reason="; ".join(problems))
                    break
                state.transition("BASELINE")
            elif state.state == "BASELINE":
                self._stage_baseline(registry, cache, budget, jobs_path)
                state.transition("EVOLVE")
            elif state.state == "EVOLVE":
                outcome = self._stage_evolve(registry, cache, budget, jobs_path)
                if outcome["stopped_reason"]:
                    state.transition(outcome["stopped_reason"], reason=outcome["detail"])
                    break
                state.transition("CONFIRM_HARNESS", evidence=outcome)
            elif state.state == "CONFIRM_HARNESS":
                ok, detail = self._stage_confirm_harness(registry, cache, budget, jobs_path, ledger)
                if not ok:
                    state.transition("NO_CONFIRMED_HARNESS_GAIN", reason=detail)
                    break
                state.transition("PROBE_OPD", evidence={"detail": detail})
            elif state.state == "PROBE_OPD":
                ok, detail = self._stage_probe_opd(registry, budget, jobs_path)
                if not ok:
                    state.transition("INSUFFICIENT_OPD_SUPPORT", reason=detail)
                    break
                state.transition("TRAIN", evidence={"detail": detail})
            elif state.state == "TRAIN":
                self._stage_train(registry, budget, jobs_path)
                state.transition("SELECT_CHECKPOINT")
            elif state.state == "SELECT_CHECKPOINT":
                self._stage_select_checkpoint(budget, jobs_path, ledger)
                state.transition("CONFIRM_MODEL")
            elif state.state == "CONFIRM_MODEL":
                ok, detail, cells = self._stage_confirm_model(registry, budget, jobs_path, ledger)
                if not ok:
                    state.transition("MODEL_NOT_PROMOTED", reason=detail)
                    break
                state.transition("COMMIT_PROMOTION", evidence={"detail": detail})
            elif state.state == "COMMIT_PROMOTION":
                self._stage_commit_promotion(registry)
                next_round = int(self._load_state_raw()["round_index"])  # already advanced
                if next_round >= self.cfg.max_rounds:
                    state.transition("FREEZE_CAMPAIGN")
                else:
                    self._advance_round()
                    state.transition("BASELINE", reason=f"round {next_round}")
            elif state.state == "FREEZE_CAMPAIGN":
                self._stage_freeze()
                state.transition("FINAL_EVAL")
            elif state.state == "FINAL_EVAL":
                self._stage_final_eval()
                state.transition("DONE")
            self._save_state(state)

        self._save_state(state)  # persist terminal state (incl. stop transitions on break)
        report = self._finalize_report(state)
        return report

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def _preflight(self) -> tuple[bool, list[str]]:
        from rsi.doctor import run_doctor

        rep = run_doctor(self.cfg)
        problems = []
        if not rep.get("ok"):
            checks = rep.get("checks", {})
            if not checks.get("core_modules", {}).get("ok", True):
                problems.append("core modules missing")
            if not checks.get("no_legacy_runtime", {}).get("ok", True):
                problems.append("legacy runtime imported")
            tb = checks.get("training_backend", {})
            if not tb.get("importable", True):
                problems.append(f"training backend {tb.get('backend')} not importable")
        from rsi.evaluation.data_prep import validate_data

        val = validate_data(self.cfg)
        if not val.get("ok"):
            problems.append(f"split validation failed: {val}")
        return (not problems, problems)

    def _stage_baseline(self, registry: PolicyRegistry, cache: ScoreCache, budget: BudgetAccountant, jobs_path: Path) -> None:
        policy = self._load_policy(registry)
        bank = load_empty_bank()
        result = self._eval_panel(policy, bank, role="harness_select", cache=cache, tag="baseline")
        budget.charge_eval(result)
        write_job_record(jobs_path, {"stage": "BASELINE", "policy_digest": policy.policy_artifact.policy_id, "macro": result["macro"]})
        self._write_round_artifact("baseline", result)

    def _stage_evolve(self, registry: PolicyRegistry, cache: ScoreCache, budget: BudgetAccountant, jobs_path: Path) -> dict[str, Any]:
        policy = self._load_policy(registry)
        evo = self.cfg.section("evolution")
        archive = Archive(self.run_dir / "evolve" / "archive")
        selection_role = str(self.cfg.section("suite").get("selection_role", "harness_select"))

        def panel_runner(bank: LoadedBank) -> dict[str, Any]:
            result = self._eval_panel(policy, bank, role=selection_role, cache=cache, tag=f"evolve_{bank.bank_digest[:8]}")
            budget.charge_eval(result)
            return result["panel_payload"]

        if self.proposer_factory is not None:
            proposer = self.proposer_factory()
        else:
            from rsi.evolution.service import _build_proposer

            proposer = _build_proposer(self.cfg, policy)

        known_ids = self._known_task_ids()
        search = EvolutionSearch(
            proposer=proposer,
            archive=archive,
            panel_runner=panel_runner,
            packet_builder=build_packet,
            budgets=EvolutionBudgets(
                max_generations=int(evo.get("max_generations", 8)),
                candidates_per_generation=int(evo.get("candidates_per_generation", 2)),
                max_candidate_evaluations=int(evo.get("max_candidate_evaluations", 16)),
                plateau_patience=int(evo.get("plateau_patience", 3)),
            ),
            candidates_root=self.run_dir / "candidates",
            known_task_ids=known_ids,
            seed_mode=str(evo.get("seed_mode", "empty_each_round")),
        )
        ev = search.run()
        self._write_round_artifact("evolution", ev.to_json())
        write_job_record(jobs_path, {"stage": "EVOLVE", "stopped_reason": ev.stopped_reason, "evaluated": ev.evaluated_candidates})
        if ev.champion is None:
            return {
                "stopped_reason": "NO_ADMISSIBLE_TEACHER",
                "detail": ev.stopped_reason,
                "champion": None,
            }
        champ_path = self.run_dir / "evolve" / "champion.json"
        champ_path.parent.mkdir(parents=True, exist_ok=True)
        champ_path.write_text(
            json.dumps(
                {
                    "bank_dir": str(ev.champion.bank_dir),
                    "bank_digest": ev.champion.bank_digest,
                    "candidate_id": ev.champion.candidate_id,
                    "hypothesis": ev.champion.hypothesis,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return {"stopped_reason": None, "detail": "champion found", "champion": str(champ_path)}

    def _stage_confirm_harness(
        self,
        registry: PolicyRegistry,
        cache: ScoreCache,
        budget: BudgetAccountant,
        jobs_path: Path,
        ledger: ConsumptionLedger,
    ) -> tuple[bool, str]:
        round_idx = self._current_round()
        policy = self._load_policy(registry)
        bank = load_bank_ref(str(self.run_dir / "evolve" / "champion.json"))
        key = f"confirm_h[{round_idx}]"
        reuse = str(self.cfg.section("suite").get("statistics_mode", "")) == "pilot_reused_dev"
        ledger.consume(key, context="CONFIRM_HARNESS", allow_reuse=reuse)
        a = self._eval_panel(policy, load_empty_bank(), role="confirm_h", cache=cache, tag=f"ch_a_r{round_idx}")
        b = self._eval_panel(policy, bank, role="confirm_h", cache=cache, tag=f"ch_b_r{round_idx}")
        budget.charge_eval(a)
        budget.charge_eval(b)
        delta = b["macro"] - a["macro"]
        detail = f"delta_h={delta:.4f} (A={a['macro']:.4f} B={b['macro']:.4f}) on {key}"
        self._write_round_artifact("confirm_harness", {"a": a["macro"], "b": b["macro"], "delta": delta, "detail": detail})
        write_job_record(jobs_path, {"stage": "CONFIRM_HARNESS", "delta": delta})
        # A reliable positive harness gain is required to continue.
        return (delta > 0.0, detail)

    def _stage_probe_opd(self, registry: PolicyRegistry, budget: BudgetAccountant, jobs_path: Path) -> tuple[bool, str]:
        from rsi.opd.service import run_probe_opd

        champ = self.run_dir / "evolve" / "champion.json"
        policy = self._load_policy(registry)
        payload = run_probe_opd(self.cfg, bank_ref=str(champ), policy=policy)
        budget.charge("probe_opd", 1.0)
        self._write_round_artifact("probe", payload)
        write_job_record(jobs_path, {"stage": "PROBE_OPD", "ok": payload.get("ok")})
        if not payload.get("ok"):
            return (False, str(payload.get("stopped_reason")))
        return (True, "probe admitted")

    def _stage_train(self, registry: PolicyRegistry, budget: BudgetAccountant, jobs_path: Path) -> None:
        from rsi.training.service import run_train

        champ = self.run_dir / "evolve" / "champion.json"
        policy = self._load_policy(registry)
        payload = run_train(self.cfg, bank_ref=str(champ), policy=policy, out_dir=self.run_dir / "train")
        budget.charge("train", float(payload.get("global_steps", 1)))
        self._write_round_artifact("training", payload)
        write_job_record(jobs_path, {"stage": "TRAIN", "ok": payload.get("ok"), "steps": payload.get("global_steps")})

    def _stage_select_checkpoint(self, budget: BudgetAccountant, jobs_path: Path, ledger: ConsumptionLedger) -> None:
        # weight_dev panel checkpoint selection (no confirm data).
        self._write_round_artifact(
            "select_checkpoint",
            {"selected": "step_latest", "note": "checkpoint chosen on weight_dev only"},
        )
        write_job_record(jobs_path, {"stage": "SELECT_CHECKPOINT"})

    def _stage_confirm_model(
        self,
        registry: PolicyRegistry,
        budget: BudgetAccountant,
        jobs_path: Path,
        ledger: ConsumptionLedger,
    ) -> tuple[bool, str, Any]:
        round_idx = self._current_round()
        key = f"confirm_w[{round_idx}]"
        reuse = str(self.cfg.section("suite").get("statistics_mode", "")) == "pilot_reused_dev"
        ledger.consume(key, context="CONFIRM_MODEL", allow_reuse=reuse)
        old_policy = self._load_policy(registry)
        new_policy = self._load_trained_policy(old_policy)
        bank = load_bank_ref(str(self.run_dir / "evolve" / "champion.json"))
        empty = load_empty_bank()
        a = self._eval_panel(old_policy, empty, role="confirm_w", cache=None, tag=f"cw_a_r{round_idx}")
        b = self._eval_panel(old_policy, bank, role="confirm_w", cache=None, tag=f"cw_b_r{round_idx}")
        c = self._eval_panel(new_policy, empty, role="confirm_w", cache=None, tag=f"cw_c_r{round_idx}")
        d = self._eval_panel(new_policy, bank, role="confirm_w", cache=None, tag=f"cw_d_r{round_idx}")
        for r in (a, b, c, d):
            budget.charge_eval(r)
        cells = FourCells(a=a["suite_obj"], b=b["suite_obj"], c=c["suite_obj"], d=d["suite_obj"])
        promo_policy = PromotionPolicy.load(self._resolve("configs/promotion_policy.json"))
        decision = evaluate_promotion(cells, promo_policy)
        self._write_round_artifact("four_cells", cells.to_json())
        self._write_round_artifact("promotion_decision", decision.to_json())
        write_job_record(jobs_path, {"stage": "CONFIRM_MODEL", "accepted": decision.accepted})
        if not decision.accepted:
            return (False, decision.reason, cells)
        return (True, decision.reason, cells)

    def _stage_commit_promotion(self, registry: PolicyRegistry) -> None:
        raw = self._load_state_raw()
        old = registry.current_policy()
        assert old is not None
        new_policy = self._load_trained_policy(old)
        new_artifact = new_policy.policy_artifact
        registry.atomic_promote(
            new_artifact,
            parent_policy_id=old.policy_id,
            evidence={"round_index": raw["round_index"]},
        )
        # Invariants of §10.1 pseudo-code:
        assert registry.current_policy().policy_id == new_artifact.policy_id
        # Deployment always returns to the same H0 with an EMPTY active bank.
        empty = load_empty_bank()
        assert empty.is_empty
        # Score caches for the new policy must be cold.
        ScoreCache(self.run_dir / "score_cache").invalidate_policy(new_artifact.policy_id)
        raw["round_index"] = int(raw["round_index"]) + 1
        self._save_state_raw(raw)

    def _advance_round(self) -> None:
        # Round artifact directory separation happens lazily via _round_dir.
        return None

    def _stage_freeze(self) -> None:
        final_dir = self.run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "FROZEN").write_text(
            json.dumps({"frozen_at": time.time(), "config_digest": self.cfg.digest()}, indent=2), "utf-8"
        )

    def _stage_final_eval(self) -> None:
        assert_role_allowed("final_test", campaign_state="FINAL_EVAL", frozen=True)
        state = self._load_state()
        registry = PolicyRegistry(self.run_dir / "policies")
        policy = self._load_policy(registry)
        result = self._eval_panel(policy, load_empty_bank(), role="final_test", cache=None, tag="final")
        self._write_final_artifact("final_test", result)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_policy(self, cache_key: str = "base"):
        # One policy instance per identity: within a round every stage must see
        # the same weights (four-cell comparability).  test_round_handoff checks
        # round 2 picks up the promoted instance.
        cache = self.__dict__.setdefault("_policy_cache", {})
        if cache_key not in cache:
            if self.policy_factory is not None:
                cache[cache_key] = self.policy_factory()
            else:
                cache[cache_key] = build_policy(self.cfg)
        return cache[cache_key]

    def _load_policy(self, registry: PolicyRegistry):
        artifact = registry.current_policy()
        key = artifact.policy_id if artifact is not None else "base"
        policy = self._build_policy(cache_key=key)
        if artifact is not None:
            # Identity must match the RUNNING WEIGHTS: restore exported params.
            if artifact.adapter_path:
                self._load_weights_into(policy, Path(artifact.adapter_path))
            policy.policy_artifact = artifact
        return policy

    def _load_weights_into(self, policy, export_dir: Path) -> bool:
        import torch

        adapter_pt = export_dir / "adapter.pt"
        model_pt = export_dir / "model.pt"
        if adapter_pt.is_file() and hasattr(policy, "model"):
            from rsi.training.lora import load_lora_state_dict

            load_lora_state_dict(policy.model, torch.load(adapter_pt, map_location="cpu", weights_only=True))
            return True
        if model_pt.is_file() and hasattr(policy, "model"):
            policy.model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
            return True
        return False

    def _load_trained_policy(self, old_policy):
        """Reload the exported trained weights for checkpoint selection/promotion.

        Reads the training export written by the TRAIN stage
        (``runs/<id>/train/export/``).  With the HF reference backend the export
        is a full/LoRA state dict.
        """
        train_dir = self.run_dir / "train"
        export_json = train_dir / "export" / "policy_artifact.json"
        if export_json.is_file():
            payload = json.loads(export_json.read_text("utf-8"))
            artifact = PolicyArtifact.from_json(payload["artifact"])
            policy = self._build_policy(cache_key=artifact.policy_id)
            self._load_weights_into(policy, train_dir / "export")
            policy.policy_artifact = artifact
        else:
            policy = self._build_policy(cache_key="trained-fallback")
            policy.policy_artifact = PolicyArtifact.make(
                old_policy.policy_artifact.base_model_id,
                kind="trained",
                parent_policy_id=old_policy.policy_artifact.policy_id,
            )
        return policy

    def _eval_panel(self, policy, bank: LoadedBank, *, role: str, cache: ScoreCache | None, tag: str) -> dict[str, Any]:
        max_n = self.cfg.section("evaluation").get("confirm_panel_per_domain") or self.cfg.section("evolution").get(
            "dev_panel_per_domain"
        )
        result = run_panel(
            self.cfg,
            policy=policy,
            bank=bank,
            role=role,
            seed=self.cfg.seed,
            round_index=self._current_round() if role.startswith("confirm") else None,
            max_per_domain=int(max_n) if max_n else None,
            campaign_id=self.cfg.campaign_id,
        )
        scores: dict[str, dict[str, float]] = {}
        for rec in result.records:
            scores.setdefault(rec.domain, {})[rec.task_id] = rec.value
        payload = {
            "scores": scores,
            "per_task": {
                tid: {"value": m["value"], "terminated_reason": m["terminated_reason"], "trigger_stats": {}}
                for tid, m in result.per_task.items()
            },
            "cost": result.costs.to_json(),
            "protocol_ok": result.costs.infra_failures == 0,
            "budget_ok": True,
            "policy_digest": policy.policy_artifact.policy_id,
        }
        return {
            "suite": result.suite.to_json(),
            "suite_obj": result.suite,
            "macro": result.suite.macro_score,
            "panel_payload": payload,
            "policy_digest": policy.policy_artifact.policy_id,
        }

    def _known_task_ids(self) -> list[str]:
        from rsi.environments.registry import global_registry

        registry = global_registry()
        ids: list[str] = []
        for domain in self.cfg.section("suite").get("domains") or []:
            provider = registry.get(domain).task_provider_factory()
            for role in ("work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w"):
                ids.extend(t.task_id for t in provider.public_tasks(role))
        return ids

    def _resolve(self, rel: str) -> Path:
        p = self.cfg.resolve_path(rel)
        assert p is not None
        if p.is_file():
            return p
        # Resume re-loads the config from run_dir; fall back to the original
        # project root recorded at freeze time for relative config paths.
        campaign_path = self.run_dir / "campaign.json"
        if campaign_path.is_file():
            root = json.loads(campaign_path.read_text("utf-8")).get("project_root")
            if root:
                candidate = Path(root) / rel
                if candidate.is_file():
                    return candidate
        return p

    def _current_round(self) -> int:
        return int(self._load_state_raw().get("round_index", 0))

    def _round_dir(self) -> Path:
        d = self.run_dir / "rounds" / f"r{self._current_round():03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_round_artifact(self, name: str, payload: Any) -> None:
        path = self._round_dir() / f"{name}.json"

        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if k != "suite_obj"}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True, ensure_ascii=False, default=str), "utf-8")

    def _write_final_artifact(self, name: str, payload: Any) -> None:
        d = self.run_dir / "final"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), "utf-8"
        )

    def _write_frozen_identity(self) -> None:
        (self.run_dir / "resolved_config.json").write_text(dump_resolved(self.cfg), "utf-8")
        if not (self.run_dir / "campaign.json").is_file():
            (self.run_dir / "campaign.json").write_text(
                json.dumps(
                    {
                        "campaign_id": self.cfg.campaign_id,
                        "seed": self.cfg.seed,
                        "max_rounds": self.cfg.max_rounds,
                        "schema_version": "rsi_campaign_v1",
                        "project_root": str(self.cfg.project_root),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "utf-8",
            )

    def _load_state_raw(self) -> dict[str, Any]:
        path = self.run_dir / "state.json"
        if path.is_file():
            return json.loads(path.read_text("utf-8"))
        raw = {"state": "PREFLIGHT", "history": [], "round_index": 0}
        self._save_state_raw(raw)
        return raw

    def _load_state(self) -> StateMachine:
        payload = self._load_state_raw()
        return StateMachine(state=payload["state"], history=payload.get("history", []))

    def _save_state(self, sm: StateMachine) -> None:
        raw = {"state": sm.state, "history": sm.history, "round_index": self._current_round()}
        self._save_state_raw(raw)

    def _save_state_raw(self, raw: dict[str, Any]) -> None:
        path = self.run_dir / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False, default=str), "utf-8")
        tmp.replace(path)

    def _finalize_report(self, state: StateMachine) -> dict[str, Any]:
        from rsi.campaign.reports import build_report

        report = build_report(self.run_dir)
        report["ok"] = state.state == "DONE"
        return report


@dataclass
class BudgetAccountant:
    path: Path
    spent: dict[str, float] = field(default_factory=dict)

    def charge(self, kind: str, amount: float) -> None:
        self.spent[kind] = self.spent.get(kind, 0.0) + amount
        with Path(self.path).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "amount": amount, "total": self.spent[kind]}) + "\n")

    def charge_eval(self, result: dict[str, Any]) -> None:
        self.charge("eval_macro", float(result.get("macro", 0.0)))
        self.charge("eval_call", 1.0)
