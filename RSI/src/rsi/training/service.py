"""CLI service for HarnessOPD internalization (`rsi.cli train`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import ProtocolViolation
from rsi.harness.loader import load_bank_ref, load_empty_bank
from rsi.opd.recipes import Recipe
from rsi.providers.factory import build_policy
from rsi.runtime.budgets import BudgetSpec
from rsi.training.backend import GlobalNormalizers  # noqa: F401  (documented contract)
from rsi.training.checkpoint import CheckpointIdentity, save_checkpoint
from rsi.training.collector import collect_rollouts
from rsi.training.export import export_adapter
from rsi.training.joint import JointStepReport, MethodSpec, joint_update


def _build_backend(cfg: CampaignConfig, policy: Any):
    training = cfg.section("training")
    backend_name = str(training.get("backend", "hf_reference"))
    lambda_opd = float(cfg.section("opd").get("lambda", 0.01))
    gate_beta = float(cfg.section("opd").get("gate_beta", 5.0))
    lr = float(training.get("learning_rate", 1e-4))
    if backend_name == "hf_reference":
        from rsi.training.hf_reference import HFReferenceBackend

        model = getattr(policy, "model", None)
        if model is None:
            raise ProtocolViolation("hf_reference backend requires a policy with a torch model")
        return HFReferenceBackend(model=model, lambda_opd=lambda_opd, gate_beta=gate_beta, lr=lr)
    if backend_name == "torch_fsdp2":
        from rsi.training.fsdp2_backend import FSDP2Backend

        model = getattr(policy, "model", None)
        if model is None:
            raise ProtocolViolation("torch_fsdp2 backend requires a policy with a torch model")
        return FSDP2Backend(model=model, lambda_opd=lambda_opd, gate_beta=gate_beta, lr=lr)
    raise ProtocolViolation(f"unknown training backend {backend_name!r}")


def run_train(cfg: CampaignConfig, *, bank_ref: str, policy: Any | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    bank = load_bank_ref(bank_ref)  # Teacher bank (H0 + Bk); Student runs H0 only.
    student_bank = load_empty_bank()
    policy = policy if policy is not None else build_policy(cfg)
    teacher_policy = policy  # batch-snapshot self-distillation: same weights
    backend = _build_backend(cfg, policy)
    spec = MethodSpec.resolve(str(cfg.section("training").get("method", "harness_opd_rl")))
    if spec.gate_mode == "none":
        backend.gate_mode = "none"

    runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
    assert runs_root is not None
    run_dir = Path(out_dir) if out_dir else runs_root / cfg.campaign_id / "train"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = cfg.section("runtime")
    budgets = BudgetSpec(
        max_actions=int(runtime_cfg.get("max_actions", 20)),
        max_extra_llm_calls=int(runtime_cfg.get("max_extra_llm_calls", 2)),
    )
    from rsi.contracts.policy import SamplingParams

    sampling = SamplingParams(max_new_tokens=int(runtime_cfg.get("max_new_tokens", 256)))
    from rsi.environments.registry import global_registry

    registry = global_registry()
    domains = list(cfg.section("suite").get("domains") or [])
    max_steps = int(cfg.section("training").get("max_optimizer_steps_per_round", 2))
    batch_episodes = int(cfg.section("training").get("batch_episodes", 4))

    step_reports: list[JointStepReport] = []
    reward_trace: list[float] = []
    global_step = 0
    for step in range(max_steps):
        # One batch per optimizer step (default); rollouts refresh each batch.
        all_rl = []
        all_opd = []
        for domain in domains:
            bundle = registry.get(domain)
            provider = bundle.task_provider_factory()
            tasks = provider.public_tasks("work_train")[:batch_episodes]
            batch = collect_rollouts(
                policy=policy,
                teacher_policy=teacher_policy,
                env_factory=bundle.adapter_factory,
                scorer_factory=bundle.scorer_factory,
                task_provider=provider,
                student_bank=student_bank,
                teacher_bank=bank,
                tasks=tasks,
                budgets=budgets,
                sampling=sampling,
                collect_opd=spec.use_opd,
                uncertain_realizability=str(cfg.section("opd").get("uncertain_realizability", "skip")),
                max_targets_per_decision=int(cfg.section("opd").get("max_targets_per_decision", 1)),
                max_shadow_events=int(cfg.section("opd").get("max_shadow_events", 8)),
                seed=cfg.seed + 1000 * step,
                campaign_id=cfg.campaign_id,
                batch_index=step,
            )
            all_rl.extend(batch.rl_rows)
            all_opd.extend(batch.opd_rows)
            reward_trace.extend(batch.rewards)
        report = joint_update(
            backend,
            rl_rows=all_rl,
            opd_rows=all_opd,
            spec=spec,
            lambda_opd=float(cfg.section("opd").get("lambda", 0.01)),
            world_size=getattr(backend, "world_size", 1),
        )
        step_reports.append(report)
        global_step += 1

    identity = CheckpointIdentity(
        policy_digest=policy.policy_artifact.policy_id,
        bank_digest=bank.bank_digest,
        runtime_contract_digest="",
        environment_digest="",
        data_digest=cfg.digest(),
        global_step=global_step,
        batch_identity=f"steps={global_step}",
        method=spec.method,
    )
    ckpt_dir = save_checkpoint(run_dir / f"step_{global_step:04d}", backend=backend, identity=identity)
    try:
        exported = export_adapter(backend, run_dir / "export", parent_policy_id=policy.policy_artifact.policy_id)
        export_ok = True
        export_error = None
    except Exception as e:  # noqa: BLE001 - fsdp2 export path may be unavailable
        export_ok = False
        export_error = repr(e)
        exported = {}

    payload = {
        "ok": export_ok,
        "command": "train",
        "method": spec.method,
        "backend": backend.name,
        "bank_ref": bank_ref,
        "bank_digest": bank.bank_digest,
        "policy_digest_in": policy.policy_artifact.policy_id,
        "global_steps": global_step,
        "step_reports": [r.to_json() for r in step_reports],
        "reward_trace": reward_trace,
        "checkpoint": str(ckpt_dir),
        "export": exported,
        "export_error": export_error,
    }
    out_path = run_dir / "train_result.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    payload["result_path"] = str(out_path)
    return payload
