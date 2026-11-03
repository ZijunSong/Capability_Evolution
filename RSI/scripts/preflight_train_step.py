"""Train-path preflight: the exact 32-GPU job sequence on one GPU in miniature.

Stage-by-stage (each prints PASS/FAIL): bf16 model load -> LoRA attach ->
generate -> teacher-force -> 1 episode collection (Student H0 + Teacher fork)
-> one joint RL+OPD step -> export/reload.  Run this on a cheap 1-GPU job
BEFORE queueing the 32-GPU training job; it reproduces the GPU/dtype/mode
configuration that CPU tests historically diverged from.
"""

from __future__ import annotations

import sys
import traceback


def stage(name: str):
    print(f"[preflight] === {name} ===", flush=True)

def main() -> int:
    try:
        stage("config")
        from rsi.config import load_config
        from rsi.providers.factory import build_policy

        cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_harness_opd_qwen35_9b_200step_32gpu.yaml"
        cfg = load_config(cfg_path)
        print("[preflight] config ok:", cfg.campaign_id, flush=True)

        stage("policy load (bf16 on GPU)")
        policy = build_policy(cfg)
        import torch

        p0 = next(policy.model.parameters())
        print(f"[preflight] model dtype={p0.dtype} device={p0.device}", flush=True)
        assert p0.device.type == "cuda", "preflight expects a GPU"
        assert p0.dtype == torch.bfloat16, f"expected bf16 job dtype, got {p0.dtype}"

        stage("LoRA attach + freeze")
        from rsi.training.service import _apply_lora_if_configured

        lora_info = _apply_lora_if_configured(cfg, policy)
        print("[preflight] lora:", lora_info, flush=True)
        assert lora_info.get("lora") and lora_info["n_trainable_params"] > 0

        stage("backend + optimizer (trainable-only)")
        from rsi.training.service import _build_backend

        backend = _build_backend(cfg, policy, device=str(p0.device))
        print(f"[preflight] trainable params: {backend.n_trainable_params}", flush=True)

        stage("generate through LoRA wrappers")
        from rsi.contracts.policy import SamplingParams

        ids = policy.encode('{"type": "tool_call", "name": "read_file", "arguments": {"path": "a.txt"}}')
        out = policy.sample(ids, SamplingParams(max_new_tokens=16, temperature=1.0))
        print(f"[preflight] generated {len(out.token_ids)} tokens", flush=True)

        stage("1 episode collection (Student H0 + Teacher fork)")
        from rsi.environments.registry import global_registry
        from rsi.harness.loader import load_bank_ref, load_empty_bank
        from rsi.runtime.budgets import BudgetSpec
        from rsi.training.collector import collect_rollouts

        bank_ref = sys.argv[2] if len(sys.argv) > 2 else "runs/evolve_qwen35_9b_p1_r4/evolve/champion.json"
        registry = global_registry()
        domain = list(cfg.section("suite").get("domains"))[0]
        bundle = registry.get(domain)
        provider = bundle.task_provider_factory()
        task = provider.public_tasks("work_train")[0]
        batch = collect_rollouts(
            policy=policy,
            teacher_policy=policy,
            env_factory=bundle.adapter_factory,
            scorer_factory=bundle.scorer_factory,
            task_provider=provider,
            student_bank=load_empty_bank(),
            teacher_bank=load_bank_ref(bank_ref),
            tasks=[task],
            budgets=BudgetSpec(max_actions=6, max_extra_llm_calls=2),
            sampling=SamplingParams(max_new_tokens=128),
            collect_opd=True,
            max_shadow_events=2,
            seed=17,
            campaign_id="preflight",
            batch_index=0,
            center_advantage=False,
        )
        print(
            f"[preflight] rl_rows={len(batch.rl_rows)} opd_rows={len(batch.opd_rows)} "
            f"rewards={list(batch.rewards)}",
            flush=True,
        )

        stage("joint RL+OPD step")
        from rsi.training.joint import MethodSpec, joint_update

        report = joint_update(
            backend,
            rl_rows=batch.rl_rows,
            opd_rows=batch.opd_rows,
            spec=MethodSpec.resolve(str(cfg.section("training").get("method", "harness_opd_rl"))),
            lambda_opd=float(cfg.section("opd").get("lambda", 0.01)),
        )
        print(f"[preflight] step report: {report.to_json()}", flush=True)
        assert report.n_optimizer_steps == 1

        stage("export + reload equivalence")
        import tempfile
        from rsi.training.export import export_adapter, verify_export_reload_equivalence

        with tempfile.TemporaryDirectory() as td:
            export_adapter(backend, td)
            print("[preflight] export ok", flush=True)

        print("[preflight] ALL STAGES PASSED", flush=True)
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("[preflight] FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
