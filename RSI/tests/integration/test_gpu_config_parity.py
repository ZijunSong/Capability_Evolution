"""GPU-config parity: run the training path in the SAME dtype/mode as the job.

Regression for the 2026-10-29 32-GPU failure: the CPU smoke ran fp32-only while
the job loads the model in **bfloat16** and attaches LoRA — the LoRA matmul
dtype mismatch crashed at the first generate.  These tests execute the real
job sequence (bf16 model → LoRA attach → generate → teacher-force → joint
backward → export/reload) so this class of CPU-vs-GPU divergence is caught
BEFORE queueing a 32-GPU job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from rsi.config import load_config
from rsi.opd.records import OPDRow, RLRow
from rsi.providers.tiny_hf import HFPolicy, build_tiny_model
from rsi.training.hf_reference import HFReferenceBackend
from rsi.training.joint import MethodSpec, joint_update
from rsi.training.lora import LoRAConfig, apply_lora, freeze_non_lora_, lora_state_dict, load_lora_state_dict

pytestmark = pytest.mark.integration

CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "toy_smoke.yaml"


def _bf16_lora_model():
    model = build_tiny_model().to(torch.bfloat16)  # the job's GPU dtype
    replaced = apply_lora(model, LoRAConfig(rank=4, alpha=8))
    assert replaced, "LoRA must attach to the tiny model projections"
    freeze_non_lora_(model)
    return model


def test_lora_forward_bf16_generate_and_backward():
    """bf16 base + LoRA: generate, teacher-force and BOTH backward passes work."""
    model = _bf16_lora_model()
    policy = HFPolicy(model, device="cpu", dtype="bfloat16")

    from rsi.contracts.policy import SamplingParams

    prompt_ids = policy.encode('{"type": "tool_call", "name": "read_file", "arguments": {"path": "a.txt"}}')
    out = policy.sample(prompt_ids, SamplingParams(max_new_tokens=8, temperature=0.0))
    assert isinstance(out.token_ids, tuple)  # generate survived the LoRA wrapper

    lp = policy.teacher_force_logprobs(prompt_ids[:6], prompt_ids[6:10])
    assert len(lp) == 4

    backend = HFReferenceBackend(model=model, lr=1e-3)
    assert backend.n_trainable_params > 0
    rl = RLRow(
        row_id="r", episode_id="e", decision_index=0,
        prompt_token_ids=tuple(prompt_ids[:6]),
        action_token_ids=tuple(prompt_ids[6:10]),
        old_logprobs=(-1.0, -1.0, -1.0, -1.0),
        action_mask=(1, 1, 1, 1),
        advantage=0.5, reward_ref="t", behavior_policy_version="p",
    )
    opd = OPDRow(
        row_id="o", episode_id="e", decision_index=0,
        student_prompt_token_ids=tuple(prompt_ids[:6]),
        teacher_prompt_token_ids=tuple(prompt_ids[:6]),
        target_token_ids=tuple(prompt_ids[6:10]),
        target_mask=(1, 1, 1, 1),
        target_action=None, target_kind="native", alignment_route="native_anchor",
        source_event_ids=(), realizability_json={}, provenance_json={},
        token_weight=(1.0, 1.0, 1.0, 1.0),
        student_logprobs=(-1.0, -1.0, -1.0, -1.0),
        teacher_logprobs=(-0.2, -0.2, -0.2, -0.2),
    )
    report = joint_update(
        backend, rl_rows=[rl], opd_rows=[opd],
        spec=MethodSpec.resolve("harness_opd_rl"), lambda_opd=0.1,
    )
    assert report.n_optimizer_steps == 1

    # only LoRA params moved
    moved = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    assert moved and all("lora_" in n for n in moved), moved

    # export/reload round-trip preserves LoRA state
    sd = lora_state_dict(model)
    assert sd and all(v.dtype == torch.float32 for v in sd.values())
    model2 = _bf16_lora_model()
    load_lora_state_dict(model2, sd)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        if "lora_" in n1:
            assert torch.equal(p1, p2), n1


def test_lora_params_follow_base_device():
    """LoRA params must be created on the base module's device (not CPU)."""
    model = build_tiny_model()
    replaced = apply_lora(model, LoRAConfig(rank=4, alpha=8))
    for wrapper in replaced.values():
        assert wrapper.lora_A.device == wrapper.base.weight.device
        assert wrapper.lora_B.device == wrapper.base.weight.device
        assert wrapper.lora_A.dtype == torch.float32  # optimizer-friendly


def test_episode_spec_sharding_covers_all_ranks():
    """The global batch must be world_size × per_rank so no rank idles."""
    from rsi.training.service import _build_episode_specs
    from rsi.training.distributed import shard_items

    cfg = load_config(CFG_PATH)
    world_size, per_rank = 8, 2
    specs = _build_episode_specs(cfg, step=0, total=per_rank * world_size)
    assert len(specs) == 16
    shards = [shard_items(specs, r, world_size) for r in range(world_size)]
    assert all(len(s) == per_rank for s in shards), [len(s) for s in shards]
    seen = [e["task_id"] for s in shards for e in s]
    assert len(seen) == len(specs)  # disjoint cover, no drops/dupes
    # deterministic across ranks: same step → same list
    assert _build_episode_specs(cfg, step=0, total=16) == specs
