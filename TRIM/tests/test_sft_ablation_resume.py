from __future__ import annotations

import copy
import random

import pytest
import torch
from torch import nn

from trim.training.sft_ablation.checkpoint import load_committed, publish_checkpoint
from trim.training.sft_ablation.contracts import profile_record, require_same_identity


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def _identity(experiment: str = "teacher_sft") -> dict:
    base = profile_record(experiment)
    base.update({key: "x" for key in (
        "weight_policy", "weight_normalization", "base_checkpoint_digest", "init_adapter_digest",
        "teacher_policy_digest", "lora_target", "dtype", "query_order_digest", "train_split",
        "corpus_digest", "retrieval_index_digest", "teacher_archive_digest", "train_env",
        "teacher_mask_digest", "student_mask_digest", "tool_schema_digest", "harness_source_digest",
        "verifier_config_version", "tokenizer_digest", "model_family", "prompt_template_digest",
        "optimizer", "scheduler", "projector_digest", "state_sampling_rule", "branch_selection",
        "empty_supervision_policy", "noop_policy",
    )})
    base.update({
        "lora_rank": 4, "lora_alpha": 4, "lora_dropout": 0.0, "max_model_len": 32,
        "max_new_tokens": 8, "action_only_supervision": True, "batch_size": 2,
        "micro_batch_size": 1, "learning_rate": 1e-3, "weight_decay": 0.0, "seed": 1,
        "max_anchor_scan_events": 8, "group_size": 1, "tensor_parallel_size": 1,
        "rollout_replicas": 1, "nnodes": 1, "nproc_per_node": 1, "gradient_ranks": 1,
        "loss": "action_ce_global_token_mean_v1",
    })
    from trim.training.sft_ablation.contracts import immutable_identity

    return immutable_identity(base)


def test_resume_restores_weights_optimizer_sampler_and_rng(tmp_path):
    torch.manual_seed(2)
    random.seed(2)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)
    opt.zero_grad()
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    opt.step()
    sched.step()
    random.random()
    saved_weights = copy.deepcopy(model.state_dict())
    saved_opt = copy.deepcopy(opt.state_dict())
    identity = _identity()
    directory = publish_checkpoint(
        tmp_path / "step-000001",
        model=model,
        optimizer=opt,
        scheduler=sched,
        identity=identity,
        optimizer_step=1,
        next_round_id=4,
        policy_version="teacher_sft:step1",
        adapter_digest="d",
        source_digest="s",
        sampler={"cursor": 9, "uses": {"r": 1}},
    )
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1)
    fresh = Tiny()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda _step: 1.0)
    manifest = load_committed(directory, model=fresh, optimizer=fresh_opt, scheduler=fresh_sched, identity=identity)
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, saved_weights[key])
    assert fresh_opt.state_dict()["state"].keys() == saved_opt["state"].keys()
    assert manifest["optimizer_step"] == 1
    assert manifest["next_round_id"] == 4
    assert manifest["sampler"]["cursor"] == 9


def test_uncommitted_checkpoint_and_cross_experiment_are_refused(tmp_path):
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    identity = _identity()
    directory = publish_checkpoint(
        tmp_path / "step-000001",
        model=model,
        optimizer=opt,
        scheduler=None,
        identity=identity,
        optimizer_step=1,
        next_round_id=2,
        policy_version="v",
        adapter_digest="d",
        source_digest="s",
        sampler={"cursor": 0},
    )
    (directory / "COMMITTED.json").unlink()
    with pytest.raises(ValueError, match="uncommitted"):
        load_committed(directory, model=model, optimizer=opt, scheduler=None, identity=identity)
    with pytest.raises(ValueError, match="resume refused"):
        require_same_identity(identity, _identity("student_onpolicy_sft"))
