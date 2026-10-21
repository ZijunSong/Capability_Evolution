"""Tiny-model checks for action-position CE and the global weighted update."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow
from trim.training.sft_engine.accelerate_engine import update_global_batch
from trim.training.sft_engine.action_loss import action_ce_numerator, full_logits_numerator
from trim.training.sft_engine.planner import plan_slots


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(40, 8)
        self.out = nn.Linear(8, 40, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_router_logits=False, logits_to_keep=None):
        del attention_mask, use_cache, output_router_logits
        hidden = self.out(self.emb(input_ids))
        if logits_to_keep is None:
            return type("Out", (), {"logits": hidden})()
        if not torch.is_tensor(logits_to_keep):
            raise RuntimeError("int logits_to_keep is not the action-position contract")
        selected = hidden.index_select(1, logits_to_keep.to(hidden.device).long())
        return type("Out", (), {"logits": selected})()


def _row(row_id: str, prompt: list[int], response: list[int], mask: list[bool], weight: float) -> SFTBatchRow:
    profile = PROFILES["teacher_sft"]
    return SFTBatchRow(
        row_id=row_id,
        experiment="teacher_sft",
        query_id="q",
        episode_id="e",
        decision_id=row_id,
        prompt_ids=prompt,
        response_ids=response,
        response_mask=mask,
        weight=weight,
        policy_version="p",
        prompt_owner=profile.prompt_owner,
        target_source=profile.supervision_provider,
    )


class _Acc:
    def __init__(self, world: int, rank: int) -> None:
        self.device = torch.device("cpu")
        self.num_processes = world
        self.process_index = rank
        self.gradient_accumulation_steps = 1

    def no_sync(self, model):
        del model
        from contextlib import nullcontext

        return nullcontext()

    def autocast(self):
        from contextlib import nullcontext

        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


def test_action_logits_match_the_full_logit_oracle():
    torch.manual_seed(0)
    left = Tiny()
    right = Tiny()
    right.load_state_dict(copy.deepcopy(left.state_dict()))
    row = _row("a", [1, 2, 3], [4, 5, 6], [True, False, True], 1.5)
    left_num = action_ce_numerator(left, row, device=torch.device("cpu"))
    right_num = full_logits_numerator(right, row, device=torch.device("cpu"), max_length=32)
    assert torch.allclose(left_num, right_num)
    left_num.backward()
    right_num.backward()
    for lhs, rhs in zip(left.parameters(), right.parameters()):
        assert torch.allclose(lhs.grad, rhs.grad)


def test_manual_ddp_mean_matches_one_process_update():
    rows = [
        _row("a", [1, 2], [3, 4], [True, True], 1.0),
        _row("b", [5], [6, 7, 8], [True, False, True], 2.0),
        _row("c", [9, 1], [2], [True], 0.5),
    ]
    torch.manual_seed(0)
    solo = Tiny()
    world = 2
    parts = []
    for _rank in range(world):
        model = Tiny()
        model.load_state_dict(copy.deepcopy(solo.state_dict()))
        parts.append(model)
    opt = torch.optim.SGD(solo.parameters(), lr=1e-2)
    part_opts = [torch.optim.SGD(model.parameters(), lr=1e-2) for model in parts]
    one = update_global_batch(_Acc(1, 0), solo, opt, None, rows, 32, grad_clip=1.0e9)
    local_grads = []
    for rank, model in enumerate(parts):
        slots = plan_slots(rows, world)[rank]
        model.zero_grad(set_to_none=True)
        mass = sum(rows[index].weight * sum(rows[index].response_mask) for index in slots if index is not None)
        z = sum(row.weight * sum(row.response_mask) for row in rows)
        reference = min(rows, key=lambda row: len(row.prompt_ids) + len(row.response_ids))
        for index in slots:
            dummy = index is None
            row = reference if dummy else rows[index]
            numerator = action_ce_numerator(model, row, device=torch.device("cpu"), dummy=dummy)
            (numerator * (world / z)).backward()
        local_grads.append([param.grad.detach().clone() for param in model.parameters()])
        assert mass >= 0
    for param, grads in zip(parts[0].parameters(), zip(*local_grads)):
        param.grad = sum(grads) / world
    part_opts[0].step()
    for lhs, rhs in zip(solo.parameters(), parts[0].parameters()):
        assert torch.allclose(lhs, rhs, atol=1e-6)
    assert one["updated"] is True
    assert one["n_rows"] == 3
    assert one["dummy_rows"] == 0
