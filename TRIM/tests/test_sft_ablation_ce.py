from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from trim.training.sft_ablation.ce import ce_update, weighted_causal_ce_parts
from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(40, 8)
        self.out = nn.Linear(8, 40)

    def forward(self, input_ids, attention_mask=None):
        return self.out(self.emb(input_ids))


def _row(row_id: str, prompt: list[int], response: list[int], weight: float = 1.0) -> SFTBatchRow:
    profile = PROFILES["teacher_sft"]
    return SFTBatchRow(
        row_id=row_id,
        experiment="teacher_sft",
        query_id="q",
        episode_id="e",
        decision_id=row_id,
        prompt_ids=prompt,
        response_ids=response,
        response_mask=[True] * len(response),
        weight=weight,
        policy_version="p",
        prompt_owner=profile.prompt_owner,
        target_source=profile.supervision_provider,
    )


def _pair():
    torch.manual_seed(0)
    first = Tiny()
    second = Tiny()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    opt_a = torch.optim.AdamW(first.parameters(), lr=1e-2, weight_decay=0.0)
    opt_b = torch.optim.AdamW(second.parameters(), lr=1e-2, weight_decay=0.0)
    return first, second, opt_a, opt_b


def test_microbatches_match_one_batch_and_formula_is_global():
    rows = [_row("a", [1, 2], [3, 4], weight=1.0), _row("b", [5], [6, 7, 8], weight=2.0)]
    left, right, opt_l, opt_r = _pair()
    sched_l = torch.optim.lr_scheduler.LambdaLR(opt_l, lambda _step: 1.0)
    sched_r = torch.optim.lr_scheduler.LambdaLR(opt_r, lambda _step: 1.0)
    one = ce_update(left, opt_l, sched_l, rows, pad_id=0, max_length=32, micro_batch_size=8)
    many = ce_update(right, opt_r, sched_r, rows, pad_id=0, max_length=32, micro_batch_size=1)
    assert one["updated"] and many["updated"]
    assert one["loss_ce"] == pytest.approx(many["loss_ce"])
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.allclose(a, b)

    labels = torch.tensor([[-100, 3, 4]])
    weights = torch.tensor([[0.0, 1.0, 2.0]])
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 3] = 2.0
    logits[0, 1, 4] = 1.0
    numerator, denominator = weighted_causal_ce_parts(logits, labels, weights)
    manual = torch.nn.functional.cross_entropy(logits[:, :-1, :].transpose(1, 2), labels[:, 1:], reduction="none")
    shifted = weights[:, 1:]
    assert torch.allclose(numerator, (manual * shifted).sum())
    assert float(denominator) == pytest.approx(3.0)


def test_empty_batch_does_not_step_adamw():
    model = Tiny()
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    result = ce_update(model, opt, None, [], pad_id=0, max_length=8, micro_batch_size=2)
    assert result["updated"] is False
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert opt.state == {}


def test_ce_module_has_no_rl_or_gap_route():
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("trim/training/sft_ablation/ce.py").read_text(encoding="utf-8")
    assert "gated_action_gap" not in text
    assert "hybrid_train_substep" not in text
    assert "advantage" not in text
