from __future__ import annotations

import pytest

from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow
from trim.training.sft_ablation.encoding import collate_ce_rows, require_visible_action


def _row(prompt, response, mask, weight=1.0):
    return SFTBatchRow(
        row_id="r",
        experiment="teacher_sft",
        query_id="q",
        episode_id="e",
        decision_id="d",
        prompt_ids=list(prompt),
        response_ids=list(response),
        response_mask=list(mask),
        weight=weight,
        policy_version="p",
        prompt_owner=PROFILES["teacher_sft"].prompt_owner,
        target_source=PROFILES["teacher_sft"].supervision_provider,
    )


def test_prompt_and_padding_are_not_supervised_and_first_action_token_is_kept():
    short = _row([1, 2], [3, 4], [True, False])
    long = _row([9], [8, 7], [True, True])
    batch = collate_ce_rows([short, long], pad_id=0, max_length=16)
    assert batch["labels"][0][:2] == [-100, -100]
    assert batch["labels"][0][2] == 3
    assert batch["labels"][0][3] == -100
    assert batch["token_weights"][0][:2] == [0.0, 0.0]
    assert batch["token_weights"][0][2] == 1.0
    assert batch["labels"][1][0] == -100
    assert batch["labels"][1][1:] == [8, 7, -100]
    assert batch["attention_mask"][1][-1] == 0
    assert batch["token_weights"][1][-1] == 0.0


def test_overlong_row_is_rejected_not_truncated():
    row = _row([1, 2, 3], [4], [True])
    with pytest.raises(ValueError, match="context budget"):
        collate_ce_rows([row], pad_id=0, max_length=3)


def test_unknown_visibility_is_rejected():
    class Enc:
        def encode(self, text):
            return []

    with pytest.raises(ValueError, match="visibility_unknown"):
        require_visible_action(
            {"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}},
            prompt_ids=[1],
            enc=Enc(),
            metadata={},
            snapshot=None,
        )
