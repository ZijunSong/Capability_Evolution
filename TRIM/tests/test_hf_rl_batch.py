from __future__ import annotations

from types import SimpleNamespace

import torch

from trim.training.hf_rl_batch import (
    gather_response_logprobs,
    iter_length_microbatches,
    pack_left_pad_teacher_forced,
    sample_groups_for_step,
    truncate_teacher_forced_pair,
)


def test_sample_groups_zero_keeps_full_pool():
    groups = [SimpleNamespace(query_id=f"q{i}") for i in range(10)]
    chosen, meta = sample_groups_for_step(groups, 0, seed=1)
    assert len(chosen) == 10
    assert meta["sampled"] is False
    assert meta["n_pool"] == 10


def test_sample_groups_is_reproducible_and_records_ids():
    groups = [SimpleNamespace(query_id=f"q{i}") for i in range(20)]
    a, meta_a = sample_groups_for_step(groups, 5, seed=7)
    b, meta_b = sample_groups_for_step(groups, 5, seed=7)
    c, _ = sample_groups_for_step(groups, 5, seed=8)
    assert [g.query_id for g in a] == [g.query_id for g in b]
    assert meta_a["sampled"] is True
    assert meta_a["query_ids"] == [g.query_id for g in a]
    assert [g.query_id for g in a] != [g.query_id for g in c]
    assert len(a) == 5


def test_length_microbatches_bucket_then_chunk():
    items = [{"n": n} for n in (10, 12, 70, 72, 73)]
    batches = list(
        iter_length_microbatches(items, size=2, length_fn=lambda row: row["n"], bucket=64)
    )
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [row["n"] for row in batches[0]] == [10, 12]


def test_pack_and_gather_left_pad_window():
    pairs = [([1, 2, 3], [10, 11]), ([7], [20, 21, 22])]
    packed = pack_left_pad_teacher_forced(pairs, pad_id=0, device="cpu")
    assert packed.input_ids.shape[0] == 2
    assert packed.max_resp == 3
    logits = torch.zeros(2, 4, 30)
    for b, resp in enumerate(packed.response_ids):
        start = packed.max_resp - len(resp)
        for j, tok in enumerate(resp):
            logits[b, start + j, tok] = 10.0
    logps = gather_response_logprobs(logits, packed.response_ids, max_resp=packed.max_resp)
    assert [t.numel() for t in logps] == [2, 3]
    assert torch.isfinite(logps[0]).all()
    assert torch.isfinite(logps[1]).all()


def test_truncate_pair_keeps_tail_of_prompt():
    prompt, resp = truncate_teacher_forced_pair(list(range(20)), [100, 101], max_full=8)
    assert len(prompt) + len(resp) == 8
    assert resp == [100, 101]
