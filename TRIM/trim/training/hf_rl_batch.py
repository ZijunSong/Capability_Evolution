"""Packed teacher-forced batches and per-step HF train sampling.

HF debug training used to run one 20B forward/backward per RL/OPD datum.
These helpers keep the same per-token math while letting a micro-batch share
one padded forward. Group sampling is an explicit setting: 0 means replay the
full query pool (legacy); a positive cap is a new job setting.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch

HF_DEFAULT_GROUPS_PER_STEP = 32
HF_DEFAULT_MICRO_BATCH = 4
HF_DEFAULT_HEARTBEAT_EVERY = 8
HF_LENGTH_BUCKET = 64
HF_MAX_FULL_TOKENS = 8192


def log_train(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print("[train] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def sample_groups_for_step(
    groups: Sequence[Any],
    n_groups: int | None,
    *,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Sample whole query groups so CISPO group-relative advantages stay valid.

    ``n_groups <= 0`` keeps the full pool (legacy all-replay). Query-id
    de-duplication of the official train pool is unchanged: we only subsample
    which groups enter this optimizer step.
    """
    pool = list(groups)
    want = 0 if n_groups is None else int(n_groups)
    if want <= 0 or want >= len(pool):
        ids = [str(getattr(g, "query_id", i)) for i, g in enumerate(pool)]
        return pool, {
            "sampled": False,
            "n_groups": len(pool),
            "n_pool": len(pool),
            "query_ids": ids[:16],
            "n_query_ids_omitted": max(0, len(ids) - 16),
            "seed": int(seed),
        }
    rng = random.Random(int(seed))
    order = list(range(len(pool)))
    rng.shuffle(order)
    pick = sorted(order[:want])
    chosen = [pool[i] for i in pick]
    ids = [str(getattr(g, "query_id", i)) for i, g in enumerate(chosen)]
    return chosen, {
        "sampled": True,
        "n_groups": len(chosen),
        "n_pool": len(pool),
        "query_ids": ids,
        "n_query_ids_omitted": 0,
        "seed": int(seed),
    }


def iter_length_microbatches(
    items: Sequence[Any],
    *,
    size: int,
    length_fn: Callable[[Any], int],
    bucket: int = HF_LENGTH_BUCKET,
) -> Iterator[list[Any]]:
    seq = list(items)
    mb = max(1, int(size))
    if mb == 1 or len(seq) <= 1:
        for item in seq:
            yield [item]
        return
    width = max(1, int(bucket))
    buckets: dict[int, list[Any]] = {}
    for item in seq:
        key = (max(1, int(length_fn(item))) + width - 1) // width
        buckets.setdefault(key, []).append(item)
    for key in sorted(buckets):
        chunk = buckets[key]
        for i in range(0, len(chunk), mb):
            yield chunk[i : i + mb]


def truncate_teacher_forced_pair(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    max_full: int = HF_MAX_FULL_TOKENS,
    keep_prefix: int = 512,
) -> tuple[list[int], list[int]]:
    prompt = list(prompt_ids)
    response = list(response_ids)
    full_len = len(prompt) + len(response)
    if full_len <= max_full or not response:
        return prompt, response
    overflow = full_len - max_full
    if overflow < len(prompt):
        budget_prompt = len(prompt) - overflow
        prefix = min(max(0, int(keep_prefix)), max(0, budget_prompt // 3))
        tail = budget_prompt - prefix
        if prefix > 0 and tail > 0:
            prompt = prompt[:prefix] + prompt[-tail:]
        else:
            prompt = prompt[overflow:]
    else:
        keep = max(1, max_full - 1)
        response = response[-keep:]
        prompt = prompt[-1:] if prompt else [0]
    return prompt, response


@dataclass
class PackedTeacherForced:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_ids: list[list[int]]
    resp_lens: list[int]
    max_resp: int


def pack_left_pad_teacher_forced(
    pairs: Sequence[tuple[list[int], list[int]]],
    *,
    pad_id: int,
    device: torch.device | str,
    max_full: int = HF_MAX_FULL_TOKENS,
) -> PackedTeacherForced:
    truncated: list[tuple[list[int], list[int]]] = []
    for prompt, response in pairs:
        truncated.append(truncate_teacher_forced_pair(prompt, response, max_full=max_full))
    max_len = max((len(p) + len(r) for p, r in truncated), default=1)
    max_resp = max((len(r) for _, r in truncated), default=0)
    batch = len(truncated)
    input_ids = torch.full((batch, max_len), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    resp_ids: list[list[int]] = []
    resp_lens: list[int] = []
    for i, (prompt, response) in enumerate(truncated):
        full = prompt + response
        start = max_len - len(full)
        if full:
            input_ids[i, start:] = torch.tensor(full, dtype=torch.long, device=device)
            attention[i, start:] = 1
        resp_ids.append(list(response))
        resp_lens.append(len(response))
    return PackedTeacherForced(
        input_ids=input_ids,
        attention_mask=attention,
        response_ids=resp_ids,
        resp_lens=resp_lens,
        max_resp=int(max_resp),
    )


# Backward recomputes softmax over vocab chunks so the FP32 log-softmax tensor
# is never materialized. The value is exact: logit[token] - logsumexp(logits).
_LOGPROB_VOCAB_CHUNK = 4096


class _SelectedTokenLogprob(torch.autograd.Function):
    """log p(token) = logit[token] - logsumexp(logits), with chunked backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_index: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"expected [N, V] logits, got {tuple(logits.shape)}")
        index = token_index.reshape(-1).to(device=logits.device, dtype=torch.long)
        logits_f = logits.float()
        gathered = logits_f.gather(1, index.view(-1, 1)).squeeze(1)
        lse = torch.logsumexp(logits_f, dim=-1)
        ctx.save_for_backward(logits, index)
        return gathered - lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, token_index = ctx.saved_tensors
        grad_out = grad_output.reshape(-1).float().unsqueeze(1)
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1, keepdim=True)
        grad = torch.empty_like(logits_f)
        vocab = int(logits_f.shape[-1])
        chunk = max(1, int(_LOGPROB_VOCAB_CHUNK))
        for start in range(0, vocab, chunk):
            end = min(vocab, start + chunk)
            prob = torch.exp(logits_f[:, start:end] - lse)
            grad[:, start:end] = -prob * grad_out
        grad.scatter_add_(1, token_index.view(-1, 1), grad_out)
        if grad.dtype != logits.dtype:
            grad = grad.to(dtype=logits.dtype)
        return grad, None


def gather_response_logprobs(
    kept_logits: torch.Tensor,
    response_ids: Sequence[Sequence[int]],
    *,
    max_resp: int,
) -> list[torch.Tensor]:
    """Per-token logprob of the sampled ids. Exact logsumexp, not a top-k denominator.

    ``logits_to_keep = max_resp + 1`` still drops the prompt logits upstream.
    This function does not allocate an FP32 log-softmax over the vocabulary.
    """
    if kept_logits.ndim != 3:
        raise ValueError(f"expected [B, T, V] logits, got {tuple(kept_logits.shape)}")
    window = kept_logits[:, :-1, :] if kept_logits.shape[1] > 1 else kept_logits
    batch, width, vocab = window.shape
    cap = min(int(max_resp), int(width))
    empty = torch.zeros(0, dtype=torch.float32, device=window.device)
    if width <= 0 or vocab <= 0 or batch <= 0:
        return [empty for _ in response_ids]
    index = torch.zeros(batch, width, dtype=torch.long, device=window.device)
    spans: list[tuple[int, int]] = []
    for i, resp in enumerate(response_ids):
        n_resp = len(resp)
        if n_resp <= 0:
            spans.append((0, 0))
            continue
        start = max(0, cap - n_resp)
        row_len = min(n_resp, width - start)
        ids = [int(tok) for tok in list(resp)[:row_len]]
        row_len = min(row_len, len(ids))
        if row_len > 0:
            index[i, start : start + row_len] = torch.tensor(ids, dtype=torch.long, device=window.device)
        spans.append((start, row_len))
    flat = _SelectedTokenLogprob.apply(window.reshape(batch * width, vocab), index.reshape(batch * width))
    logp = flat.view(batch, width)
    out: list[torch.Tensor] = []
    for row, (start, row_len) in enumerate(spans):
        if row_len <= 0:
            out.append(logp.new_zeros(0))
        else:
            out.append(logp[row, start : start + row_len])
    return out
