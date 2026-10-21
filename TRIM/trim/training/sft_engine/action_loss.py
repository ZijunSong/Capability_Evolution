"""Action-position CE numerator.

``logits[t]`` predicts ``input_ids[t + 1]``. The first supervised action
token is predicted from the last prompt position. Positions that the
response mask drops are not scored and do not enter the numerator.
"""

from __future__ import annotations

from typing import Any, Sequence


def supervised_positions(prompt_len: int, response_mask: Sequence[bool]) -> list[int]:
    kept = [index for index, keep in enumerate(response_mask) if keep]
    if not kept or prompt_len < 1:
        raise ValueError("Invalid action row")
    return [prompt_len - 1 + index for index in kept]


def action_ce_numerator(model: Any, row: Any, *, device: Any, dummy: bool = False) -> Any:
    """Weighted sum of token CE at supervised action positions.

    A dummy slot uses the same graph and then multiplies by zero. It must
    not be represented by forging a zero-weight row.
    """
    import torch
    import torch.nn.functional as F

    kept = [index for index, keep in enumerate(row.response_mask) if keep]
    if not kept or not row.prompt_ids:
        raise ValueError("Invalid action row")
    positions = torch.tensor(
        [len(row.prompt_ids) - 1 + index for index in kept],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [int(row.response_ids[index]) for index in kept],
        dtype=torch.long,
        device=device,
    )
    inputs = torch.tensor([list(row.prompt_ids) + list(row.response_ids)], dtype=torch.long, device=device)
    outputs = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        use_cache=False,
        output_router_logits=False,
        logits_to_keep=positions,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if isinstance(logits, tuple):
        logits = logits[0]
    if tuple(logits.shape[:2]) != (1, len(kept)):
        raise RuntimeError(
            "Model did not honor action-position logits selection "
            f"(got {tuple(logits.shape)}, expected {(1, len(kept), 'vocab')})"
        )
    per_token = F.cross_entropy(logits[0].float(), targets, reduction="none")
    factor = 0.0 if dummy else float(row.weight)
    return per_token.sum() * factor


def full_logits_numerator(model: Any, row: Any, *, device: Any, max_length: int) -> Any:
    """Oracle: full-sequence logits, then the existing weighted CE. Test only."""
    import torch

    from trim.training.sft_ablation.ce import weighted_causal_ce_parts
    from trim.training.sft_ablation.encoding import collate_ce_rows

    batch = collate_ce_rows([row], pad_id=0, max_length=max_length)
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
    attention = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
    labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
    weights = torch.tensor(batch["token_weights"], dtype=torch.float32, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False, output_router_logits=False)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if isinstance(logits, tuple):
        logits = logits[0]
    numerator, _denominator = weighted_causal_ce_parts(logits, labels, weights)
    return numerator
