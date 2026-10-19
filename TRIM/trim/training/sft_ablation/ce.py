"""Action-only CE. One global denominator per optimizer update."""

from __future__ import annotations

from typing import Any, Sequence

from trim.training.sft_ablation.contracts import LOSS_VERSION, SFTBatchRow, supervised_weight
from trim.training.sft_ablation.encoding import collate_ce_rows


def global_supervised_mass(rows: Sequence[SFTBatchRow]) -> float:
    return float(sum(supervised_weight(row) for row in rows))


def weighted_causal_ce_parts(logits: Any, labels: Any, token_weights: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = token_weights[:, 1:].float()
    valid = shifted_labels.ne(-100)
    shifted_weights = shifted_weights * valid
    per_token = F.cross_entropy(
        logits[:, :-1, :].float().transpose(1, 2),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    numerator = (per_token * shifted_weights).sum()
    denominator = shifted_weights.sum().detach()
    if not torch.isfinite(numerator) or not torch.isfinite(denominator):
        raise FloatingPointError("nonfinite action CE")
    return numerator, denominator


def ce_update(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    rows: Sequence[SFTBatchRow],
    *,
    pad_id: int,
    max_length: int,
    micro_batch_size: int,
    grad_clip: float = 1.0,
    device: Any = None,
) -> dict[str, Any]:
    """One zero_grad, one global-Z accumulation, one clip, one AdamW step.

    Microbatch numerators are divided by the update-wide token mass. They are
    not averaged across microbatches, and the coordinator world size is not
    applied: Scheme A has a single gradient rank.
    """
    import torch

    material = [row for row in rows if supervised_weight(row) > 0]
    mass = global_supervised_mass(material)
    if not material or mass <= 0:
        return {
            "updated": False,
            "loss_ce": None,
            "grad_norm": None,
            "n_rows": 0,
            "weighted_tokens": 0.0,
            "loss_version": LOSS_VERSION,
        }
    if device is None:
        device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    numerator_total = torch.zeros((), device=device)
    width = max(1, int(micro_batch_size))
    for start in range(0, len(material), width):
        chunk = list(material[start : start + width])
        batch = collate_ce_rows(chunk, pad_id=pad_id, max_length=max_length)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        attention = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        weights = torch.tensor(batch["token_weights"], dtype=torch.float32, device=device)
        logits = _forward_logits(model, input_ids, attention)
        numerator, _local_den = weighted_causal_ce_parts(logits, labels, weights)
        (numerator / mass).backward()
        numerator_total = numerator_total + numerator.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.requires_grad],
        float(grad_clip),
    )
    if not torch.isfinite(numerator_total) or not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite action CE or gradient")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss = float((numerator_total / mass).detach().cpu())
    return {
        "updated": True,
        "loss_ce": loss,
        "grad_norm": float(grad_norm.detach().cpu()),
        "n_rows": len(material),
        "weighted_tokens": float(mass),
        "loss_version": LOSS_VERSION,
    }


def _forward_logits(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits
