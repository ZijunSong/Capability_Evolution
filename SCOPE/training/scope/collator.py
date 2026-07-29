"""Collate SDI batches with action-span labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from training.scope.losses import action_span_labels
from training.scope.prompting import format_sdi_example


@dataclass
class SDIBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sample_weights: torch.Tensor
    meta: list[dict[str, Any]]


def collate_sdi_batch(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 4096,
) -> SDIBatch:
    prompts: list[str] = []
    targets: list[str] = []
    weights: list[float] = []
    meta: list[dict[str, Any]] = []

    for ex in examples:
        state_text = str(
            ex.get("student_state_text")
            or (ex.get("decision_state") or {}).get("rendered_context")
            or ""
        )
        target = ex.get("target_action")
        if not target:
            continue
        prompt, target_text = format_sdi_example(state_text, target)
        prompts.append(prompt)
        targets.append(target_text)
        weights.append(float(ex.get("sample_weight", 1.0)))
        meta.append(
            {
                "route": ex.get("route"),
                "capability_id": ex.get("capability_id"),
                "task_id": ex.get("task_id"),
            }
        )

    if not prompts:
        raise ValueError("empty SDI batch")

    full_texts = [p + t for p, t in zip(prompts, targets)]
    enc = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Per-sample action span: tokens after prompt prefix
    action_starts: list[int] = []
    action_ends: list[int] = []
    for i, (prompt, target) in enumerate(zip(prompts, targets)):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        # Find prompt length in encoded full (handle BOS if present)
        full_ids = input_ids[i].tolist()
        # Heuristic: match suffix of prompt+target in truncated sequence
        combined = prompt_ids + target_ids
        start = 0
        for off in range(min(len(full_ids), len(combined))):
            if full_ids[off : off + len(target_ids)] == target_ids[-len(target_ids) :]:
                start = off
                break
        else:
            start = max(0, len(full_ids) - len(target_ids))
        end = min(len(full_ids), start + len(target_ids))
        action_starts.append(start)
        action_ends.append(end)

    labels = action_span_labels(
        input_ids,
        attention_mask,
        action_starts,
        action_ends,
    )
    return SDIBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sample_weights=torch.tensor(weights, dtype=torch.float32),
        meta=meta,
    )
