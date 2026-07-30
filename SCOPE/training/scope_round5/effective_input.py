"""Build student-visible effective inputs for Round 5 observability audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from training.scope.compact_target import compact_target_from_sample
from training.scope.prompting import format_operation_prompt_from_sample


@dataclass(frozen=True)
class EffectiveInputRecord:
    sample_id: str
    query_id: str
    gold_operation: str
    raw_decision_state: dict[str, Any]
    rendered_prompt: str
    effective_prompt: str
    prompt_sha256: str
    effective_token_ids: tuple[int, ...]
    token_length_before: int
    token_length_after: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_id": self.query_id,
            "gold_operation": self.gold_operation,
            "raw_decision_state": self.raw_decision_state,
            "rendered_prompt": self.rendered_prompt,
            "effective_prompt": self.effective_prompt,
            "prompt_sha256": self.prompt_sha256,
            "effective_token_ids": list(self.effective_token_ids),
            "token_length_before": self.token_length_before,
            "token_length_after": self.token_length_after,
            "truncated": self.truncated,
        }


def _state_text(sample: dict[str, Any]) -> str:
    return str(
        sample.get("student_state_text")
        or (sample.get("decision_state") or {}).get("rendered_context")
        or ""
    )


def _query_id(sample: dict[str, Any]) -> str:
    ds = sample.get("decision_state") or {}
    return str(
        sample.get("task_id")
        or ds.get("task_id")
        or ds.get("episode_id")
        or sample.get("episode_id")
        or sample.get("sample_id")
        or ""
    )


def build_effective_input(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 4096,
) -> EffectiveInputRecord:
    """Mirror training path: state_text + candidate context → tokenize → truncate."""
    rendered = format_operation_prompt_from_sample(sample)
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    before = len(token_ids)
    truncated = before > max_length
    eff_ids = tuple(token_ids[:max_length] if truncated else token_ids)
    after = len(eff_ids)
    eff_prompt = tokenizer.decode(list(eff_ids), skip_special_tokens=False)
    sha = hashlib.sha256(",".join(str(i) for i in eff_ids).encode()).hexdigest()

    ct = compact_target_from_sample(sample)
    gold = ct.operation.value if ct else str(
        (sample.get("target_action") or {}).get("operation", "")
    )

    return EffectiveInputRecord(
        sample_id=str(sample.get("sample_id") or sample.get("event_id") or ""),
        query_id=_query_id(sample),
        gold_operation=gold,
        raw_decision_state=sample.get("decision_state") or {},
        rendered_prompt=rendered,
        effective_prompt=eff_prompt,
        prompt_sha256=sha,
        effective_token_ids=eff_ids,
        token_length_before=before,
        token_length_after=after,
        truncated=truncated,
    )


def dump_effective_inputs(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    out_path: Path,
    *,
    max_length: int = 4096,
) -> list[EffectiveInputRecord]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[EffectiveInputRecord] = []
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            rec = build_effective_input(sample, tokenizer, max_length=max_length)
            records.append(rec)
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return records
