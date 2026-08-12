"""Single-source effective input builder for rollback training/eval/live replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from training.scope.checkpoint_candidates import (
    assign_local_checkpoint_ids,
    global_to_local_id,
    summarize_candidate,
)
from training.scope.rollback_operation_objectives import format_rollback_operation_prompt


@dataclass(frozen=True)
class RollbackEffectiveInput:
    query_id: str
    turn: int
    state_source: str
    effective_input_text: str
    prompt_sha256: str
    token_ids_sha256: str
    candidate_list: list[dict[str, Any]]
    candidate_list_sha256: str
    gold_operation: str
    gold_checkpoint_local_id: str | None
    gold_checkpoint_global_id: str | None
    gold_in_candidates: bool
    truncated: bool
    token_length_before: int
    token_length_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "turn": self.turn,
            "state_source": self.state_source,
            "effective_input_text": self.effective_input_text,
            "prompt_sha256": self.prompt_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "candidate_list": self.candidate_list,
            "candidate_list_sha256": self.candidate_list_sha256,
            "gold_operation": self.gold_operation,
            "gold_checkpoint_local_id": self.gold_checkpoint_local_id,
            "gold_checkpoint_global_id": self.gold_checkpoint_global_id,
            "gold_in_candidates": self.gold_in_candidates,
            "truncated": self.truncated,
            "token_length_before": self.token_length_before,
            "token_length_after": self.token_length_after,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_token_ids(token_ids: list[int] | tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(str(i) for i in token_ids).encode()).hexdigest()


def _state_text(sample: dict[str, Any]) -> str:
    ds = sample.get("decision_state") or {}
    return str(
        sample.get("student_state_text")
        or ds.get("rendered_context")
        or json.dumps(ds, ensure_ascii=False)
    )


def _gold_operation(sample: dict[str, Any]) -> str:
    ta = sample.get("target_action") or {}
    return str(ta.get("operation") or sample.get("operation") or "CONTINUE")


def _gold_checkpoint(sample: dict[str, Any]) -> str | None:
    ta = sample.get("target_action") or {}
    ck = ta.get("checkpoint_id") or sample.get("checkpoint_id")
    return str(ck) if ck else None


def build_rollback_effective_input(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    state_source: str = "offline_valid",
    hint: str = "",
    max_length: int = 8100,
    include_candidate_summaries: bool = True,
) -> RollbackEffectiveInput:
    ds = sample.get("decision_state") or {}
    raw_candidates = list(ds.get("available_checkpoints") or sample.get("available_checkpoints") or [])
    ordered, local_to_global = assign_local_checkpoint_ids(raw_candidates)
    candidate_summaries = [summarize_candidate(c) for c in ordered] if include_candidate_summaries else []
    state_text = _state_text(sample)
    if include_candidate_summaries and candidate_summaries:
        state_text = state_text + "\nCandidate checkpoints:\n" + "\n".join(candidate_summaries)

    prompt = format_rollback_operation_prompt(
        state_text,
        available_checkpoints=ordered if not include_candidate_summaries else ordered,
        hint=hint,
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    before = len(token_ids)
    # Keep the suffix so "Recovery operation:" and decision head remain intact.
    # Reserve a few tokens so verbalizer scoring (CONTINUE/REPLAN/ROLLBACK_TO)
    # still fits under common vLLM max_model_len after concat.
    verbalizer_reserve = 8
    prompt_budget = max(max_length - verbalizer_reserve, 1)
    truncated = before > prompt_budget
    eff_ids = token_ids[-prompt_budget:] if truncated else token_ids
    after = len(eff_ids)
    eff_prompt = tokenizer.decode(eff_ids, skip_special_tokens=False)

    gold_op = _gold_operation(sample)
    gold_ck = _gold_checkpoint(sample)
    gold_local = global_to_local_id(gold_ck, local_to_global)
    gold_in = gold_ck is None or gold_ck in set(local_to_global.values())

    candidate_payload = [
        {
            "local_checkpoint_id": c.get("local_checkpoint_id"),
            "checkpoint_id": c.get("checkpoint_id"),
            "summary": summarize_candidate(c),
        }
        for c in ordered
    ]
    cand_sha = _sha256_text(json.dumps(candidate_payload, sort_keys=True, ensure_ascii=False))

    return RollbackEffectiveInput(
        query_id=str(sample.get("query_id") or ds.get("task_id") or ""),
        turn=int(ds.get("turn_id", sample.get("turn_id", 0))),
        state_source=state_source,
        effective_input_text=eff_prompt,
        prompt_sha256=_sha256_text(eff_prompt),
        token_ids_sha256=_sha256_token_ids(eff_ids),
        candidate_list=candidate_payload,
        candidate_list_sha256=cand_sha,
        gold_operation=gold_op,
        gold_checkpoint_local_id=gold_local,
        gold_checkpoint_global_id=gold_ck,
        gold_in_candidates=gold_in,
        truncated=truncated,
        token_length_before=before,
        token_length_after=after,
    )
