"""Stage-1 prompt views A0–A4 for Round11 state-factorization audit / training."""

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

VIEW_NAMES = ("A0", "A1", "A2", "A3", "A4")


@dataclass(frozen=True)
class Stage1ViewResult:
    view: str
    effective_input_text: str
    prompt_sha256: str
    candidate_list: list[dict[str, Any]]
    gold_operation: str
    gold_checkpoint_local_id: str | None
    gold_checkpoint_global_id: str | None
    gold_in_candidates: bool
    truncated: bool
    token_length_before: int
    token_length_after: int


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_text(sample: dict[str, Any]) -> str:
    ds = sample.get("decision_state") or {}
    return str(
        sample.get("student_state_text")
        or ds.get("rendered_context")
        or json.dumps(ds, ensure_ascii=False)
    )


def _gold_operation(sample: dict[str, Any]) -> str:
    ta = sample.get("target_action") or {}
    return str(ta.get("operation") or sample.get("gold_operation") or sample.get("operation") or "CONTINUE")


def _gold_checkpoint(sample: dict[str, Any]) -> str | None:
    ta = sample.get("target_action") or {}
    ck = ta.get("checkpoint_id") or sample.get("gold_checkpoint_global_id") or sample.get("checkpoint_id")
    return str(ck) if ck else None


def _candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    ds = sample.get("decision_state") or {}
    raw = list(ds.get("available_checkpoints") or sample.get("available_checkpoints") or sample.get("candidate_list") or [])
    # Normalize frozen candidate_list rows into checkpoint dicts.
    normed: list[dict[str, Any]] = []
    for c in raw:
        if "checkpoint_id" not in c and c.get("local_checkpoint_id"):
            continue
        normed.append(
            {
                "checkpoint_id": c.get("checkpoint_id"),
                "turn_id": c.get("turn_id", c.get("relative_turn", 0)),
                "relative_turn": c.get("relative_turn", c.get("turn_id", 0)),
                "n_curated": c.get("n_curated", c.get("evidence_count", 0)),
                "n_pool": c.get("n_pool", 0),
                "n_verified": c.get("n_verified", c.get("verified_count", 0)),
                "evidence_count": c.get("evidence_count", c.get("n_curated", 0)),
                "verified_count": c.get("verified_count", c.get("n_verified", 0)),
                "remaining_recovery_budget": c.get("remaining_recovery_budget", "?"),
                "state_hash": c.get("state_hash"),
                "local_checkpoint_id": c.get("local_checkpoint_id"),
            }
        )
    return normed


def _feasibility_scalars(sample: dict[str, Any], ordered: list[dict[str, Any]]) -> str:
    ds = sample.get("decision_state") or {}
    turn = int(ds.get("turn_id", sample.get("turn", sample.get("turn_id", 0))) or 0)
    budget = ds.get("remaining_recovery_budget", ds.get("remaining_search_budget", "?"))
    latest_age = "?"
    last_success_dist = "?"
    if ordered:
        latest_turn = max(int(c.get("relative_turn", c.get("turn_id", 0)) or 0) for c in ordered)
        latest_age = max(turn - latest_turn, 0)
        # Prefer highest turn strictly before current as "last successful".
        prior = [int(c.get("relative_turn", c.get("turn_id", 0)) or 0) for c in ordered]
        prior = [t for t in prior if t < turn] or prior
        last_success_dist = max(turn - max(prior), 0) if prior else "?"
    lines = [
        "Rollback feasibility:",
        f"checkpoint_count={len(ordered)}",
        f"remaining_rollback_budget={budget}",
        f"current_turn={turn}",
        f"last_successful_checkpoint_distance={last_success_dist}",
        f"has_valid_checkpoint={bool(ordered)}",
        f"latest_checkpoint_age={latest_age}",
    ]
    return "\n".join(lines)


def _failure_progress_scalars(sample: dict[str, Any]) -> str:
    ds = sample.get("decision_state") or {}
    last_type = ds.get("last_action_type", "?")
    # Success/failure proxy from unsupported/supported claim counts + tool errors.
    unsupported = len(ds.get("unsupported_claims") or [])
    supported = len(ds.get("supported_claims") or [])
    tool_err = ds.get("tool_error_summary") or ""
    last_fail = bool(tool_err) or unsupported > supported
    repeated = ds.get("repeated_query_count", ds.get("repeated_failure_count", 0))
    # Progress proxies already present on DecisionState.
    progress = supported
    new_evidence = len(ds.get("evidence_claims") or []) + len(ds.get("curated_document_ids") or [])
    lines = [
        "Failure/progress:",
        f"last_action_type={last_type}",
        f"last_action_success_failure={'failure' if last_fail else 'success'}",
        f"repeated_failure_count={repeated}",
        f"progress_since_checkpoint={progress}",
        f"new_evidence_since_checkpoint={new_evidence}",
    ]
    return "\n".join(lines)


def _truncate(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> tuple[str, bool, int, int]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    before = len(token_ids)
    verbalizer_reserve = 8
    prompt_budget = max(max_length - verbalizer_reserve, 1)
    truncated = before > prompt_budget
    eff_ids = token_ids[-prompt_budget:] if truncated else token_ids
    after = len(eff_ids)
    return tokenizer.decode(eff_ids, skip_special_tokens=False), truncated, before, after


def build_stage1_view(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    view: str,
    *,
    max_length: int = 8100,
    hint: str = "",
) -> Stage1ViewResult:
    if view not in VIEW_NAMES:
        raise ValueError(f"unknown view {view}")
    ordered, local_to_global = assign_local_checkpoint_ids(_candidates(sample))
    state = _state_text(sample)
    summaries = [summarize_candidate(c) for c in ordered]

    if view == "A0":
        # Current full Stage1: state + candidate semantic summaries + ID list.
        body = state
        if summaries:
            body = body + "\nCandidate checkpoints:\n" + "\n".join(summaries)
        prompt = format_rollback_operation_prompt(body, available_checkpoints=ordered, hint=hint)
    elif view == "A1":
        # Pure state-only: no candidate summaries and no checkpoint ID list.
        prompt = format_rollback_operation_prompt(state, available_checkpoints=None, hint=hint)
    elif view == "A2":
        body = state + "\n" + _feasibility_scalars(sample, ordered)
        prompt = format_rollback_operation_prompt(body, available_checkpoints=None, hint=hint)
    elif view == "A3":
        body = (
            state
            + "\n"
            + _feasibility_scalars(sample, ordered)
            + "\n"
            + _failure_progress_scalars(sample)
        )
        prompt = format_rollback_operation_prompt(body, available_checkpoints=None, hint=hint)
    else:  # A4
        # Same structure/length as A0, but mask candidate semantic text.
        placeholders = [
            f"{c.get('local_checkpoint_id', f'C{i}')}:MASKED_CANDIDATE_{i}"
            for i, c in enumerate(ordered)
        ]
        body = state
        if placeholders:
            body = body + "\nCandidate checkpoints:\n" + "\n".join(placeholders)
        # Keep ID-list shape but replace global IDs with stable placeholders.
        masked = [
            {**c, "checkpoint_id": f"MASKED_ID_{c.get('local_checkpoint_id', i)}"}
            for i, c in enumerate(ordered)
        ]
        prompt = format_rollback_operation_prompt(body, available_checkpoints=masked, hint=hint)

    eff, truncated, before, after = _truncate(prompt, tokenizer, max_length)
    gold_op = _gold_operation(sample)
    gold_ck = _gold_checkpoint(sample)
    gold_local = global_to_local_id(gold_ck, local_to_global)
    gold_in = gold_ck is None or gold_ck in set(local_to_global.values())
    candidate_payload = [
        {
            "local_checkpoint_id": c.get("local_checkpoint_id"),
            "checkpoint_id": c.get("checkpoint_id"),
            "summary": summarize_candidate(c),
            "relative_turn": c.get("relative_turn", c.get("turn_id", 0)),
            "evidence_count": c.get("evidence_count", c.get("n_curated", 0)),
            "turn_id": c.get("turn_id", c.get("relative_turn", 0)),
            "n_pool": c.get("n_pool", 0),
        }
        for c in ordered
    ]
    return Stage1ViewResult(
        view=view,
        effective_input_text=eff,
        prompt_sha256=_sha256_text(eff),
        candidate_list=candidate_payload,
        gold_operation=gold_op,
        gold_checkpoint_local_id=gold_local,
        gold_checkpoint_global_id=gold_ck,
        gold_in_candidates=gold_in,
        truncated=truncated,
        token_length_before=before,
        token_length_after=after,
    )


def build_stage2_prompt(sample: dict[str, Any]) -> str:
    """Rollback-specific Stage2 input: candidate list + metadata only (no Stage1 body)."""
    ordered, _ = assign_local_checkpoint_ids(_candidates(sample))
    ds = sample.get("decision_state") or {}
    turn = int(ds.get("turn_id", sample.get("turn", sample.get("turn_id", 0))) or 0)
    lines = [
        "Checkpoint selection:",
        f"current_turn={turn}",
        f"remaining_rollback_budget={ds.get('remaining_recovery_budget', ds.get('remaining_search_budget', '?'))}",
        "Candidates:",
    ]
    for c in ordered:
        lines.append(summarize_candidate(c) + f",id={c.get('checkpoint_id')}")
    lines.append("Select checkpoint:")
    return "\n".join(lines)
