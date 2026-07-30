"""Scorer loading and parity utilities for Round 6."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.capability.dup_operation import DupOperation
from training.scope.decision_config import DupDecisionConfig, DEFAULT_DECISION_CONFIG
from training.scope.compact_target import compact_target_from_sample
from training.scope.operation_scorer import score_operations, OperationScoreResult
from training.scope.prompting import format_operation_prompt_from_sample
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig
from training.scope_round6.common import (
    BASE_MODEL,
    adapter_path,
    merged_path,
    sha256_ids,
    sha256_text,
    seed_from_tag,
)
from training.scope_round6.metrics import ScoredRow


def load_hf_trainer(
    *,
    adapter: Path | None = None,
    merged: Path | None = None,
    gpu: str = "cuda:0",
    loss_mode: str = "discriminative_ce",
) -> DupSDITrainer:
    model_path = str(merged or BASE_MODEL)
    cfg = SDITrainConfig(
        model_path=model_path,
        output_dir=Path("/tmp/scope_r6_eval"),
        adapter_path=str(adapter) if adapter else None,
        loss_mode=loss_mode,
        eval_only=True,
        device=gpu,
    )
    return DupSDITrainer(cfg)


def load_merged_model(merged: Path, gpu: str = "cuda:0") -> tuple[Any, Any, torch.device]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(merged), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(merged), torch_dtype=dtype, trust_remote_code=True
    )
    dev = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    return model, tokenizer, dev


def load_adapter_model(adapter: Path, gpu: str = "cuda:0") -> tuple[Any, Any, torch.device]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    dev = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    return model, tokenizer, dev


def _operation_context(sample: dict[str, Any]) -> tuple[str | None, list[str]]:
    ds = sample.get("decision_state") or {}
    cid = sample.get("candidate_evidence_id")
    if not cid:
        compact = compact_target_from_sample(sample)
        if compact:
            cid = compact.candidate_id
    curated = ds.get("curated_document_ids") or ds.get("curated_evidence_ids") or []
    return str(cid) if cid else None, [str(x) for x in curated]


def score_sample_hf(
    model,
    tokenizer,
    sample: dict[str, Any],
    device: torch.device,
    decision_cfg: DupDecisionConfig = DEFAULT_DECISION_CONFIG,
) -> ScoredRow:
    state_text = str(
        sample.get("student_state_text")
        or (sample.get("decision_state") or {}).get("rendered_context")
        or ""
    )
    cid, curated = _operation_context(sample)
    result = score_operations(
        model, tokenizer, state_text,
        device=device, candidate_id=cid, curated_document_ids=curated,
    )
    sk = result.scores[DupOperation.KEEP_EVIDENCE.value]
    ss = result.scores[DupOperation.SKIP_DUPLICATE.value]
    margin = ss - sk
    pred = decision_cfg.predict_from_margin(margin).value
    label = str(
        sample.get("shadow_operation")
        or (compact_target_from_sample(sample).operation.value
            if compact_target_from_sample(sample) else "")
    ).upper()
    if not label:
        label = DupOperation.KEEP_EVIDENCE.value
    return ScoredRow(label=label, margin=margin, score_keep=sk, score_skip=ss, prediction=pred)


def score_samples_hf(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    device: torch.device,
    decision_cfg: DupDecisionConfig = DEFAULT_DECISION_CONFIG,
) -> list[ScoredRow]:
    return [score_sample_hf(model, tokenizer, s, device, decision_cfg) for s in samples]


def parity_predictions(
    preds_a: list[str],
    preds_b: list[str],
) -> float:
    if not preds_a:
        return 1.0
    matches = sum(1 for a, b in zip(preds_a, preds_b) if a == b)
    return matches / len(preds_a)


def input_hashes(sample: dict[str, Any], tokenizer) -> dict[str, str]:
    rendered = format_operation_prompt_from_sample(sample)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    ds = sample.get("decision_state") or {}
    state_text = str(ds.get("rendered_context") or ds.get("query") or "")
    return {
        "rendered_input_hash": sha256_text(rendered),
        "token_ids_hash": sha256_ids(ids),
        "state_hash": sha256_text(state_text),
    }


def scorer_paths_for_tag(tag: str) -> dict[str, Path]:
    seed = seed_from_tag(tag)
    return {
        "adapter": adapter_path(seed),
        "merged": merged_path(seed),
    }
