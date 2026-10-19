"""Experiment identity, row schema, and resume fingerprints.

This module must stay free of torch and Harness imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class SFTProfile:
    experiment: str
    state_source: str
    prompt_owner: str
    supervision_provider: str
    projection: bool
    online_refresh: bool
    loss: str = "action_ce_v1"
    rl_enabled: bool = False
    gap_enabled: bool = False
    gate_enabled: bool = False
    reward_filter: str = "none"
    teacher_harness: str = "all"
    eval_harness: str = "zero"


PROFILES = {
    "teacher_sft": SFTProfile(
        "teacher_sft",
        "fixed_teacher_rollout",
        "teacher_full",
        "full_model_trajectory_v1",
        False,
        False,
    ),
    "projected_teacher_sft": SFTProfile(
        "projected_teacher_sft",
        "projected_teacher_replay",
        "student_zero",
        "full_model_trajectory_v1",
        True,
        False,
    ),
    "student_onpolicy_sft": SFTProfile(
        "student_onpolicy_sft",
        "current_student_rollout",
        "student_zero",
        "trim_component_branches_v1",
        True,
        True,
    ),
}

SCHEMA_VERSION = "sft_ablation_contract_v1"
ROW_SCHEMA_VERSION = "sft_batch_row_v1"
LOSS_VERSION = "action_ce_global_token_mean_v1"
WEIGHT_NORMALIZATION = "global_supervised_token_sum_v1"

# A uses a uniform weight. B/C keep the projector effective weight, including
# same-state sharing already applied by the current TRIM projector.
WEIGHT_POLICY = {
    "teacher_sft": "uniform_one_v1",
    "projected_teacher_sft": "projection_effective_weight_v1",
    "student_onpolicy_sft": "projection_effective_weight_v1",
}

IMMUTABLE_KEYS = (
    "schema_version",
    "experiment",
    "state_source",
    "prompt_owner",
    "supervision_provider",
    "projection",
    "online_refresh",
    "loss",
    "rl_enabled",
    "gap_enabled",
    "gate_enabled",
    "reward_filter",
    "teacher_harness",
    "eval_harness",
    "weight_policy",
    "weight_normalization",
    "base_checkpoint_digest",
    "init_adapter_digest",
    "teacher_policy_digest",
    "lora_target",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "dtype",
    "query_order_digest",
    "train_split",
    "corpus_digest",
    "retrieval_index_digest",
    "teacher_archive_digest",
    "train_env",
    "teacher_mask_digest",
    "student_mask_digest",
    "tool_schema_digest",
    "harness_source_digest",
    "verifier_config_version",
    "tokenizer_digest",
    "model_family",
    "prompt_template_digest",
    "max_model_len",
    "max_new_tokens",
    "action_only_supervision",
    "batch_size",
    "micro_batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "seed",
    "projector_digest",
    "state_sampling_rule",
    "branch_selection",
    "empty_supervision_policy",
    "max_anchor_scan_events",
    "noop_policy",
    "group_size",
    "tensor_parallel_size",
    "rollout_replicas",
    "nnodes",
    "nproc_per_node",
    "gradient_ranks",
)


def fingerprint(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def profile_record(experiment: str) -> dict[str, Any]:
    if experiment not in PROFILES:
        raise ValueError(f"unknown SFT experiment: {experiment}")
    return {"schema_version": SCHEMA_VERSION, **asdict(PROFILES[experiment])}


def require_same_identity(saved: Mapping[str, Any] | None, live: Mapping[str, Any] | None) -> None:
    """Reject resume when any immutable identity field differs."""
    if not isinstance(saved, dict) or not isinstance(live, dict):
        raise ValueError("missing SFT identity")
    required = {"schema_version", "experiment", "state_source", "loss"}
    if not required.issubset(saved) or not required.issubset(live):
        raise ValueError("incomplete SFT identity")
    keys = set(saved) | set(live)
    mismatch = sorted(
        key for key in keys if key not in saved or key not in live or saved[key] != live[key]
    )
    if mismatch:
        raise ValueError(f"cross-experiment/config resume refused: {mismatch}")


def immutable_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in IMMUTABLE_KEYS if key not in record]
    if missing:
        raise ValueError(f"incomplete SFT identity: {missing}")
    return {key: record[key] for key in IMMUTABLE_KEYS}


@dataclass
class SFTBatchRow:
    row_id: str
    experiment: str
    query_id: str
    episode_id: str
    decision_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[bool]
    weight: float
    policy_version: str
    prompt_owner: str
    target_source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, max_length: int) -> None:
        if self.experiment not in PROFILES:
            raise ValueError("unknown SFT experiment")
        profile = PROFILES[self.experiment]
        if self.prompt_owner != profile.prompt_owner:
            raise ValueError("wrong prompt owner")
        if self.target_source != profile.supervision_provider:
            raise ValueError("wrong target source")
        if not self.prompt_ids or not self.response_ids:
            raise ValueError("empty prompt/target")
        if len(self.response_ids) != len(self.response_mask):
            raise ValueError("target/mask length mismatch")
        if not any(self.response_mask):
            raise ValueError("zero supervised tokens")
        if not isfinite(self.weight) or self.weight <= 0:
            raise ValueError("invalid CE row weight")
        if len(self.prompt_ids) + len(self.response_ids) > max_length:
            raise ValueError("sequence exceeds sampled context budget")
        if not self.policy_version:
            raise ValueError("missing source policy version")
        tokens = self.prompt_ids + self.response_ids
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("invalid token IDs")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = ROW_SCHEMA_VERSION
        payload["response_mask"] = [bool(bit) for bit in self.response_mask]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SFTBatchRow":
        schema = data.get("schema_version", ROW_SCHEMA_VERSION)
        if schema != ROW_SCHEMA_VERSION:
            raise ValueError(f"unsupported SFT row schema: {schema}")
        return cls(
            row_id=str(data["row_id"]),
            experiment=str(data["experiment"]),
            query_id=str(data["query_id"]),
            episode_id=str(data["episode_id"]),
            decision_id=str(data["decision_id"]),
            prompt_ids=[int(x) for x in data["prompt_ids"]],
            response_ids=[int(x) for x in data["response_ids"]],
            response_mask=[bool(x) for x in data["response_mask"]],
            weight=float(data["weight"]),
            policy_version=str(data["policy_version"]),
            prompt_owner=str(data["prompt_owner"]),
            target_source=str(data["target_source"]),
            metadata=dict(data.get("metadata") or {}),
        )


def supervised_weight(row: SFTBatchRow) -> float:
    return float(row.weight) * float(sum(1 for bit in row.response_mask if bit))


def package_digest(rows: list[SFTBatchRow]) -> str:
    ordered = sorted((row.to_dict() for row in rows), key=lambda item: item["row_id"])
    return fingerprint(ordered)
