"""Policy artifacts, runtime contracts and sampling parameters.

``RuntimeContract`` and ``PolicyArtifact`` are stored separately and combined
into a run identity so that a model update is never mistaken for an H0 change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rsi.contracts.common import JSONValue, digest_of

POLICY_ARTIFACT_SCHEMA = "rsi_policy_v1"
RUNTIME_CONTRACT_SCHEMA = "rsi_runtime_contract_v1"


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def digest(self) -> str:
        return digest_of(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "stop": list(self.stop),
                "seed": self.seed,
            }
        )


@dataclass(frozen=True)
class PolicyArtifact:
    """Immutable identity of a model policy (base + optional adapter)."""

    policy_id: str  # digest over the fields below
    base_model_id: str
    kind: str = "base"  # base | trained | exported | merged
    adapter_path: str | None = None
    adapter_digest: str | None = None
    parent_policy_id: str | None = None
    extra: Mapping[str, JSONValue] = field(default_factory=dict)

    @staticmethod
    def make(
        base_model_id: str,
        *,
        kind: str = "base",
        adapter_path: str | None = None,
        adapter_digest: str | None = None,
        parent_policy_id: str | None = None,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> "PolicyArtifact":
        fields = {
            "base_model_id": base_model_id,
            "kind": kind,
            "adapter_path": adapter_path,
            "adapter_digest": adapter_digest,
            "parent_policy_id": parent_policy_id,
            "extra": dict(extra or {}),
        }
        return PolicyArtifact(policy_id=digest_of(fields), **fields)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "schema": POLICY_ARTIFACT_SCHEMA,
            "policy_id": self.policy_id,
            "base_model_id": self.base_model_id,
            "kind": self.kind,
            "adapter_path": self.adapter_path,
            "adapter_digest": self.adapter_digest,
            "parent_policy_id": self.parent_policy_id,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_json(obj: Mapping[str, JSONValue]) -> "PolicyArtifact":
        art = PolicyArtifact(
            policy_id=str(obj["policy_id"]),
            base_model_id=str(obj["base_model_id"]),
            kind=str(obj.get("kind", "base")),
            adapter_path=obj.get("adapter_path"),  # type: ignore[arg-type]
            adapter_digest=obj.get("adapter_digest"),  # type: ignore[arg-type]
            parent_policy_id=obj.get("parent_policy_id"),  # type: ignore[arg-type]
            extra=dict(obj.get("extra") or {}),  # type: ignore[arg-type]
        )
        recomputed = PolicyArtifact.make(
            art.base_model_id,
            kind=art.kind,
            adapter_path=art.adapter_path,
            adapter_digest=art.adapter_digest,
            parent_policy_id=art.parent_policy_id,
            extra=art.extra,
        ).policy_id
        if recomputed != art.policy_id:
            raise ValueError("PolicyArtifact digest mismatch on load")
        return art


@dataclass(frozen=True)
class RuntimeContract:
    """Frozen H0 identity.  Must not change across campaign rounds."""

    h0_source_hash: str
    base_prompt_hash: str
    parser_id: str
    context_policy_id: str
    base_tool_schema_hash: str
    visibility_protocol: str
    budgets: Mapping[str, JSONValue]
    scoring_protocol: str
    env_identity: Mapping[str, JSONValue] = field(default_factory=dict)
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    contract_version: str = RUNTIME_CONTRACT_SCHEMA

    def digest(self) -> str:
        return digest_of(
            {
                "contract_version": self.contract_version,
                "h0_source_hash": self.h0_source_hash,
                "base_prompt_hash": self.base_prompt_hash,
                "parser_id": self.parser_id,
                "context_policy_id": self.context_policy_id,
                "base_tool_schema_hash": self.base_tool_schema_hash,
                "visibility_protocol": self.visibility_protocol,
                "budgets": dict(self.budgets),
                "scoring_protocol": self.scoring_protocol,
                "env_identity": dict(self.env_identity),
                "tokenizer_hash": self.tokenizer_hash,
                "chat_template_hash": self.chat_template_hash,
            }
        )

    def to_json(self) -> dict[str, JSONValue]:
        d = {
            "contract_version": self.contract_version,
            "h0_source_hash": self.h0_source_hash,
            "base_prompt_hash": self.base_prompt_hash,
            "parser_id": self.parser_id,
            "context_policy_id": self.context_policy_id,
            "base_tool_schema_hash": self.base_tool_schema_hash,
            "visibility_protocol": self.visibility_protocol,
            "budgets": dict(self.budgets),
            "scoring_protocol": self.scoring_protocol,
            "env_identity": dict(self.env_identity),
            "tokenizer_hash": self.tokenizer_hash,
            "chat_template_hash": self.chat_template_hash,
        }
        return d


@dataclass(frozen=True)
class RunIdentity:
    """Joint identity of one evaluation / rollout job (JobKey material)."""

    runtime_contract_digest: str
    policy_digest: str
    bank_digest: str
    environment_digest: str
    split_digest: str
    seed: int
    decoding_digest: str
    budget_digest: str
    scorer_version: str
    projector_version: str
    config_digest: str

    def job_key(self) -> str:
        return digest_of(
            {
                "runtime_contract_digest": self.runtime_contract_digest,
                "policy_digest": self.policy_digest,
                "bank_digest": self.bank_digest,
                "environment_digest": self.environment_digest,
                "split_digest": self.split_digest,
                "seed": self.seed,
                "decoding_digest": self.decoding_digest,
                "budget_digest": self.budget_digest,
                "scorer_version": self.scorer_version,
                "projector_version": self.projector_version,
                "config_digest": self.config_digest,
            }
        )
