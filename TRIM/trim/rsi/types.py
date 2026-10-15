"""Campaign artifacts for the fixed-H0 RSI loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_POLICY = "trim_policy_artifact_v1"
SCHEMA_PANEL = "trim_panel_manifest_v1"
SCHEMA_DECISION = "trim_round_decision_v1"


@dataclass
class PolicyArtifact:
    """Weights handed from one RSI round to the next.

    The exported directory is a cumulative LoRA on the original base, or a
    merged checkpoint that HF and vLLM both load. ``digest`` identifies the
    weights actually served, not the parent directory name.
    """

    schema_version: str
    policy_id: str
    base_model_ref: str
    base_model_sha256: str
    export_dir: str
    deployment: str
    tokenizer_ref: str
    chat_template_sha256: str
    digest: str
    round_id: str
    parent_digest: str
    init_mode: str = "cumulative_adapter"
    cross_round_optimizer: str = "reset"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyArtifact":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class PanelManifest:
    round_id: str
    work_ids: tuple[str, ...]
    select_ids: tuple[str, ...]
    weight_dev_ids: tuple[str, ...]
    confirm_h_ids: tuple[str, ...]
    confirm_w_ids: tuple[str, ...]
    cluster_ids: dict[str, str] = field(default_factory=dict)
    dataset_manifest_hash: str = ""
    consumed_confirm_h: bool = False
    consumed_confirm_w: bool = False
    schema_version: str = SCHEMA_PANEL

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("work_ids", "select_ids", "weight_dev_ids", "confirm_h_ids", "confirm_w_ids"):
            payload[key] = list(payload[key])
        return payload


@dataclass
class BudgetLedgerEntry:
    campaign_id: str
    round_id: str
    job_id: str
    phase: str
    rollout_episodes: int = 0
    generated_tokens: int = 0
    gpu_hours: float = 0.0
    wall_clock_hours: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundDecision:
    status: str
    reason: str
    stop_search: bool = False
    eligible: bool = False
    confirm_h_pass: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_DECISION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
