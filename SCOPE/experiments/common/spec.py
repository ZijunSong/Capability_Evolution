"""ExperimentSpec: explicit, auditable experiment declaration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = (
    "experiment_id",
    "group",
    "method",
    "capability",
    "variant",
    "changed_factor",
    "base_model",
    "dataset",
    "runtime_config",
    "seed",
    "output_dir",
)


@dataclass
class ExperimentSpec:
    experiment_id: str
    group: str
    method: str
    capability: str
    variant: str
    changed_factor: str
    base_model: str
    checkpoint: str | None = None
    dataset: str = "browsecomp_plus"
    train_manifest: str | None = None
    valid_manifest: str | None = None
    test_manifest: str | None = None
    runtime_config: str = "harness/configs/modules_minimal_v2.yaml"
    retriever: str = "bm25"
    prompt_renderer: str = "default"
    decision_state_schema: str = "v2"
    shadow_source: str = "same_state_on_policy"
    verification_mode: str = "full_gate"
    routing_mode: str = "endorse_correct_balanced"
    target_format: str = "typed_operation_o7"
    objective: str = "discriminative_ce"
    lora_rank: int = 64
    lora_alpha: int = 128
    learning_rate: float = 2.0e-4
    epochs: float = 1.0
    max_steps: int | None = None
    effective_batch_size: int = 8
    seed: int = 42
    rollout_seed: int | None = None
    max_turns: int = 35
    max_tokens: int = 2048
    temperature: float = 1.0
    threshold: float = 0.0
    gpu: int | str | None = None
    output_dir: str = ""
    expected_metrics: list[str] = field(default_factory=list)
    parent_experiment: str | None = None
    notes: str = ""
    # Free-form extras for ablation-specific knobs (must be declared in changed_factor).
    extras: dict[str, Any] = field(default_factory=dict)
    smoke_query_limit: int | None = None
    dry_run: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        missing = [f for f in REQUIRED_FIELDS if getattr(self, f, None) in (None, "")]
        if missing:
            raise ValueError(f"ExperimentSpec missing required fields: {missing}")
        if self.rollout_seed is None:
            self.rollout_seed = self.seed
        if not self.output_dir:
            raise ValueError("output_dir is required (no silent default)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        known = {f.name for f in fields(cls)}
        payload = dict(data)
        extras = dict(payload.pop("extras", {}) or {})
        unknown = {k: v for k, v in list(payload.items()) if k not in known}
        for k in unknown:
            extras[k] = payload.pop(k)
        payload["extras"] = extras
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid ExperimentSpec yaml: {path}")
        return cls.from_dict(data)

    def resolved_output_dir(self, repo_root: str | Path) -> Path:
        p = Path(self.output_dir)
        if not p.is_absolute():
            p = Path(repo_root) / p
        return p
