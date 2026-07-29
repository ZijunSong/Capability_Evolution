"""Build Dup-only SDI dataset from v3 supervision samples (query-level split)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.scope.dataset_builder import split_by_query, write_split_jsonl
from training.scope.schema import DecisionSupervisionSampleV3


@dataclass
class DupDatasetManifest:
    n_samples: int
    n_train: int
    n_valid: int
    n_queries: int
    route_counts: dict[str, int]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "scope.dataset.dup_only.v1",
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_queries": self.n_queries,
            "route_counts": dict(self.route_counts),
            "provenance": dict(self.provenance),
        }


def load_samples_jsonl(path: Path) -> list[DecisionSupervisionSampleV3]:
    out: list[DecisionSupervisionSampleV3] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(DecisionSupervisionSampleV3.from_dict(json.loads(line)))
    return out


def filter_dup_trainable(
    samples: list[DecisionSupervisionSampleV3],
) -> list[DecisionSupervisionSampleV3]:
    return [
        s
        for s in samples
        if s.capability_id == "duplicate_evidence" and int(s.train_mask) == 1
    ]


def build_dup_dataset(
    samples_path: Path,
    out_dir: Path,
    *,
    valid_fraction: float = 0.1,
    seed: int = 42,
    provenance: dict[str, Any] | None = None,
) -> tuple[list[DecisionSupervisionSampleV3], list[DecisionSupervisionSampleV3], DupDatasetManifest]:
    all_samples = filter_dup_trainable(load_samples_jsonl(samples_path))
    train, valid = split_by_query(
        all_samples, valid_fraction=valid_fraction, seed=seed
    )
    route_counts: dict[str, int] = {}
    for s in all_samples:
        route_counts[s.route.value] = route_counts.get(s.route.value, 0) + 1
    queries = {s.task_id or s.episode_id for s in all_samples}
    manifest = DupDatasetManifest(
        n_samples=len(all_samples),
        n_train=len(train),
        n_valid=len(valid),
        n_queries=len(queries),
        route_counts=route_counts,
        provenance=dict(provenance or {}),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split_jsonl(out_dir / "train.jsonl", train)
    write_split_jsonl(out_dir / "valid.jsonl", valid)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return train, valid, manifest
