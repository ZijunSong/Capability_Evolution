"""Two linked archives (task book §7.3).

- ``runtime_archive``: every legal candidate's performance, cost, diversity and
  lineage.  Candidates without OPD support are welcome here.
- ``distillation_archive``: candidates that passed the *real* OPD probe, with
  their effective supervision coverage, provenance and limitations.

Archives are append-only JSONL with a manifest; selection never rewrites
history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of


@dataclass(frozen=True)
class RuntimeEntry:
    candidate_id: str
    bank_digest: str
    parent_bank_digest: str
    generation: int
    operation: str
    hypothesis: str
    scores: Mapping[str, Mapping[str, float]]  # domain -> task -> value
    macro_score: float
    cost: Mapping[str, float]
    lineage: tuple[str, ...] = ()
    status: str = "legal"  # legal | invalid | rejected
    model_output_ref: str | None = None

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "bank_digest": self.bank_digest,
            "parent_bank_digest": self.parent_bank_digest,
            "generation": self.generation,
            "operation": self.operation,
            "hypothesis": self.hypothesis,
            "scores": {d: dict(t) for d, t in self.scores.items()},
            "macro_score": self.macro_score,
            "cost": dict(self.cost),
            "lineage": list(self.lineage),
            "status": self.status,
            "model_output_ref": self.model_output_ref,
        }


@dataclass(frozen=True)
class DistillationEntry:
    candidate_id: str
    bank_digest: str
    probe_report: Mapping[str, JSONValue]
    valid_opd_samples: int
    effective_supervised_tokens: int
    opd_grad_norm: float | None
    coverage: Mapping[str, float]
    limitations: tuple[str, ...] = ()
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "bank_digest": self.bank_digest,
            "probe_report": dict(self.probe_report),
            "valid_opd_samples": self.valid_opd_samples,
            "effective_supervised_tokens": self.effective_supervised_tokens,
            "opd_grad_norm": self.opd_grad_norm,
            "coverage": dict(self.coverage),
            "limitations": list(self.limitations),
            "provenance": dict(self.provenance),
        }


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime_path = self.root / "runtime_archive.jsonl"
        self.distill_path = self.root / "distillation_archive.jsonl"

    def append_runtime(self, entry: RuntimeEntry) -> None:
        with self.runtime_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json(), sort_keys=True, ensure_ascii=False) + "\n")

    def append_distillation(self, entry: DistillationEntry) -> None:
        with self.distill_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json(), sort_keys=True, ensure_ascii=False) + "\n")

    def runtime_entries(self) -> list[RuntimeEntry]:
        return [self._runtime_from_json(obj) for obj in self._read(self.runtime_path)]

    def distillation_entries(self) -> list[DistillationEntry]:
        return [self._distill_from_json(obj) for obj in self._read(self.distill_path)]

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    @staticmethod
    def _runtime_from_json(obj: dict[str, Any]) -> RuntimeEntry:
        return RuntimeEntry(
            candidate_id=str(obj["candidate_id"]),
            bank_digest=str(obj["bank_digest"]),
            parent_bank_digest=str(obj["parent_bank_digest"]),
            generation=int(obj["generation"]),
            operation=str(obj["operation"]),
            hypothesis=str(obj["hypothesis"]),
            scores={d: dict(t) for d, t in dict(obj["scores"]).items()},
            macro_score=float(obj["macro_score"]),
            cost=dict(obj["cost"]),
            lineage=tuple(obj.get("lineage") or ()),
            status=str(obj.get("status", "legal")),
            model_output_ref=obj.get("model_output_ref"),
        )

    @staticmethod
    def _distill_from_json(obj: dict[str, Any]) -> DistillationEntry:
        return DistillationEntry(
            candidate_id=str(obj["candidate_id"]),
            bank_digest=str(obj["bank_digest"]),
            probe_report=dict(obj["probe_report"]),
            valid_opd_samples=int(obj["valid_opd_samples"]),
            effective_supervised_tokens=int(obj["effective_supervised_tokens"]),
            opd_grad_norm=obj.get("opd_grad_norm"),
            coverage=dict(obj["coverage"]),
            limitations=tuple(obj.get("limitations") or ()),
            provenance=dict(obj.get("provenance") or {}),
        )
