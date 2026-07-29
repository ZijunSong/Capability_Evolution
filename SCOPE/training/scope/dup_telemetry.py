"""Closed-loop duplicate evidence admission telemetry (Round 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.capability.dup_decision_point import is_duplicate_candidate
from harness.capability.dup_operation import DupOperation


@dataclass
class AdmissionEvent:
    candidate_evidence_id: str
    candidate_is_duplicate: bool
    student_operation: str | None = None
    shadow_operation: str | None = None
    route: str | None = None
    realized_runtime_action: dict[str, Any] | None = None
    actually_curated: bool = False
    query_id: str = ""
    turn_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_evidence_id": self.candidate_evidence_id,
            "candidate_is_duplicate": self.candidate_is_duplicate,
            "student_operation": self.student_operation,
            "shadow_operation": self.shadow_operation,
            "route": self.route,
            "realized_runtime_action": self.realized_runtime_action,
            "actually_curated": self.actually_curated,
            "query_id": self.query_id,
            "turn_id": self.turn_id,
        }


@dataclass
class DupTelemetryAggregator:
    """Aggregate admission events into Round 3 primary behavioral metrics."""

    events: list[AdmissionEvent] = field(default_factory=list)
    operation_parse_failures: int = 0
    action_realizer_failures: int = 0
    hidden_fallback_count: int = 0

    def add(self, event: AdmissionEvent) -> None:
        self.events.append(event)

    def _confusion(self) -> dict[str, int]:
        keep_tp = keep_fp = keep_fn = skip_tp = skip_fp = skip_fn = 0
        n_dup = n_unique = 0
        n_dup_curated = 0
        n_unique_skipped = 0
        n_keep = n_skip = 0

        for ev in self.events:
            if ev.candidate_is_duplicate:
                n_dup += 1
            else:
                n_unique += 1
            op = (ev.student_operation or "").upper()
            if op == DupOperation.KEEP_EVIDENCE.value:
                n_keep += 1
            elif op == DupOperation.SKIP_DUPLICATE.value:
                n_skip += 1
            if ev.candidate_is_duplicate and ev.actually_curated:
                n_dup_curated += 1
            if not ev.candidate_is_duplicate and op == DupOperation.SKIP_DUPLICATE.value:
                n_unique_skipped += 1

            shadow = (ev.shadow_operation or "").upper()
            if shadow == DupOperation.KEEP_EVIDENCE.value:
                if op == DupOperation.KEEP_EVIDENCE.value:
                    keep_tp += 1
                else:
                    keep_fn += 1
                if op == DupOperation.SKIP_DUPLICATE.value:
                    keep_fp += 1
            elif shadow == DupOperation.SKIP_DUPLICATE.value:
                if op == DupOperation.SKIP_DUPLICATE.value:
                    skip_tp += 1
                else:
                    skip_fn += 1
                if op == DupOperation.KEEP_EVIDENCE.value:
                    skip_fp += 1

        return {
            "n_duplicate_candidates": n_dup,
            "n_unique_candidates": n_unique,
            "n_keep": n_keep,
            "n_skip": n_skip,
            "n_dup_curated": n_dup_curated,
            "n_unique_skipped": n_unique_skipped,
            "keep_tp": keep_tp,
            "keep_fp": keep_fp,
            "keep_fn": keep_fn,
            "skip_tp": skip_tp,
            "skip_fp": skip_fp,
            "skip_fn": skip_fn,
        }

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        return {"precision": prec, "recall": rec, "f1": f1}

    def summarize(self) -> dict[str, Any]:
        c = self._confusion()
        n_dup = c["n_duplicate_candidates"]
        n_unique = c["n_unique_candidates"]
        dup_curate_rate = c["n_dup_curated"] / max(n_dup, 1)
        false_skip_rate = c["n_unique_skipped"] / max(n_unique, 1)
        keep = self._prf(c["keep_tp"], c["keep_fp"], c["keep_fn"])
        skip = self._prf(c["skip_tp"], c["skip_fp"], c["skip_fn"])
        macro_f1 = (keep["f1"] + skip["f1"]) / 2.0
        balanced_acc = (
            keep["recall"] + skip["recall"]
        ) / 2.0 if (keep["recall"] + skip["recall"]) > 0 else 0.0
        unique_ratio = n_unique / max(n_dup + n_unique, 1)
        return {
            **c,
            "duplicate_curate_rate": dup_curate_rate,
            "false_skip_rate": false_skip_rate,
            "unique_evidence_ratio": unique_ratio,
            "KEEP_EVIDENCE": keep,
            "SKIP_DUPLICATE": skip,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "operation_parse_failures": self.operation_parse_failures,
            "action_realizer_failures": self.action_realizer_failures,
            "hidden_fallback_count": self.hidden_fallback_count,
            "telemetry_complete": self.operation_parse_failures == 0
            and self.hidden_fallback_count == 0,
        }

    def to_events_jsonl(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def classify_candidate(
    candidate_id: str,
    curated_ids: list[str] | tuple[str, ...],
) -> bool:
    return is_duplicate_candidate(candidate_id, curated_ids)
