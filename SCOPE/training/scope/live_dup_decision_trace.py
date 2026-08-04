"""Event-level live duplicate decision contract trace (Round 7)."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.capability.dup_operation import DupOperation
from training.scope.decide_dup_operation import COMPARISON_OPERATOR, decide_dup_operation

SCHEMA_VERSION = "round7.v1"
SCORE_TYPE = "mean_logprob"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()


@dataclass
class LiveDupDecisionTrace:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    event_id: str = ""
    query_id: str = ""
    turn_index: int = 0
    decision_index: int = 0

    model_id: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    seed: int = 0
    backend: str = "vllm_live"

    decision_state_sha256: str = ""
    decision_state_json: dict[str, Any] = field(default_factory=dict)
    candidate_evidence_sha256: str = ""
    candidate_evidence_id: str = ""
    pool_state_sha256: str = ""

    rendered_prompt_sha256: str = ""
    rendered_prompt: str = ""
    input_ids_sha256: str = ""
    input_length: int = 0
    keep_verbalizer_token_ids: list[int] = field(default_factory=list)
    skip_verbalizer_token_ids: list[int] = field(default_factory=list)

    score_type: str = SCORE_TYPE
    score_keep: float = 0.0
    score_skip: float = 0.0
    margin_definition: str = "score_skip-score_keep"
    margin: float = 0.0

    threshold_source: str = "fixed_zero"
    threshold_key: str = ""
    threshold: float = 0.0
    comparison_operator: str = COMPARISON_OPERATOR
    predicted_operation_pre_realizer: str = ""
    predicted_operation_post_realizer: str = ""

    shadow_label: str = ""
    shadow_route: str = ""
    shadow_label_state_sha256: str = ""

    actually_curated: bool = False
    action_name: str = ""
    action_payload_sha256: str = ""

    parse_success: bool = True
    fallback_used: bool = False
    fallback_reason: str | None = None
    timeout: bool = False
    exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveDupDecisionTrace:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LiveDupDecisionTraceWriter:
    """Append-only trace writer with prompt sidecar deduplication."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.trace_path = self.output_dir / "live_dup_decision_trace.jsonl"
        self.prompt_dir = self.output_dir / "prompt_sidecar"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self._written_prompt_hashes: set[str] = set()
        self._lock = threading.Lock()

    def write(self, trace: LiveDupDecisionTrace) -> None:
        trace.run_id = self.run_id
        with self._lock:
            if trace.rendered_prompt and trace.rendered_prompt_sha256:
                if trace.rendered_prompt_sha256 not in self._written_prompt_hashes:
                    sidecar = self.prompt_dir / f"{trace.rendered_prompt_sha256}.txt"
                    sidecar.write_text(trace.rendered_prompt, encoding="utf-8")
                    self._written_prompt_hashes.add(trace.rendered_prompt_sha256)
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> list[LiveDupDecisionTrace]:
        if not self.trace_path.exists():
            return []
        traces: list[LiveDupDecisionTrace] = []
        with self.trace_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(LiveDupDecisionTrace.from_dict(json.loads(line)))
        return traces


def build_event_id(query_id: str, turn_index: int, candidate_hash: str) -> str:
    return f"{query_id}:{turn_index}:{candidate_hash}"


def make_trace_from_decision(
    *,
    query_id: str,
    turn_index: int,
    decision_index: int,
    decision_state: dict[str, Any],
    rendered_prompt: str,
    input_ids: list[int],
    score_keep: float,
    score_skip: float,
    threshold: float,
    threshold_source: str,
    threshold_key: str,
    predicted_pre: DupOperation,
    predicted_post: DupOperation,
    candidate_evidence_id: str,
    shadow_label: str,
    shadow_route: str,
    actually_curated: bool,
    action_payload: dict[str, Any],
    model_id: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    seed: int,
    backend: str,
    keep_token_ids: list[int] | None = None,
    skip_token_ids: list[int] | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    exception: str | None = None,
) -> LiveDupDecisionTrace:
    decision = decide_dup_operation(
        score_keep=score_keep, score_skip=score_skip, threshold=threshold
    )
    cand_hash = sha256_text(candidate_evidence_id)
    pool_ids = decision_state.get("pool_document_ids") or []
    return LiveDupDecisionTrace(
        event_id=build_event_id(query_id, turn_index, cand_hash),
        query_id=query_id,
        turn_index=turn_index,
        decision_index=decision_index,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        seed=seed,
        backend=backend,
        decision_state_sha256=sha256_json(decision_state),
        decision_state_json=decision_state,
        candidate_evidence_sha256=cand_hash,
        candidate_evidence_id=candidate_evidence_id,
        pool_state_sha256=sha256_json({"pool_document_ids": list(pool_ids)}),
        rendered_prompt_sha256=sha256_text(rendered_prompt),
        rendered_prompt=rendered_prompt,
        input_ids_sha256=sha256_ids(input_ids),
        input_length=len(input_ids),
        keep_verbalizer_token_ids=list(keep_token_ids or []),
        skip_verbalizer_token_ids=list(skip_token_ids or []),
        score_keep=decision.score_keep,
        score_skip=decision.score_skip,
        margin=decision.margin,
        threshold_source=threshold_source,
        threshold_key=threshold_key,
        threshold=threshold,
        comparison_operator=decision.comparison_operator,
        predicted_operation_pre_realizer=predicted_pre.value,
        predicted_operation_post_realizer=predicted_post.value,
        shadow_label=shadow_label,
        shadow_route=shadow_route,
        shadow_label_state_sha256=sha256_json(decision_state),
        actually_curated=actually_curated,
        action_name="curate",
        action_payload_sha256=sha256_json(action_payload),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        exception=exception,
    )
