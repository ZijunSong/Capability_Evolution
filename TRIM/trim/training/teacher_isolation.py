"""Teacher branch isolation and component-kind routing (T04 / T05)."""

from __future__ import annotations

from typing import Any, Callable

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_realizability import fork_snapshot

# View: privilege is rendering. Auto-state: original env hooks. Tool: callable.
COMPONENT_KIND: dict[str, str] = {
    "sentence_compress": "view",
    "token_budget_marker": "view",
    "adaptive_rerank_instruction": "view",
    "importance_tagging": "view",
    "evidence_graph": "view",
    "auto_populate_first_search": "auto_state",
    "content_dedup": "auto_state",
    "chunk_neighbors": "auto_state",
    "subtractive_curation": "auto_state",
    "verify_tool": "callable_tool",
}


def student_state_hash(snapshot: EnvironmentSnapshot) -> str:
    return snapshot.content_hash()


def run_teacher_branch_isolated(
    student_snapshot: EnvironmentSnapshot,
    fn: Callable[[EnvironmentSnapshot], Any],
) -> tuple[Any, EnvironmentSnapshot]:
    """Run Teacher work on a fork. Student snapshot must be unchanged."""
    before = student_state_hash(student_snapshot)
    reward_before = (student_snapshot.metadata or {}).get("reward")
    branched = fork_snapshot(student_snapshot)
    result = fn(branched)
    after = student_state_hash(student_snapshot)
    if before != after:
        raise RuntimeError("Teacher branch mutated the Student snapshot")
    if (student_snapshot.metadata or {}).get("reward") != reward_before:
        raise RuntimeError("Teacher branch mutated Student reward")
    return result, student_snapshot
