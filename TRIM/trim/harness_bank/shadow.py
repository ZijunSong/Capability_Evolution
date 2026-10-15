"""Isolated teacher fork. The student episode object is not stepped."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from trim.harness_bank.loader import BankLoadError, load_bank
from trim.harness_bank.runtime import EpisodeBankRuntime
from trim.harness_bank.types import BankBranchTrace, BankSpec
from trim.rsi.canonical import sha256_json
from trim.training.teacher_branch import SOURCE_CAPABILITY, TeacherBranchResult

ENV_RESTORE_VERSION = "rsi_fork_v1"


class StudentEnv:
    """Small mutable episode used by isolation tests and the teacher fork."""

    def __init__(self, visible: dict[str, Any], *, reward: float = 0.0) -> None:
        self.visible = deepcopy(visible)
        self.files = deepcopy(visible.get("files") or {})
        self.reward = float(reward)
        self.sampled_action = deepcopy(visible.get("sampled_action"))
        self.visible_events: list[dict[str, Any]] = []
        self._reward_hash = sha256_json({"reward": self.reward})

    def export(self) -> dict[str, Any]:
        return {
            "visible": deepcopy(self.visible),
            "files": deepcopy(self.files),
            "reward": self.reward,
            "reward_hash": self._reward_hash,
            "sampled_action": deepcopy(self.sampled_action),
            "visible_events": deepcopy(self.visible_events),
        }

    def search(self, query: str, hits: list[str]) -> None:
        pool = self.visible.setdefault("pool", {})
        for doc_id in hits:
            pool[str(doc_id)] = {"id": str(doc_id)}
        self.visible["n_tool_calls"] = int(self.visible.get("n_tool_calls") or 0) + 1
        self.visible_events.append(
            {
                "step": int(self.visible.get("step") or 0),
                "action": {"name": "search_corpus", "arguments": {"query": query}},
                "visible": deepcopy(self.visible),
            }
        )

    def curate(self, add_ids: list[str]) -> None:
        curated = self.visible.setdefault("curated", {})
        for doc_id in add_ids:
            curated[str(doc_id)] = {"id": str(doc_id)}
        self.visible["n_tool_calls"] = int(self.visible.get("n_tool_calls") or 0) + 1


class TeacherFork:
    def __init__(self, student: StudentEnv, bank: BankSpec) -> None:
        self.student_export = student.export()
        self.visible = deepcopy(self.student_export["visible"])
        self.files = deepcopy(self.student_export["files"])
        self.runtime = EpisodeBankRuntime(bank, role="teacher")
        self.bank = bank
        self.runtime.on_reset(
            {
                "query": self.visible.get("query") or "",
                "visible_doc_ids": list((self.visible.get("pool") or {}).keys()),
                "curated_ids": list((self.visible.get("curated") or {}).keys()),
                "step": int(self.visible.get("step") or 0),
                "visible_search_history": list(self.visible.get("tool_history") or []),
            }
        )
        self.effects: list[dict[str, Any]] = []

    def search(self, query: str, hits: list[str]) -> None:
        pool = self.visible.setdefault("pool", {})
        for doc_id in hits:
            pool[str(doc_id)] = {"id": str(doc_id)}
        self.visible["n_tool_calls"] = int(self.visible.get("n_tool_calls") or 0) + 1
        self.effects.append(
            {
                "name": "search_corpus",
                "arguments": {"query": query},
                "new_doc_ids": list(hits),
            }
        )

    def curate(self, add_ids: list[str]) -> None:
        curated = self.visible.setdefault("curated", {})
        for doc_id in add_ids:
            curated[str(doc_id)] = {"id": str(doc_id)}
        self.effects.append({"name": "curate", "arguments": {"add_ids": list(add_ids)}})

    def student_unchanged(self, student: StudentEnv) -> bool:
        return student.export() == self.student_export


def fork_teacher(student: StudentEnv, bank: BankSpec) -> TeacherFork:
    return TeacherFork(student, bank)


def restore_private_state(runtime: EpisodeBankRuntime, visible_events: list[dict[str, Any]], *, step: int) -> None:
    """Replay only events whose step is <= the decision. Later events are a hard error."""
    for event in visible_events:
        event_step = int(event.get("step") or 0)
        if event_step > step:
            raise RuntimeError(
                f"refusing to build the teacher condition from step {event_step} after decision {step}"
            )
    runtime.on_reset({"step": step, "visible_events": visible_events})


def bank_teacher_fn(bank: BankSpec | str, *, component_id: str = "all") -> Callable[..., TeacherBranchResult]:
    loaded = load_bank(bank) if isinstance(bank, str) else bank
    if not loaded.active_components:
        raise BankLoadError("empty bank cannot be a TRIM teacher")

    def _fn(point: Any) -> TeacherBranchResult:
        missing = [cid for cid in loaded.active_components if cid not in {spec.id for spec in loaded.components}]
        if missing:
            raise BankLoadError(f"UNREGISTERED_BANK_COMPONENT: {missing}")
        for spec in loaded.components:
            if not spec.recipe:
                raise BankLoadError(f"UNREGISTERED_BANK_COMPONENT: {spec.id} has no recipe")
        pre_hash = str(getattr(point, "pre_action_snapshot_hash", "") or "")
        events, guidance = collect_teacher_events(loaded, point)
        return TeacherBranchResult(
            decision_state_id=str(getattr(point, "decision_point_id", "") or ""),
            component_id=str(component_id),
            triggered=bool(events) or bool(guidance),
            source_type=SOURCE_CAPABILITY,
            events=events,
            teacher_condition={
                "continuous": True,
                "bank_joint": True,
                "bank_sha256": loaded.bank_sha256,
                "env_restore_version": ENV_RESTORE_VERSION,
                "student_pre_state_hash": pre_hash,
                "teacher_guidance": guidance,
                "teacher_prompt_token_ids": list(getattr(point, "teacher_prompt_token_ids", None) or []),
                "not_a_continuous_teacher_rollout": False,
            },
            skip_reason=None,
            worth_supervising=True,
        )

    return _fn


def collect_teacher_events(bank: BankSpec, point: Any) -> tuple[list[Any], str]:
    """Run bank hooks on a copy of the student snapshot. The snapshot stays put."""
    from trim.harness_bank.adaptation import event_from_effect

    snap = getattr(point, "pre_action_snapshot", None)
    before = snap.content_hash() if snap is not None and hasattr(snap, "content_hash") else ""
    wm = dict(getattr(snap, "working_memory", None) or {})
    visible = {
        "query": str(wm.get("query") or getattr(snap, "query_text", "") or ""),
        "visible_doc_ids": [str(x) for x in (wm.get("accessible_doc_ids") or [])],
        "curated_ids": [str(x) for x in (wm.get("curated_ids") or [])],
        "step": int(getattr(snap, "step", 0) or 0),
        "visible_search_history": list(getattr(snap, "tool_history", None) or []),
    }
    runtime = EpisodeBankRuntime(bank, role="teacher")
    runtime.on_reset(dict(visible))
    rendered = runtime.render(dict(visible), role="teacher")
    events = []
    for index, raw in enumerate(runtime.events):
        name = str(raw.get("action_name") or raw.get("name") or "")
        action = None if not name else {"name": name, "arguments": dict(raw.get("arguments") or {})}
        events.append(
            event_from_effect(
                component_id=str(raw.get("component_id") or bank.active_components[0]),
                action=action,
                order=index,
                provenance=list(raw.get("information_provenance") or [{"source": "visible_state", "available_to_student": True}]),
            )
        )
    if snap is not None and before and snap.content_hash() != before:
        raise RuntimeError("bank teacher mutated the student snapshot")
    return events, str(rendered.get("teacher_guidance") or "")


def trace_from_fork(
    fork: TeacherFork,
    *,
    branch_id: str,
    policy_digest: str,
    student_pre_state_hash: str,
    teacher_prompt_token_ids: list[int] | None = None,
) -> BankBranchTrace:
    from trim.harness_bank.adaptation import event_from_effect

    events = []
    for index, effect in enumerate(fork.effects):
        events.append(
            event_from_effect(
                component_id=fork.bank.active_components[0] if fork.bank.active_components else branch_id,
                action=effect,
                order=index,
                provenance=[{"source": "teacher_fork", "available_to_student": True}],
            )
        )
    return BankBranchTrace(
        branch_id=branch_id,
        bank_sha256=fork.bank.bank_sha256,
        source_policy_digest=policy_digest,
        student_pre_state_hash=student_pre_state_hash,
        events=tuple(events),
        teacher_prompt_token_ids=tuple(teacher_prompt_token_ids or []),
        teacher_state_hash=sha256_json(fork.visible),
        component_effect_ids=tuple(fork.bank.active_components),
        continuous=True,
        information_provenance=tuple(
            {"source": "teacher_fork", "available_to_student": True} for _ in events
        ),
        cost=dict(fork.runtime.cost),
    )
