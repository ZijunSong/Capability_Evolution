from __future__ import annotations

from types import SimpleNamespace

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_events import HarnessEvent, model_action
from trim.training.projected_teacher_sft.replay import (
    ExecuteResult,
    ReplayDecision,
    replay_projected_trace,
    row_from_projected_step,
)


class Enc:
    family = "plain"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_tokens(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def _snap(docs: list[str]) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        query_id="q",
        step=0,
        harness_mask={"verify_tool": False},
        working_memory={"curated_ids": list(docs), "accessible_doc_ids": list(docs)},
    )


class Env:
    def __init__(self, *, reveal_d1: bool, claim_d1: bool = False) -> None:
        self.reveal_d1 = reveal_d1
        self.claim_d1 = claim_d1
        self.pool: list[str] = []
        self.curated: list[str] = []
        self.ended = False
        self.executed: list[str] = []

    def decision(self) -> ReplayDecision:
        visible = ["d1"] if self.claim_d1 else list(dict.fromkeys(self.pool + self.curated))
        return ReplayDecision(
            EnvironmentSnapshot(
                query_id="q",
                step=0,
                harness_mask={"verify_tool": False},
                working_memory={
                    "curated_ids": list(self.curated),
                    "accessible_doc_ids": list(dict.fromkeys(self.pool + self.curated)),
                },
            ),
            [1, 2, 3],
            {"prompt_visible_doc_ids": visible},
        )

    def execute(self, action: dict) -> ExecuteResult:
        name = action["name"]
        self.executed.append(name)
        if name == "search_corpus":
            if self.reveal_d1:
                self.pool = ["d1"]
            return ExecuteResult(ok=True, post_state_hash="after-search")
        if name == "curate":
            wanted = set(action["arguments"].get("add_ids") or [])
            visible = set(self.pool + self.curated)
            if not wanted.issubset(visible):
                return ExecuteResult(ok=False, reason="inaccessible_reference")
            self.curated = sorted(visible | wanted)
            return ExecuteResult(ok=True, post_state_hash="after-curate")
        if name == "end_search":
            self.ended = True
            return ExecuteResult(ok=True, post_state_hash="end")
        return ExecuteResult(ok=False, reason=name)


def _event(event_id: str, name: str) -> HarnessEvent:
    return model_action(name, {"query": "q"} if name == "search_corpus" else {}, event_id=event_id)


def _choose(before, window):
    event = window[0]
    name = event.action_name
    if name == "verify":
        return SimpleNamespace(kind=SimpleNamespace(value="skip"), reject_reason="verify", anchor_event_id=None), []
    if name == "bad":
        return SimpleNamespace(kind=SimpleNamespace(value="direct"), anchor_event_id=event.event_id, reject_reason=None), []
    arguments = {"query": "q"} if name == "search_corpus" else {}
    if name == "curate":
        arguments = {"add_ids": ["d1"], "remove_ids": []}
    step = SimpleNamespace(target_action={"name": name, "arguments": arguments}, target_text="", weight=1.0)
    return SimpleNamespace(kind=SimpleNamespace(value="direct"), anchor_event_id=event.event_id, reject_reason=None), [step]


def _make(before, step, projection):
    return row_from_projected_step(
        before,
        step,
        projection,
        enc=Enc(),
        query_id="q",
        episode_id="ep",
        policy_version="teacher",
        max_model_len=8192,
        max_new_tokens=2048,
        row_id=f"row-{projection.anchor_event_id}",
    )


def test_replay_executes_search_before_curate_and_skips_verify():
    events = [
        _event("e-search", "search_corpus"),
        _event("e-verify", "verify"),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, event_id="e-curate"),
        _event("e-end", "end_search"),
    ]
    env = Env(reveal_d1=True)
    rows, audit = replay_projected_trace(events, env, _choose, _make, scan=1, max_steps=8)
    assert [row.metadata["native_projected_action"]["name"] for row in rows] == [
        "search_corpus",
        "curate",
        "end_search",
    ]
    assert env.executed == ["search_corpus", "curate", "end_search"]
    assert any(item["status"] == "skip_window" for item in audit)
    assert rows[1].prompt_ids == [1, 2, 3]
    assert rows[0].metadata["student_post_state_hash"] == "after-search"


def test_teacher_only_document_is_not_injected_and_failed_execute_drops_the_row():
    events = [
        _event("e-search", "search_corpus"),
        model_action("curate", {"add_ids": ["d1"], "remove_ids": []}, event_id="e-curate"),
    ]
    env = Env(reveal_d1=False, claim_d1=True)
    rows, audit = replay_projected_trace(events, env, _choose, _make, scan=1, max_steps=8)
    assert [row.metadata["native_projected_action"]["name"] for row in rows] == ["search_corpus"]
    assert audit[-1]["status"] == "stop_execution"
    assert audit[-1]["reason"] == "inaccessible_reference"


def test_rejected_anchor_keeps_the_following_event():
    events = [_event("e-bad", "bad"), _event("e-end", "end_search")]
    env = Env(reveal_d1=True)
    rows, audit = replay_projected_trace(events, env, _choose, _make, scan=2, max_steps=8)
    assert [item["status"] for item in audit] == ["reject_anchor", "aligned"]
    assert rows[0].metadata["native_projected_action"]["name"] == "end_search"


def test_duplicate_event_ids_are_rejected():
    events = [_event("same", "search_corpus"), _event("same", "end_search")]
    try:
        replay_projected_trace(events, Env(reveal_d1=True), _choose, _make, scan=1, max_steps=4)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate event ids were accepted")
