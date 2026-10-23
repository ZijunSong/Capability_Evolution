"""Toy task worlds.

Two genuinely different basic-tool catalogs prove the core is domain-agnostic
(``test_cross_environment_core``) without any benchmark-name routing:

- :class:`ToyFileWorld` — virtual filesystem arithmetic/text tasks
  (``list_dir`` / ``read_file`` / ``write_file`` / finish).
- :class:`ToyKVWorld` — key-value store tasks (``put`` / ``get`` / ``scan`` /
  ``delete`` / finish).

Both fully support snapshot/fork/replay (the mutable state is an explicit
value object), carry real official-style scorers, and mark every synthetic
artifact with ``synthetic=true`` so report merge rejects them from formal
results.  These are protocol-real, not performance-relevant.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue, canonical_json, digest_of
from rsi.contracts.errors import EnvironmentError_, ProtocolViolation
from rsi.contracts.observation import (
    ArtifactRef,
    Observation,
    ObservationProvenance,
    ScoreRecord,
    SnapshotRef,
    StepResult,
    ToolSpec,
)
from rsi.contracts.task import PrivateTaskRef, PublicTask

TOY_FILE_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_dir",
        description="List file names in a directory of the virtual workspace.",
        parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        capabilities=("fs_read",),
    ),
    ToolSpec(
        name="read_file",
        description="Read the text content of a file in the virtual workspace.",
        parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        capabilities=("fs_read",),
    ),
    ToolSpec(
        name="write_file",
        description="Write text content to a file in the virtual workspace (creates/overwrites).",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        capabilities=("fs_write",),
    ),
)

TOY_KV_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="put",
        description="Store a string value under a key.",
        parameters_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
        capabilities=("kv_write",),
    ),
    ToolSpec(
        name="get",
        description="Read the value stored under a key.",
        parameters_schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        capabilities=("kv_read",),
    ),
    ToolSpec(
        name="scan",
        description="List all keys currently stored, sorted.",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        capabilities=("kv_read",),
    ),
    ToolSpec(
        name="delete",
        description="Delete a key from the store.",
        parameters_schema={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        capabilities=("kv_write",),
    ),
)


def _toy_snapshot(kind: str, state_obj: Any) -> SnapshotRef:
    digest = digest_of(state_obj)
    return SnapshotRef(ref_id=f"{kind}:{digest[:16]}", digest=digest, kind="full")


class _ToySnapshotStore:
    """Process-local snapshot registry shared by an adapter and its forks."""

    def __init__(self) -> None:
        self._snaps: dict[str, Any] = {}

    def put(self, ref: SnapshotRef, state: Any) -> None:
        self._snaps[ref.ref_id] = copy.deepcopy(state)

    def get(self, ref: SnapshotRef) -> Any:
        if ref.ref_id not in self._snaps:
            raise EnvironmentError_(f"unknown snapshot {ref.ref_id}")
        return copy.deepcopy(self._snaps[ref.ref_id])

    def digest_matches(self, ref: SnapshotRef, state: Any) -> bool:
        return digest_of(state) == ref.digest


@dataclass
class _ToyFileState:
    files: dict[str, str] = field(default_factory=dict)
    expected_out: str = ""
    task_id: str = ""
    rng_state: int = 0
    clock: int = 0

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "files": dict(self.files),
            "expected_out": self.expected_out,
            "task_id": self.task_id,
            "rng_state": self.rng_state,
            "clock": self.clock,
        }


class ToyFileWorld:
    """Virtual-filesystem task world (domain ``toy_file``)."""

    name = "toy_file"
    version = "toy_file_v1"
    supports_shadow_fork = True

    def __init__(self, store: _ToySnapshotStore | None = None):
        self._store = store or _ToySnapshotStore()
        self._state = _ToyFileState()
        self._started = False

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        meta = dict(public_task.metadata)
        files = {str(k): str(v) for k, v in dict(meta.get("files") or {}).items()}
        if not files:
            raise EnvironmentError_(f"task {public_task.task_id} has no workspace files")
        # NOTE: expected answers never enter PublicTask metadata; scoring lives
        # entirely on the evaluator side (ToyScorer).
        self._state = _ToyFileState(
            files=files,
            expected_out="",
            task_id=public_task.task_id,
            rng_state=int(seed),
            clock=0,
        )
        self._started = True
        return Observation(
            content=f"workspace ready with files: {sorted(files)}",
            provenance=ObservationProvenance(source_tool="reset", source_ref=public_task.task_id),
        )

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return TOY_FILE_TOOLS

    def step(self, action: Action) -> StepResult:
        self._require_started()
        self._state.clock += 1
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(
                    content="finished",
                    provenance=ObservationProvenance(source_tool="finish", source_ref=self._state.task_id),
                ),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        obs = self._run_tool(action)
        return StepResult(
            observation=obs,
            done=False,
            snapshot=self.snapshot(),
            cost={"tool_calls": 1},
        )

    def _run_tool(self, call: ToolCall) -> Observation:
        known = {t.name for t in TOY_FILE_TOOLS}
        if call.name not in known:
            return Observation(
                content=f"unknown tool {call.name!r}",
                ok=False,
                error_kind="model_invalid_action",
                provenance=ObservationProvenance(source_tool=call.name, source_ref="schema"),
            )
        args = dict(call.arguments)
        path = str(args.get("path", ""))
        if ".." in path.split("/"):
            return Observation(
                content="path escapes workspace",
                ok=False,
                error_kind="tool_error",
                provenance=ObservationProvenance(source_tool=call.name, source_ref=path),
            )
        if call.name == "list_dir":
            names = sorted({p.split("/")[0] if "/" in p else p for p in self._state.files})
            return Observation(
                content=json.dumps(names),
                provenance=ObservationProvenance(source_tool="list_dir", source_ref=path or "."),
            )
        if call.name == "read_file":
            if path not in self._state.files:
                return Observation(
                    content=f"no such file: {path}",
                    ok=False,
                    error_kind="tool_error",
                    provenance=ObservationProvenance(source_tool="read_file", source_ref=path),
                )
            return Observation(
                content=self._state.files[path],
                provenance=ObservationProvenance(source_tool="read_file", source_ref=path),
            )
        if call.name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                return Observation(
                    content="content must be a string",
                    ok=False,
                    error_kind="model_invalid_action",
                    provenance=ObservationProvenance(source_tool="write_file", source_ref=path),
                )
            self._state.files[path] = content
            return Observation(
                content=f"wrote {len(content)} chars to {path}",
                provenance=ObservationProvenance(source_tool="write_file", source_ref=path),
            )
        raise EnvironmentError_(f"unhandled tool {call.name}")

    def snapshot(self) -> SnapshotRef:
        self._require_started()
        ref = _toy_snapshot("toy_file", self._state.to_json())
        self._store.put(ref, self._state.to_json())
        return ref

    def fork(self, snapshot: SnapshotRef) -> "ToyFileWorld":
        state_json = self._store.get(snapshot)
        if not self._store.digest_matches(snapshot, state_json):
            raise ProtocolViolation("snapshot digest mismatch on fork")
        child = ToyFileWorld(store=self._store)
        child._state = _ToyFileState(
            files=dict(state_json["files"]),  # type: ignore[arg-type]
            expected_out=str(state_json["expected_out"]),
            task_id=str(state_json["task_id"]),
            rng_state=int(state_json["rng_state"]),
            clock=int(state_json["clock"]),
        )
        child._started = True
        return child

    def state_digest(self) -> str:
        return digest_of(self._state.to_json() if self._started else {"unstarted": True})

    def close(self) -> None:
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "tool_schema_hash": digest_of([t.digest() for t in TOY_FILE_TOOLS]),
            "supports_shadow_fork": True,
            "synthetic": True,
        }

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("toy_file world not reset")

    # -- scorer ----------------------------------------------------------
    def final_artifact(self) -> ArtifactRef:
        out = self._state.files.get("out.txt", "")
        return ArtifactRef(ref_id=f"toy_file:{self._state.task_id}", path="out.txt", digest=digest_of(out))

    def private_expected(self) -> str:
        return self._state.expected_out


class ToyKVWorld:
    """Key-value store task world (domain ``toy_kv``) — a second, structurally
    different basic catalog on the same H0/runtime."""

    name = "toy_kv"
    version = "toy_kv_v1"
    supports_shadow_fork = True

    def __init__(self, store: _ToySnapshotStore | None = None):
        self._store = store or _ToySnapshotStore()
        self._state: dict[str, Any] = {"kv": {}, "expected": "", "task_id": "", "clock": 0}
        self._started = False

    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        meta = dict(public_task.metadata)
        kv = {str(k): str(v) for k, v in dict(meta.get("initial_kv") or {}).items()}
        # Expected answers stay on the evaluator side only.
        self._state = {
            "kv": kv,
            "expected": "",
            "task_id": public_task.task_id,
            "clock": 0,
            "rng_state": int(seed),
        }
        self._started = True
        return Observation(
            content=f"kv store ready with keys: {sorted(kv)}",
            provenance=ObservationProvenance(source_tool="reset", source_ref=public_task.task_id),
        )

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return TOY_KV_TOOLS

    def step(self, action: Action) -> StepResult:
        if not self._started:
            raise EnvironmentError_("toy_kv world not reset")
        self._state["clock"] += 1
        if isinstance(action, Finish):
            self._state["final_answer"] = action.answer or ""
            return StepResult(
                observation=Observation(content="finished"),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        return StepResult(
            observation=self._run_tool(action),
            done=False,
            snapshot=self.snapshot(),
            cost={"tool_calls": 1},
        )

    def _run_tool(self, call: ToolCall) -> Observation:
        if call.name not in {t.name for t in TOY_KV_TOOLS}:
            return Observation(
                content=f"unknown tool {call.name!r}",
                ok=False,
                error_kind="model_invalid_action",
                provenance=ObservationProvenance(source_tool=call.name, source_ref="schema"),
            )
        kv: dict[str, str] = self._state["kv"]
        args = dict(call.arguments)
        key = str(args.get("key", ""))
        if call.name == "put":
            val = args.get("value")
            if not isinstance(val, str):
                return Observation(content="value must be string", ok=False, error_kind="model_invalid_action")
            kv[key] = val
            return Observation(content=f"stored {key}", provenance=ObservationProvenance(source_tool="put", source_ref=key))
        if call.name == "get":
            if key not in kv:
                return Observation(content=f"missing key {key}", ok=False, error_kind="tool_error")
            return Observation(content=kv[key], provenance=ObservationProvenance(source_tool="get", source_ref=key))
        if call.name == "scan":
            return Observation(content=json.dumps(sorted(kv)), provenance=ObservationProvenance(source_tool="scan", source_ref="*"))
        if call.name == "delete":
            kv.pop(key, None)
            return Observation(content=f"deleted {key}", provenance=ObservationProvenance(source_tool="delete", source_ref=key))
        raise EnvironmentError_(f"unhandled tool {call.name}")

    def snapshot(self) -> SnapshotRef:
        ref = _toy_snapshot("toy_kv", self._state)
        self._store.put(ref, self._state)
        return ref

    def fork(self, snapshot: SnapshotRef) -> "ToyKVWorld":
        state = self._store.get(snapshot)
        if not self._store.digest_matches(snapshot, state):
            raise ProtocolViolation("snapshot digest mismatch on fork")
        child = ToyKVWorld(store=self._store)
        child._state = state
        child._started = True
        return child

    def state_digest(self) -> str:
        return digest_of(self._state if self._started else {"unstarted": True})

    def close(self) -> None:
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "tool_schema_hash": digest_of([t.digest() for t in TOY_KV_TOOLS]),
            "supports_shadow_fork": True,
            "synthetic": True,
        }

    def final_artifact(self) -> ArtifactRef:
        # For KV tasks the final answer *is* the artifact: its digest is what the
        # evaluator compares against the private expected answer.
        answer = str(self._state.get("final_answer", ""))
        return ArtifactRef(
            ref_id=f"toy_kv:{self._state['task_id']}",
            kind="answer",
            digest=digest_of(answer),
        )


# ---------------------------------------------------------------------------
# Task construction (deterministic; used by `data prepare` for toy domains)
# ---------------------------------------------------------------------------


def build_toy_file_tasks(n: int, seed: int, domain_prefix: str = "toy_file") -> list[tuple[PublicTask, str]]:
    """Deterministic arithmetic/text tasks over the virtual filesystem.

    Returns ``(public_task, expected_output)`` pairs.  The expected output is
    evaluator-side only and must never be placed into ``PublicTask`` metadata.
    """
    import random

    rng = random.Random(seed)
    tasks: list[tuple[PublicTask, str]] = []
    ops = ("sum", "max", "concat", "count_x")
    for i in range(n):
        op = ops[i % len(ops)]
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        word = "lorem"
        if op == "sum":
            files = {"a.txt": str(a), "b.txt": str(b)}
            expected = str(a + b)
            instr = "Compute the sum of the integers in a.txt and b.txt. Write the result to out.txt."
        elif op == "max":
            files = {"a.txt": str(a), "b.txt": str(b)}
            expected = str(max(a, b))
            instr = "Write the larger of the two integers in a.txt and b.txt to out.txt."
        elif op == "concat":
            sa, sb = f"alpha{a}", f"beta{b}"
            files = {"a.txt": sa, "b.txt": sb}
            expected = sa + sb
            instr = "Concatenate the text of a.txt and b.txt (a first) and write it to out.txt."
        else:
            hits = rng.randint(0, 3)
            files = {
                "a.txt": " ".join([word] * hits + ["x"]),
                "b.txt": " ".join([word] * (2 - hits) + ["y"]),
                "c.txt": "nothing here",
            }
            expected = str(hits)
            instr = f"Count how many files in the workspace contain the word '{word}' exactly as content tokens; write the count to out.txt."
        tid = f"{domain_prefix}-{seed}-{i:04d}"
        tasks.append(
            (
                PublicTask(
                    task_id=tid,
                    domain=domain_prefix,
                    instruction=instr,
                    group_id=f"{domain_prefix}-g{i // 4}",  # 4 tasks per group cluster
                    metadata={"files": files, "op": op},
                ),
                expected,
            )
        )
    return tasks


def build_toy_kv_tasks(n: int, seed: int) -> list[tuple[PublicTask, str]]:
    """KV tasks; returns ``(public_task, expected_answer)`` with the answer kept
    evaluator-side only."""
    import random

    rng = random.Random(seed + 77)
    tasks: list[tuple[PublicTask, str]] = []
    for i in range(n):
        k1, k2 = f"k{rng.randint(1, 5)}", f"k{rng.randint(6, 9)}"
        v1, v2 = f"v{rng.randint(10, 99)}", f"w{rng.randint(10, 99)}"
        mode = i % 2
        if mode == 0:
            instr = f"Move the value stored under {k1} to key 'out' using put, then finish with that value as the answer."
            expected = v1
        else:
            instr = f"Delete key {k2} if present, then finish with the value of {k1} as the answer."
            expected = v1
        tid = f"toy_kv-{seed}-{i:04d}"
        tasks.append(
            (
                PublicTask(
                    task_id=tid,
                    domain="toy_kv",
                    instruction=instr,
                    group_id=f"toy_kv-g{i // 3}",
                    metadata={"initial_kv": {k1: v1, k2: v2}},
                ),
                expected,
            )
        )
    return tasks


class ToyScorer:
    """Official-style scorer for toy worlds (metric on [0,1], synthetic=True).

    Compares the produced artifact digest against the private expected-content
    digest held here on the evaluator side (``PrivateTaskRef`` never carries it).
    """

    scorer_version = "toy_scorer_v1"

    def __init__(self, expected: Mapping[str, str]):
        self._expected = dict(expected)

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        task_id = private_task_ref.ref_id
        if task_id not in self._expected:
            raise ProtocolViolation(f"no private expectation for {task_id}")
        expected = self._expected[task_id]
        expected_digest = digest_of(expected)
        ok = final_artifact.digest == expected_digest
        return ScoreRecord(
            task_id=task_id,
            domain=private_task_ref.domain,
            metric_name="task_success",
            value=1.0 if ok else 0.0,
            success=ok,
            details={"expected_digest": expected_digest, "produced_digest": final_artifact.digest},
            synthetic=True,
        )


def score_text(expected: str, produced: str, *, task_id: str = "", domain: str = "toy") -> ScoreRecord:
    """Direct text-compare scoring used by the toy evaluators."""
    ok = produced.strip() == expected.strip()
    return ScoreRecord(
        task_id=task_id,
        domain=domain,
        metric_name="task_success",
        value=1.0 if ok else 0.0,
        success=ok,
        details={"expected_digest": digest_of(expected), "produced_digest": digest_of(produced)},
        synthetic=True,
    )
