"""test_empty_bank_identity / test_real_code_materialization /
test_parent_feedback_continuity / test_prune_semantics."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.artifact import CandidatePatch
from rsi.evolution.materializer import materialize_candidate
from rsi.evolution.packet import TraceFragment, bank_source_files, build_packet
from rsi.evolution.diagnosis import diagnose_traces
from rsi.evolution.validator import validate_materialized
from rsi.harness.loader import load_bank, load_empty_bank
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop
from rsi.environments.toy import ToyFileWorld, build_toy_file_tasks
from rsi.providers.scripted import ScriptedPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import toy_file_solver_plan, toy_finalizer_patch  # noqa: E402

_OBS = re.compile(r"\[\d+\] observation: (.*)")


def reader_only_plan(view_text: str, step: int):
    """Reads a.txt and b.txt then finishes WITHOUT writing out.txt."""
    obs = _OBS.findall(view_text)[1:]
    n = len(obs)
    if n == 0:
        return ToolCall(name="read_file", arguments={"path": "a.txt"})
    if n == 1:
        return ToolCall(name="read_file", arguments={"path": "b.txt"})
    return Finish(answer=None)


def _run(policy, bank, task):
    env = ToyFileWorld()
    loop = MinimalLoop(env=env, policy=policy, bank=bank, budgets=BudgetSpec(max_actions=6))
    return loop.run_episode(task, seed=3)


def test_empty_bank_identity():
    """minimal (in-memory) and on-disk empty bank produce identical behavior."""
    disk = load_bank(Path(__file__).resolve().parents[2] / "banks" / "empty")
    mem = load_empty_bank()
    assert disk.is_empty and mem.is_empty
    task = build_toy_file_tasks(1, 17)[0][0]
    p = ScriptedPolicy(toy_file_solver_plan)
    r1 = _run(p, mem, task)
    r2 = _run(p, disk, task)
    # identical rendered prompt ids
    assert [d.student_prompt_token_ids for d in r1.trace.decisions] == [
        d.student_prompt_token_ids for d in r2.trace.decisions
    ]
    # identical tool set / observations
    assert [e.to_json()["observation"] for e in r1.trace.events] == [
        e.to_json()["observation"] for e in r2.trace.events
    ]
    # no extra calls for either
    assert sum(e.cost.extra_llm_calls for e in r1.trace.events) == 0
    assert sum(e.cost.extra_llm_calls for e in r2.trace.events) == 0
    assert mem.bank_digest == disk.bank_digest or mem.is_empty and disk.is_empty


def test_real_code_materialization(tmp_path):
    """Proposer-fixture code lands on disk byte-for-byte and is executed."""
    empty = load_empty_bank()
    patch = toy_finalizer_patch(empty.bank_digest)
    mat = materialize_candidate(
        patch,
        parent_bank_root=empty.root,
        parent_bank_digest=empty.bank_digest,
        out_root=tmp_path,
    )
    on_disk = (mat.bank_dir / "components" / "toy_finalizer.py").read_text("utf-8")
    assert on_disk == patch.files[0].content  # no template substitution
    report = validate_materialized(mat.bank_dir, known_task_ids=["toy_file-17-0000"])
    assert report.ok, report.problems

    # Execution proof: the new code runs and produces its effect.
    task, expected = build_toy_file_tasks(1, 17)[0]
    bank = load_bank(mat.bank_dir)
    res = _run(ScriptedPolicy(toy_file_solver_plan), bank, task)
    assert res.trace.terminated_reason == "finish"
    assert res.final_artifact is not None and res.final_artifact.digest is not None
    from rsi.environments.toy import digest_of

    # solver_plan writes out.txt itself; prove the component path separately:
    res_reader = _run(ScriptedPolicy(reader_only_plan), bank, task)
    # reader-only + finalizer bank => out.txt synthesized by the component
    assert res_reader.final_artifact.digest == digest_of(expected)
    assert bank.bank_digest != empty.bank_digest


def test_parent_feedback_continuity(tmp_path):
    """Second generation packet contains the parent's real source and new results."""
    empty = load_empty_bank()
    patch = toy_finalizer_patch(empty.bank_digest)
    mat = materialize_candidate(
        patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=tmp_path
    )
    parent = load_bank(mat.bank_dir)
    sources = bank_source_files(parent)
    assert "components/toy_finalizer.py" in sources
    assert "ToyFinalizer" in sources["components/toy_finalizer.py"]

    diagnosis = diagnose_traces(
        generation=1,
        per_task={"t1": {"value": 0.0, "terminated_reason": "tool_error"}},
    )
    packet = build_packet(
        generation=1,
        policy_digest="p",
        parent_bank=parent,
        baseline_scores={"toy_file": {"t1": 0.0}},
        parent_scores={"toy_file": {"t1": 1.0}},
        diagnosis=diagnosis,
        failure_fragments=(TraceFragment(task_id="t1", outcome="fail", events_text="...", decisions=3),),
    )
    assert packet.parent_files == sources
    assert packet.generation == 1
    assert packet.diagnosis.failure_categories[0].kind == "tool_error"
    assert packet.paired_summary["toy_file"]["mean_delta"] == 1.0


def test_prune_semantics(tmp_path):
    """prune removes the component: manifest drops it and its effects disappear."""
    empty = load_empty_bank()
    add = toy_finalizer_patch(empty.bank_digest)
    mat = materialize_candidate(
        add, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=tmp_path
    )
    parent = load_bank(mat.bank_dir)
    assert len(parent.components) == 1

    prune = CandidatePatch(
        parent_bank_digest=parent.bank_digest,
        operation="prune",
        hypothesis="remove finalizer",
        delete_paths=("components/toy_finalizer.py",),
    )
    mat2 = materialize_candidate(
        prune, parent_bank_root=parent.root, parent_bank_digest=parent.bank_digest, out_root=tmp_path
    )
    pruned = load_bank(mat2.bank_dir)
    assert pruned.manifest.components == ()
    assert not (mat2.bank_dir / "components" / "toy_finalizer.py").exists()

    # Effects and cost gone: reader-only policy under pruned bank == bare H0.
    task, expected = build_toy_file_tasks(1, 17)[0]
    from rsi.environments.toy import digest_of

    res_pruned = _run(ScriptedPolicy(reader_only_plan), pruned, task)
    res_bare = _run(ScriptedPolicy(reader_only_plan), load_empty_bank(), task)
    assert res_pruned.final_artifact.digest != digest_of(expected)  # no synthesis
    assert res_pruned.final_artifact.digest == res_bare.final_artifact.digest
