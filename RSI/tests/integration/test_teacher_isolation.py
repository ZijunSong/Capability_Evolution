"""test_teacher_isolation / test_hidden_grader_inaccessible /
test_cross_environment_core / test_split_group_disjointness /
test_unimplemented_fails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.errors import NotImplementedCapability, ProtocolViolation
from rsi.contracts.events import Event, EventKind, Visibility
from rsi.contracts.task import SplitAssignment, SplitManifest
from rsi.environments.toy import ToyFileWorld, ToyKVWorld, build_toy_file_tasks, build_toy_kv_tasks
from rsi.harness.loader import load_bank, load_empty_bank
from rsi.opd.teacher import TeacherForkRunner
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop
from rsi.providers.scripted import ScriptedPolicy

from helpers import toy_finalizer_patch  # noqa: E402


class WriterPlan:
    """Plan that writes to the environment aggressively (for isolation test)."""

    def __call__(self, view_text: str, step: int):
        if step == 0:
            return ToolCall(name="write_file", arguments={"path": "pwned.txt", "content": "x"})
        return Finish(answer=None)


def test_teacher_isolation(tmp_path):
    """Teacher fork writes files / triggers tools; Student state is unchanged."""
    empty = load_empty_bank()
    patch = toy_finalizer_patch(empty.bank_digest)
    from rsi.evolution.materializer import materialize_candidate

    mat = materialize_candidate(
        patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=tmp_path
    )
    bank = load_bank(mat.bank_dir)

    task, _ = build_toy_file_tasks(1, 17)[0]
    env = ToyFileWorld()
    env.reset(task, seed=1)
    pre_digest = env.state_digest()

    runner = TeacherForkRunner(
        env_factory=ToyFileWorld,
        policy=ScriptedPolicy(WriterPlan()),
        bank=bank,  # Teacher runs H0 + Bk (finalizer may call tools too)
        budgets=BudgetSpec(max_actions=8, max_tool_calls=20),
        max_shadow_events=6,
    )
    snap = env.snapshot()
    branch = runner.run_branch(
        base_env=env,
        snapshot_ref=snap,
        task=task,
        student_events=(),
        decision_index=0,
    )
    # Teacher branch did perform writes in its fork...
    assert any(
        e.action is not None and getattr(e.action, "name", "") == "write_file" for e in branch.events
    )
    # ...but the Student environment digest is unchanged (run_branch asserts too)
    assert env.state_digest() == pre_digest
    # and the branch events are private
    assert all(e.visibility == Visibility.TEACHER_PRIVATE for e in branch.events)
    env.close()


def test_hidden_grader_inaccessible():
    """Private expectations never enter PublicTask metadata or Actor views."""
    tasks = build_toy_file_tasks(4, 17)
    for task, expected in tasks:
        # No answer field of any kind in the public metadata; digit coincidence
        # with input files is legitimate (e.g. max task) and not a leak.
        assert "expected_out" not in task.metadata
        assert "expected_answer" not in task.metadata
        assert "answer" not in task.metadata
        assert "gold" not in task.metadata
        assert "solution" not in task.metadata
    # Candidate-side sandbox refuses env mutation and score access paths are
    # outside the editable scope (validator), while the scorer keeps the map.
    from rsi.evolution.validator import validate_candidate_files

    bad = validate_candidate_files(
        files={"components/x.py": "value = 1\n"},
        manifest_patch={"id": "x", "entrypoint": "components.x:X", "version": "1", "files": ["components/x.py"]},
        known_task_ids=["toy_file-17-0000"],
    )
    # harmless component passes
    assert bad.ok, bad.problems
    leak = validate_candidate_files(
        files={"components/x.py": "ans = 'toy_file-17-0000'\n"},
        manifest_patch={},
        known_task_ids=["toy_file-17-0000"],
    )
    assert not leak.ok  # task-id constant in code is refused


def test_cross_environment_core():
    """Same H0 loop + same empty bank over two different basic catalogs."""
    bank = load_empty_bank()
    policy = ScriptedPolicy(lambda view, step: Finish(answer="done"))
    for factory, builder in ((ToyFileWorld, build_toy_file_tasks), (ToyKVWorld, build_toy_kv_tasks)):
        task = builder(1, 17)[0][0]
        env = factory()
        loop = MinimalLoop(env=env, policy=policy, bank=bank, budgets=BudgetSpec(max_actions=4))
        res = loop.run_episode(task, seed=1)
        assert res.trace.terminated_reason == "finish"
        # no benchmark-name routing: the core only saw tool specs
        names = {t.name for t in env.tool_catalog()}
        assert names in ({"read_file", "write_file", "list_dir"}, {"put", "get", "scan", "delete"})
        env.close()


def test_split_group_disjointness():
    groups = {"g0": ("t0", "t1"), "g1": ("t2", "t3")}
    good = SplitManifest(
        manifest_id="m",
        domain="toy",
        assignments=(
            SplitAssignment(role="work_train", task_ids=("t0", "t1")),
            SplitAssignment(role="confirm_h", task_ids=("t2", "t3"), round_index=0),
        ),
        groups=groups,
    )
    good.check_group_disjoint()
    bad = SplitManifest(
        manifest_id="m",
        domain="toy",
        assignments=(
            SplitAssignment(role="work_train", task_ids=("t0",)),
            SplitAssignment(role="confirm_w", task_ids=("t1",), round_index=0),
        ),
        groups=groups,
    )
    with pytest.raises(ValueError):
        bad.check_group_disjoint()


def test_unimplemented_fails(tmp_path):
    """Missing capability fails loudly — no silent fallback."""
    from rsi.environments.registry import global_registry
    from rsi.training.fsdp2_backend import FSDP2Backend
    from rsi.providers.tiny_hf import build_tiny_model

    registry = global_registry()
    with pytest.raises(NotImplementedCapability):
        registry.resolve("appworld")  # optional dep missing -> loud failure
    with pytest.raises(NotImplementedCapability):
        FSDP2Backend(model=build_tiny_model())  # no process group -> loud failure
    # backend that cannot consume OPD must not silently degrade
    from rsi.training.joint import joint_update

    class NoOPDBackend:
        name = "fake"
        consumes_opd = False

        def zero_grad(self):
            pass

        def optimizer_step(self):
            return {}

    with pytest.raises(NotImplementedCapability):
        joint_update(
            NoOPDBackend(),
            rl_rows=[],
            opd_rows=[],
            spec=MethodSpec_resolve(),
        )


def MethodSpec_resolve():
    from rsi.training.joint import MethodSpec

    return MethodSpec.resolve("harness_opd_rl")
