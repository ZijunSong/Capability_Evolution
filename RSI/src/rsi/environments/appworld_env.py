"""AppWorld adapter — application interaction / multi-step tool tasks.

Official source: https://github.com/StonyBrookNLP/appworld (package 0.1.3,
data bundle 0.1.0, frozen at ``data prepare``).  Official TGC (per-task goal
completion, ``TestTracker.pass_percentage``) is the task metric; scenario-level
aggregation lives in ``rsi.evaluation.aggregation``.

Privacy boundary: the Actor-side adapter runs with ``load_ground_truth=False``;
the scorer constructs its own evaluator-side instance with ground truth loaded
against the *same* experiment output DBs.  ``PublicTask`` carries only the
instruction; goals/tests never enter the actor process.

Snapshot/fork: AppWorld exposes ``save_state``/``load_state`` — full world
snapshot (app DBs, interaction state), so ``supports_shadow_fork=True``.

Basic tools are the environment's own capability surface: interactive Python
execution against the app APIs (``python_exec``) plus finish.  No planning or
domain strategy ships in the adapter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue, digest_of, digest_of_file
from rsi.contracts.errors import EnvironmentError_, NotImplementedCapability
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

APPWORLD_TOOL_SPEC: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="python_exec",
        description=(
            "Execute a Python snippet inside the AppWorld runtime with access to "
            "the app APIs (import appworld / use api objects). Returns stdout/stderr."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        capabilities=("python_exec", "app_api"),
    ),
)

APPWORLD_VERSION = "appworld-0.1.3+data-0.1.0"


def _require_appworld() -> Any:
    try:
        import appworld  # noqa: F401
    except ImportError as e:
        raise NotImplementedCapability(
            "AppWorld adapter requires the optional 'appworld' package "
            "(pip install -e '.[appworld]'). No fallback environment is used."
        ) from e
    return appworld


class AppWorldAdapter:
    """EnvironmentAdapter over the official AppWorld environment."""

    name = "appworld"
    version = APPWORLD_VERSION
    supports_shadow_fork = True

    def __init__(self, data_root: str | None = None, *, max_interactions: int = 40):
        _require_appworld()
        self._data_root = data_root or os.environ.get("APPWORLD_ROOT", "/mnt/songzijun/datasets/appworld")
        data_dir = os.path.join(self._data_root, "data")
        # Require real data content — an empty `data/` dir is not an install.
        if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
            raise NotImplementedCapability(
                f"AppWorld data not found (or empty) under {data_dir} — run "
                "`appworld download data --root <root>` first (frozen via data prepare)"
            )
        # AppWorld resolves its root through APPWORLD_ROOT at path-store time.
        os.environ["APPWORLD_ROOT"] = self._data_root
        self._max_interactions = int(max_interactions)
        self._env: Any = None
        self._task_id: str = ""
        self._experiment: str = "rsi"
        self._snap_counter = 0
        self._started = False
        # Replay log for replay-prefix forks (task book §5.2): the world is
        # deterministic (frozen clock + seeded RNG), so re-executing the
        # recorded code in a fresh env reproduces the state; verified by
        # checkpoint-content digest.
        self._replay_code: list[str] = []
        self._seed: int = 0

    def check_ready(self) -> tuple[bool, str]:
        """Panel-level capability probe: the apps code bundle must be unpacked
        (``appworld install``); a bare pip install ships it encrypted."""
        try:
            import appworld.apps.admin  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"appworld apps bundle not unpacked ({e!r}); run `appworld install`"
        return True, "ok"

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        from appworld.environment import AppWorld

        self._task_id = str(public_task.metadata.get("appworld_task_id") or public_task.task_id)
        # Colon-free experiment name: appworld's checkpoint writer silently
        # fails to persist for experiment names containing ':' (load then
        # raises).  Sanitize rather than patch vendored code.
        safe_task = self._task_id.replace(":", "_")
        self._experiment = f"rsi-{safe_task}-{seed}"
        # Actor side: ground truth NEVER loaded here.
        self._env = AppWorld(
            task_id=self._task_id,
            experiment_name=self._experiment,
            max_interactions=self._max_interactions,
            load_ground_truth=False,
            random_seed=int(seed),
        )
        self._env.initialize()
        self._seed = int(seed)
        self._started = True
        return Observation(
            content=(
                f"AppWorld task ready. Use python_exec to interact with the app APIs. "
                f"Task: {public_task.instruction}"
            ),
            provenance=ObservationProvenance(source_tool="reset", source_ref=self._task_id),
        )

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return APPWORLD_TOOL_SPEC

    def step(self, action: Action) -> StepResult:
        self._require_started()
        if isinstance(action, Finish):
            completed = bool(self._env.task_completed())
            return StepResult(
                observation=Observation(
                    content=f"finished (task_completed={completed})",
                    provenance=ObservationProvenance(source_tool="finish", source_ref=self._task_id),
                ),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        if action.name != "python_exec":
            return StepResult(
                observation=Observation(
                    content=f"unknown tool {action.name!r}",
                    ok=False,
                    error_kind="model_invalid_action",
                ),
                done=False,
            )
        code = action.arguments.get("code")
        if not isinstance(code, str):
            return StepResult(
                observation=Observation(content="code must be a string", ok=False, error_kind="model_invalid_action"),
                done=False,
            )
        try:
            output = self._env.execute(code)
            self._replay_code.append(code)
        except Exception as e:  # noqa: BLE001 - execution errors are observations
            output = f"EXEC_ERROR: {e!r}"
        return StepResult(
            observation=Observation(
                content=str(output),
                provenance=ObservationProvenance(source_tool="python_exec", source_ref=self._experiment),
            ),
            done=False,
            snapshot=self.snapshot(),
            cost={"tool_calls": 1},
        )

    def snapshot(self) -> SnapshotRef:
        """Checkpoint content digest + replay position.

        NOTE: the vendor ``AppWorld.load_state`` calls ``close_all()`` first,
        invalidating every other live instance — so forks do NOT use
        ``load_state`` on a live base env.  ``fork`` is replay-based instead
        (task book §5.2: replay the legal prefix and verify the digest).
        """
        self._require_started()
        self._snap_counter += 1
        state_id = self._env.save_state(f"snap{self._snap_counter}")
        digest = self._checkpoint_digest(str(state_id))
        return SnapshotRef(
            ref_id=str(state_id),
            digest=digest,
            kind="replay_prefix",
        )

    def _checkpoint_digest(self, state_id: str) -> str:
        base = (
            Path(self._data_root) / "experiments" / "outputs" / self._experiment
            / "tasks" / self._task_id / "checkpoints" / state_id
        )
        files = {}
        if base.is_dir():
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    files[str(f.relative_to(base))] = digest_of_file(str(f))
        return digest_of({"replay": digest_of(self._replay_code), "checkpoint": files})

    def fork(self, snapshot: SnapshotRef) -> "AppWorldAdapter":
        """Replay-prefix fork: a fresh env re-executes the recorded code.

        The base environment stays fully live (no ``load_state``/``close_all``).
        Reproduction is verified against the snapshot digest; a mismatch is a
        hard error (the task book requires digest verification).
        """
        from appworld.environment import AppWorld

        child = AppWorldAdapter(self._data_root, max_interactions=self._max_interactions)
        child._task_id = self._task_id
        child._seed = self._seed
        child._experiment = f"{self._experiment}-fork{self._snap_counter}"
        child._env = AppWorld(
            task_id=self._task_id,
            experiment_name=child._experiment,
            max_interactions=self._max_interactions,
            load_ground_truth=False,
            random_seed=self._seed,
        )
        child._env.initialize()
        for code in self._replay_code:
            child._env.execute(code)
            child._replay_code.append(code)
        # verify the replayed state reproduces the snapshot
        child._snap_counter = self._snap_counter
        state_id = child._env.save_state("forkcheck")
        actual = child._checkpoint_digest(str(state_id))
        expected_replay = digest_of(self._replay_code)
        if digest_of({"replay": expected_replay}) != digest_of({"replay": digest_of(child._replay_code)}):
            raise EnvironmentError_("fork replay log mismatch")
        if snapshot.kind == "replay_prefix" and snapshot.digest:
            # digest encodes both replay and checkpoint files; the replay part
            # must match exactly — that is the determinism guarantee.
            if snapshot.digest != actual:
                raise EnvironmentError_(
                    "fork replay digest mismatch: state not reproduced "
                    f"({snapshot.digest[:12]} != {actual[:12]})"
                )
        child._started = True
        return child

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        return digest_of(
            {"task": self._task_id, "exp": self._experiment, "replay": digest_of(self._replay_code)}
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "data_root": self._data_root,
            "tool_schema_hash": digest_of([t.digest() for t in APPWORLD_TOOL_SPEC]),
            "supports_shadow_fork": True,
            "synthetic": False,
        }

    def final_artifact(self) -> ArtifactRef:
        return ArtifactRef(
            ref_id=f"appworld:{self._task_id}:{self._experiment}",
            kind="env_state",
            digest=digest_of({"task": self._task_id, "exp": self._experiment}),
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("appworld adapter not reset")


class AppWorldScorer:
    """Official AppWorld evaluation (TGC) on the evaluator side.

    Ground truth is loaded only here — a fresh ``AppWorld`` instance with
    ``load_ground_truth=True`` scores the actor's experiment outputs without
    ever exposing goals/tests to the actor process.
    """

    scorer_version = f"appworld-tgc-{APPWORLD_VERSION}"

    def __init__(self, data_root: str | None = None):
        self._data_root = data_root or os.environ.get("APPWORLD_ROOT", "/mnt/songzijun/datasets/appworld")

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        parts = (final_artifact.ref_id or "").split(":")
        task_id, experiment = (parts[1], parts[2]) if len(parts) >= 3 else (private_task_ref.ref_id, "rsi")
        os.environ["APPWORLD_ROOT"] = self._data_root
        from appworld.environment import AppWorld

        scorer_env = AppWorld(
            task_id=task_id,
            experiment_name=experiment,
            load_ground_truth=True,
            ground_truth_mode="minimal",
            raise_on_failure=False,
        )
        scorer_env.initialize()
        try:
            tracker = scorer_env.evaluate(suppress_errors=True)
            pct = float(getattr(tracker, "pass_percentage", 0.0) or 0.0)
            value = max(0.0, min(1.0, pct / 100.0))
            details = {
                "pass_percentage": pct,
                "pass_count": int(getattr(tracker, "pass_count", 0) or 0),
                "total_count": int(getattr(tracker, "total_count", 0) or 0),
                "success": bool(getattr(tracker, "success", False)),
            }
        finally:
            try:
                scorer_env.close()
            except Exception:  # noqa: BLE001
                pass
        return ScoreRecord(
            task_id=private_task_ref.ref_id,
            domain="appworld",
            metric_name="task_goal_completion",
            value=value,
            success=details.get("success", value >= 1.0),
            details=details,
            scorer_version=self.scorer_version,
            synthetic=False,
        )
