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

import hashlib
import json
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

    def __init__(self, data_root: str | None = None, *, max_interactions: int = 40, out_root: str | None = None):
        _require_appworld()
        self._data_root = data_root or os.environ.get("APPWORLD_ROOT", "/mnt/songzijun/datasets/appworld")
        # Trajectory persistence (paired-v3 trace gap): the adapter kept only
        # the in-memory replay log — final_artifact was a bare reference with
        # no file, so appworld had no post-hoc trace samples. When out_root is
        # set (RSI_APPWORLD_OUT), final_artifact() writes a bounded JSON
        # trajectory next to the ref (same shape as alfworld/webshop).
        self._out_root = Path(out_root or os.environ.get("RSI_APPWORLD_OUT") or "")
        self._trajectory: list[dict[str, Any]] = []
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
        # PR-2 (2026-11-03 audit §4.1): the task-scoped AppWorld constructor
        # already initializes (its ``initialize=True`` default path calls
        # initialize() -> close_all() + preamble + _set_datetime + _save_state).
        # An explicit second initialize() here re-ran close_all() over every
        # live world and re-executed the preamble — do NOT call it again.
        self._seed = int(seed)
        self._started = True
        self._trajectory = [{
            "kind": "reset",
            "task_id": self._task_id,
            "experiment": self._experiment,
            "instruction_head": str(public_task.instruction)[:400],
        }]
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
            # PR (2026-11-05 bad-case audit): the official evaluator reads the
            # predicted answer from the supervisor Task.answer field, which is
            # only written by ``apis.supervisor.complete_task(answer=...)``.
            # This branch used to ignore ``Finish.answer`` entirely, so every
            # scored task read the initial "<<NOT_GIVEN>>" placeholder and the
            # answer assertion failed 53/53 regardless of the model's work.
            # Submit the actor's answer on its behalf (answer=None is a valid
            # submission — the GT for action-less tasks is JSON null).
            submit_note = ""
            answer = getattr(action, "answer", None)
            submitted = False
            # Always submit on finish: answer=None is a meaningful submission
            # ("task asks no answer" — the API default marks the task success
            # with JSON null), matching GT tasks whose answer.json is null.
            import json as _json

            # complete_task stores json.dumps(answer); the evaluator does
            # json.loads on it. The submitted code must decode to the SAME
            # typed value: inline the JSON literal directly (json.loads('null')
            # → None) instead of double-encoding, which turned None into the
            # string 'null' (r3 lesson: json.loads('"null"') is not None).
            try:
                literal = _json.dumps(answer)
            except (TypeError, ValueError):
                literal = _json.dumps(str(answer))
            try:
                submit_code = f"apis.supervisor.complete_task(answer=json.loads({literal!r}))"
                result = self._env.execute(submit_code)
                text = str(result)
                # r3 lesson: AppWorld's safety_guard returns "Execution failed.
                # Traceback: …" instead of raising — string-match it too.
                failed = ("EXEC_ERROR" in text) or ("Execution failed" in text) or ("Traceback" in text)
                # Record in the replay log so forks reproduce the answer
                # submission (same contract as python_exec actions).
                self._replay_code.append(submit_code)
                submitted = not failed
                submit_note = f" answer_submitted={literal}" if submitted else f" answer_submit_failed: {text}"
            except Exception as e:  # noqa: BLE001 - submit failure must not mask the finish
                submit_note = f" answer_submit_error: {e!r}"
            completed = bool(self._env.task_completed())
            self._trajectory.append({
                "kind": "finish",
                "answer_head": (literal or "")[:200],
                "submitted": submitted,
                "task_completed": completed,
            })
            return StepResult(
                observation=Observation(
                    content=f"finished (task_completed={completed}{submit_note})",
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
        self._trajectory.append({
            "kind": "tool_call",
            "tool": "python_exec",
            "code_head": str(code)[:400],
            "obs_head": str(output)[:400],
            "obs_len": len(str(output)),
        })
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
        # PR-2: constructor already initialized the child (see reset()). The
        # old explicit child._env.initialize() here called close_all() again,
        # which can tear down the base env's live handles too — the exact
        # fork-correctness hazard the docstring warns about.
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
        # PR-6 (audit §9.4): fork correctness requires the BASE to still work
        # after the child exists — the official constructor/initialize path
        # calls close_all() process-globally, so the child's construction can
        # invalidate the base's live handles even without an explicit
        # load_state.  Prove the base survived by executing one legal no-op
        # action on it; a failure here is a hard error, not a silent poison.
        try:
            probe = self._env.execute("pass")
            del probe
        except Exception as e:  # noqa: BLE001
            raise EnvironmentError_(
                f"fork broke the base environment (probe execute failed): {e!r}"
            ) from e
        return child

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        # PR-6 (audit §9.4): the digest must describe the BUSINESS state, not
        # the bookkeeping identity.  Including the experiment name (which
        # embeds run/arm/task naming and fork counters) made semantically
        # identical states hash differently across arms — breaking snapshot
        # comparisons.  Hash the replay prefix (the deterministic execution
        # history that defines the state) plus a checkpoint-files digest.
        ckpt_digest = ""
        try:
            latest = self._env.save_state("digestcheck")
            ckpt_digest = self._checkpoint_digest(str(latest))
        except Exception:  # noqa: BLE001 - digest is best-effort telemetry
            ckpt_digest = ""
        return digest_of(
            {
                "task": self._task_id,
                "replay": digest_of(self._replay_code),
                "checkpoint": ckpt_digest,
            }
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
        record = {
            "task_id": self._task_id,
            "experiment": self._experiment,
            "trajectory": self._trajectory,
            "replay_code": list(self._replay_code),
        }
        path: str | None = None
        if self._out_root:
            try:
                out_dir = Path(self._out_root) / (
                    self._task_id.replace(":", "_").replace("/", "_") or "task"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                fp = out_dir / "final_artifact.json"
                fp.write_text(
                    json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), "utf-8"
                )
                path = str(fp)
            except OSError:
                path = None  # best-effort: scoring does not depend on the file
        return ArtifactRef(
            ref_id=f"appworld:{self._task_id}:{self._experiment}",
            kind="env_state",
            digest=digest_of({"task": self._task_id, "exp": self._experiment}),
            path=path,
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("appworld adapter not reset")


class AppWorldScorer:
    """Official AppWorld scoring on the evaluator side, without a new world.

    PR-2 of the 2026-11-03 measurement audit: the previous implementation
    constructed a task-scoped ``AppWorld`` for the SAME (task_id, experiment)
    the actor had used and then called ``initialize()`` on it.  The official
    constructor already initializes (``initialize=True`` path) and
    ``initialize()`` itself does ``close_all()`` + ``_save_state`` — so scoring
    re-ran the task preamble over the actor's final output DB and could reset
    or repoint the very state being graded (the 318-pair identical-score
    anomaly's prime suspect).

    The official offline entrypoint is ``appworld.evaluator.evaluate_task``:
    it loads the task, freezes the task datetime, reads the experiment's
    output DB from disk, runs the ground-truth tests, and returns a
    ``TestTracker`` — no ``AppWorld`` construction, no ``close_all``, no write
    to the business DB (the only writes are the evaluation report files under
    ``tasks/<id>/evaluation/``, which are evaluator-owned).

    Metric semantics (appworld_strict_v2):
      - ``success``        = ``tracker.success`` (ALL tests pass) — the
        official TGC aggregates the MEAN of this, not the pass ratio.
      - ``value``          = ``pass_percentage / 100`` — kept as the
        auxiliary partial credit field, NOT aggregated as TGC.
    """

    scorer_version = f"appworld-strict-v2-{APPWORLD_VERSION}"

    # metric names: primary is strict success; partial is auxiliary only.
    METRIC_STRICT = "appworld_strict_success"
    METRIC_PARTIAL = "appworld_partial_test_pass_rate"

    def __init__(self, data_root: str | None = None):
        self._data_root = data_root or os.environ.get("APPWORLD_ROOT", "/mnt/songzijun/datasets/appworld")

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        parts = (final_artifact.ref_id or "").split(":")
        task_id, experiment = (parts[1], parts[2]) if len(parts) >= 3 else (private_task_ref.ref_id, "rsi")
        import appworld as _appworld_pkg

        _appworld_pkg.update_root(self._data_root)
        from appworld.common.path_store import path_store  # reloaded vs new root
        from appworld.evaluator import evaluate_task

        # Verify the actor actually persisted an output DB for this experiment
        # before scoring; a missing output dir is an EVAL_INVALID measurement,
        # not a 0-score "method failure".
        output_db = Path(path_store.experiment_outputs) / experiment / "tasks" / task_id / "dbs"
        if not output_db.is_dir():
            return ScoreRecord(
                task_id=private_task_ref.ref_id,
                domain="appworld",
                metric_name=self.METRIC_STRICT,
                value=0.0,
                success=False,
                details={
                    "measurement_status": "EVAL_INVALID",
                    "error_kind": "missing_output_db",
                    "output_db": str(output_db),
                    "experiment": experiment,
                },
                scorer_version=self.scorer_version,
                synthetic=False,
            )
        # Hash the business DB before/after scoring: the evaluator must not
        # mutate actor state (report files excluded — evaluator-owned).
        pre_hash = self._db_tree_digest(output_db)
        try:
            tracker = evaluate_task(
                task_id=task_id,
                experiment_name=experiment,
                suppress_errors=True,
                save_report=True,
            )
        except Exception as e:  # noqa: BLE001 - scorer failure is a measurement failure
            return ScoreRecord(
                task_id=private_task_ref.ref_id,
                domain="appworld",
                metric_name=self.METRIC_STRICT,
                value=0.0,
                success=False,
                details={
                    "measurement_status": "EVAL_INVALID",
                    "error_kind": "scorer_exception",
                    "error": repr(e),
                    "experiment": experiment,
                },
                scorer_version=self.scorer_version,
                synthetic=False,
            )
        post_hash = self._db_tree_digest(output_db)
        num_tests = int(getattr(tracker, "num_tests", 0) or 0)
        pass_count = int(getattr(tracker, "pass_count", 0) or 0)
        pct = float(getattr(tracker, "pass_percentage", 0.0) or 0.0)
        strict_success = bool(getattr(tracker, "success", False))
        # PR (2026-11-05, r4 plateau post-mortem): the evolution signal used
        # to read the strict 0/1 binary — with a 9B actor whose strict rate
        # is 0/16, every candidate scored 0.0 forever and PLATEAU was a
        # mathematical certainty (3gwab r3, v3hvt r4: 8 candidates all 0.0).
        # The partial credit (official TGC semantics, H0_v2 baseline 34.74%)
        # was computed into details and then dropped by every consumer.
        # value is now the continuous partial rate; strict stays in
        # ``success`` for confirm-gate semantics.
        return ScoreRecord(
            task_id=private_task_ref.ref_id,
            domain="appworld",
            metric_name=self.METRIC_PARTIAL,
            value=max(0.0, min(1.0, pct / 100.0)),
            success=strict_success,
            details={
                "measurement_status": "VALID",
                "metric_version": "appworld_partial_v1",
                "strict_success": strict_success,
                "partial_test_pass_rate": max(0.0, min(1.0, pct / 100.0)),
                "pass_percentage": pct,
                "pass_count": pass_count,
                "num_tests": num_tests,
                "experiment": experiment,
                "db_unchanged_after_scoring": pre_hash == post_hash,
                "db_digest": post_hash,
            },
            scorer_version=self.scorer_version,
            synthetic=False,
        )

    @staticmethod
    def _db_tree_digest(base: Path) -> str:
        """Digest over the business DB files only (exclude evaluator reports)."""
        h = hashlib.sha256()
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            if rel.startswith("evaluation"):
                continue  # evaluator-owned report path, not actor state
            h.update(rel.encode())
            h.update(digest_of_file(str(f)).encode())
        return h.hexdigest()
