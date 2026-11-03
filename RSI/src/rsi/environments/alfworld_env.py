"""ALFWorld adapter — household text tasks over the official TextWorld games.

Official source: https://github.com/alfworld/alfworld (package frozen at
``data prepare`` via ``alfworld_src_commit`` in the frozen manifest).  Official
metric: per-task success (``state['won']``) aggregated to success rate over the
panel — the same completion check the official ``scripts/run_eval.py`` reports.

Data (``RSI_ALFWORLD_ROOT``, default ``/mnt/songzijun/datasets/alfworld``):
the official ``$ALFWORLD_DATA`` layout produced by ``scripts/prepare_alfworld_data.py`` —
``json_2.1.1/<split>/<task>/trial_*/{traj_data.json,game.tw-pddl}`` (co-located
as required by ``AlfredTWEnv.collect_game_files``) plus ``logic/``.  The
``alfworld`` package is installed editable from the frozen source tree
``RSI_ALFWORLD_SRC`` (default ``/mnt/songzijun/datasets/alfworld_src``) by the
prepare script; package code never mutates ``sys.path``.

Privacy boundary: the game file encodes the goal and the win condition — the
Actor's environment necessarily runs the game (the task instruction is part of
the first observation, as in the official protocol), but the *win evaluation*
stays scorer-side: the actor only ever produces a text command trajectory; the
Scorer re-executes it on its own env instance and reads the engine's ``won``.
Reward values are never runtime observations (``ScoreRecord`` is scorer-side).

Tool surface: the environment's own action space is free-text commands
(``go to …``, ``take …``, …) — one basic tool ``command``.  The official state
dict's ``admissible_commands`` are carried in ``Observation.structured`` (they
are engine output, not hidden info) and never injected into the text.

Snapshot/fork: eval games are deterministic (``domain_randomization`` off,
``AlfredDemangler(shuffle=False)``) — replay-prefix snapshots (command log +
digest verification) like the AppWorld/WebShop adapters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue, digest_of
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

DEFAULT_ALFWORLD_ROOT = "/mnt/songzijun/datasets/alfworld"
DEFAULT_ALFWORLD_SRC = "/mnt/songzijun/datasets/alfworld_src"

ALFWORLD_VERSION = "alfworld-0.5.0+data-json-2.1.x-frozen-v1"

ALFWORLD_TOOL_SPEC: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="command",
        description=(
            "Send one natural-language text command to the household world "
            "(e.g. 'go to drawer 1', 'take apple from table', 'heat apple with "
            "microwave', 'move apple to diningtable'). Returns the engine feedback."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        capabilities=("text_command",),
    ),
)


def _require_alfworld_deps() -> Any:
    missing = []
    for mod in ("textworld", "termcolor", "tqdm"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"ALFWorld adapter requires optional packages {missing} "
            "(pip install -e '.[alfworld]'). No fallback environment is used."
        )


def _bootstrap_alfworld(src_root: Path, data_root: Path) -> None:
    """Pin the official ALFWORLD_DATA path (the package reads it at import).

    The ``alfworld`` package itself must be *installed* from the frozen source
    tree (``pip install -e <src_root>`` by ``scripts/prepare_alfworld_data.py``)
    — package code never mutates ``sys.path``.  Uninstalled → loud failure at
    the import sites below, never a fallback environment.
    """
    os.environ["ALFWORLD_DATA"] = str(data_root)


def extract_instruction(traj: dict[str, Any]) -> str:
    """Best-effort goal text from ``traj_data.json`` (schema varies slightly
    across ALFRED dumps); falls back to a stable placeholder like AppWorld.

    Official json_2.1.1 dumps carry the goal under
    ``turk_annotations.anns[i].task_desc`` — ``turk_annotations`` is a dict
    ``{"anns": [...]}`` (older trees used a bare list), so both shapes are
    handled before the legacy ``annots``/``annotations`` fallbacks.
    """
    for key in ("task_desc", "goal_desc", "instruction"):
        v = traj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    goal = traj.get("goal")
    if isinstance(goal, dict):
        for key in ("task_desc", "goal_desc", "description", "task"):
            v = goal.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    scene_goal = traj.get("scene_goal")
    if isinstance(scene_goal, dict):
        v = scene_goal.get("description") or scene_goal.get("task_desc")
        if isinstance(v, str) and v.strip():
            return v.strip()

    turk = traj.get("turk_annotations")
    anns: Any = None
    if isinstance(turk, dict):
        anns = turk.get("anns")
    elif isinstance(turk, list):
        anns = turk
    if isinstance(anns, list):
        for first in anns:
            if isinstance(first, dict):
                for key in ("task_desc", "goal_desc", "desc", "description"):
                    v = first.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            elif isinstance(first, str) and first.strip():
                return first.strip()

    anns = traj.get("annots") or traj.get("annotations")
    if isinstance(anns, list) and anns:
        first = anns[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            v = first.get("desc") or first.get("description")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


class AlfWorldAdapter:
    """EnvironmentAdapter over a single official ALFWorld TextWorld game."""

    name = "alfworld"
    version = ALFWORLD_VERSION
    supports_shadow_fork = True

    def __init__(
        self,
        data_root: str | None = None,
        *,
        src_root: str | None = None,
        max_steps: int = 50,
        out_root: str | None = None,
    ):
        _require_alfworld_deps()
        self._root = Path(data_root or os.environ.get("RSI_ALFWORLD_ROOT", DEFAULT_ALFWORLD_ROOT))
        self._src = Path(src_root or os.environ.get("RSI_ALFWORLD_SRC", DEFAULT_ALFWORLD_SRC))
        self._max_steps = int(max_steps)
        self._out_root = Path(out_root or os.environ.get("RSI_ALFWORLD_OUT") or self._root / "rsi_outputs")
        self._gym_env: Any = None
        self._task_id = ""
        self._game_file = ""
        self._commands: list[str] = []
        self._state: Any = None
        self._started = False

    def check_ready(self) -> tuple[bool, str]:
        try:
            _require_alfworld_deps()
        except NotImplementedCapability as e:
            return False, str(e)
        _bootstrap_alfworld(self._src, self._root)
        try:
            import alfworld  # noqa: F401
            import textworld.gym  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"alfworld/textworld.gym not importable ({e!r})"
        logic = self._root / "logic"
        if not (logic / "alfred.pddl").is_file() or not (logic / "alfred.twl2").is_file():
            return False, f"ALFWorld logic files missing under {logic} (run scripts/prepare_alfworld_data.py)"
        return True, "ok"

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        _require_alfworld_deps()
        _bootstrap_alfworld(self._src, self._root)
        rel = str(public_task.metadata.get("alfworld_game_file") or "")
        game_file = (self._root / rel).resolve()
        try:
            game_file.relative_to(self._root.resolve())
        except ValueError as e:
            raise EnvironmentError_(f"game file escapes data root: {rel}") from e
        if not game_file.is_file():
            raise EnvironmentError_(
                f"ALFWorld game file missing: {game_file} — run scripts/prepare_alfworld_data.py"
            )
        self._task_id = public_task.task_id
        self._game_file = str(game_file)
        self._commands = []
        self._close_env()
        self._gym_env = self._make_env([self._game_file])
        try:
            # textworld.gym batch API: (obs_list, infos dict-of-lists)
            obs, infos = self._gym_env.reset()
            self._state = self._unpack(obs, infos)
        except Exception as e:  # noqa: BLE001
            raise EnvironmentError_(f"ALFWorld game reset failed: {e!r}") from e
        self._started = True
        return Observation(
            content=self._feedback(self._state),
            provenance=ObservationProvenance(source_tool="reset", source_ref=self._task_id),
            structured=self._structured(self._state),
        )

    def _make_env(self, game_files: list[str]) -> Any:
        """Single-game TextWorld gym env with the official ALFWorld wrappers."""
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        except Exception as e:  # noqa: BLE001
            raise NotImplementedCapability(
                f"ALFWorld text env not importable ({e!r}); install the optional "
                "'alfworld' extra and prepare RSI_ALFWORLD_SRC"
            ) from e
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            game_files,
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=self._max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        return textworld.gym.make(env_id)

    @staticmethod
    def _unpack(obs: Any, infos: Any) -> dict[str, Any]:
        """Flatten one game out of the textworld.gym batch API."""
        text = obs[0] if isinstance(obs, (list, tuple)) and len(obs) else obs
        return {
            "obs": str(text),
            "won": bool((infos.get("won") or [False])[0]),
            "admissible_commands": [str(c) for c in ((infos.get("admissible_commands") or [[]])[0])],
            "gamefile": (infos.get("extra.gamefile") or [None])[0],
        }

    @staticmethod
    def _feedback(state: Any) -> str:
        return str(state.get("obs", state))

    def _structured(self, state: Any) -> dict[str, JSONValue]:
        return {"admissible_commands": list(state.get("admissible_commands") or [])}

    @staticmethod
    def _won(state: Any) -> bool:
        return bool(state.get("won", False))

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return ALFWORLD_TOOL_SPEC

    def step(self, action: Action) -> StepResult:
        self._require_started()
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(
                    content="finished",
                    done=True,
                    provenance=ObservationProvenance(source_tool="finish", source_ref=self._task_id),
                ),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        if action.name != "command":
            return StepResult(
                observation=Observation(content=f"unknown tool {action.name!r}", ok=False,
                                        error_kind="model_invalid_action"),
                done=False,
            )
        text = action.arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            return StepResult(
                observation=Observation(content="text must be a non-empty string", ok=False,
                                        error_kind="model_invalid_action"),
                done=False,
            )
        return self._step_raw(text.strip())

    def _step_raw(self, command: str) -> StepResult:
        self._commands.append(command)
        try:
            obs_l, _scores, dones, infos = self._gym_env.step([command])
            state = self._unpack(obs_l, infos)
        except Exception as e:  # noqa: BLE001 - execution errors are observations
            return StepResult(
                observation=Observation(content=f"ENV_ERROR: {e!r}", ok=False, error_kind="tool_error"),
                done=False,
            )
        self._state = state
        done = bool(dones[0]) or self._won(state) or len(self._commands) >= self._max_steps
        obs = Observation(
            content=self._feedback(state),
            provenance=ObservationProvenance(source_tool="command", source_ref=self._task_id),
            structured=self._structured(state),
        )
        return StepResult(observation=obs, done=done, snapshot=self.snapshot(), cost={"tool_calls": 1})

    def snapshot(self) -> SnapshotRef:
        self._require_started()
        return SnapshotRef(
            ref_id=f"replay:{len(self._commands)}",
            digest=self._replay_digest(self._commands),
            kind="replay_prefix",
        )

    def _replay_digest(self, commands: list[str]) -> str:
        return digest_of({"task": self._task_id, "commands": list(commands)})

    def fork(self, snapshot: SnapshotRef) -> "AlfWorldAdapter":
        """Replay-prefix fork: a fresh env re-executes the recorded commands."""
        child = AlfWorldAdapter(
            str(self._root), src_root=str(self._src), max_steps=self._max_steps, out_root=str(self._out_root)
        )
        child._task_id = self._task_id
        child._game_file = self._game_file
        child._gym_env = child._make_env([self._game_file])
        child._state = child._gym_env.reset()
        child._started = True
        for cmd in self._commands:
            child._step_raw(cmd)
        if snapshot.kind == "replay_prefix" and snapshot.digest:
            if child._replay_digest(child._commands) != snapshot.digest:
                raise EnvironmentError_(
                    f"fork replay digest mismatch: state not reproduced ({snapshot.digest[:12]})"
                )
        return child

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        return self._replay_digest(self._commands)

    def close(self) -> None:
        self._close_env()
        self._started = False

    def _close_env(self) -> None:
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:  # noqa: BLE001
                pass
        self._gym_env = None

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "data_root": str(self._root),
            "src_root": str(self._src),
            "tool_schema_hash": digest_of([t.digest() for t in ALFWORLD_TOOL_SPEC]),
            "supports_shadow_fork": True,
            "synthetic": False,
        }

    def final_artifact(self) -> ArtifactRef:
        """Write the actor-side command trajectory (no win/reward inside — the
        scorer re-executes and reads the engine's own completion check)."""
        record = {
            "task_id": self._task_id,
            "game_file": self._game_file,
            "commands": list(self._commands),
        }
        out_dir = self._out_root / (self._task_id.replace(":", "_").replace("/", "_") or "task")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "final_artifact.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), "utf-8")
        return ArtifactRef(
            ref_id=f"alfworld:{self._task_id}",
            path=str(path),
            digest=digest_of(record),
            kind="trajectory",
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("alfworld adapter not reset")


class AlfWorldScorer:
    """Official ALFWorld evaluation (success = engine ``won``) on the evaluator side.

    Re-executes the recorded command trajectory on the private game file in the
    scorer's own env instance and reads the engine completion state — the same
    check ``scripts/run_eval.py`` aggregates into success rate.
    """

    scorer_version = f"alfworld-won-{ALFWORLD_VERSION}"

    def __init__(self, data_root: str | None = None, *, src_root: str | None = None, max_steps: int = 50):
        self._root = Path(data_root or os.environ.get("RSI_ALFWORLD_ROOT", DEFAULT_ALFWORLD_ROOT))
        self._src = Path(src_root or os.environ.get("RSI_ALFWORLD_SRC", DEFAULT_ALFWORLD_SRC))
        self._max_steps = int(max_steps)

    def check_ready(self) -> tuple[bool, str]:
        try:
            _require_alfworld_deps()
        except NotImplementedCapability as e:
            return False, str(e)
        _bootstrap_alfworld(self._src, self._root)
        try:
            import textworld.gym  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"textworld.gym not importable ({e!r})"
        return True, "ok"

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        _require_alfworld_deps()
        _bootstrap_alfworld(self._src, self._root)
        task_id = private_task_ref.ref_id
        record: dict[str, Any] = {}
        if final_artifact.path and Path(final_artifact.path).is_file():
            record = json.loads(Path(final_artifact.path).read_text("utf-8"))
        commands = [str(c) for c in record.get("commands") or []]
        game_rel = str(record.get("game_file") or "")
        game_file = Path(game_rel) if game_rel.startswith("/") else (self._root / game_rel)
        if not game_file.is_file():
            raise EnvironmentError_(f"private game file missing for {task_id}: {game_file}")
        adapter = AlfWorldAdapter(
            str(self._root), src_root=str(self._src), max_steps=self._max_steps
        )
        gym_env = adapter._make_env([str(game_file)])
        try:
            obs, infos = gym_env.reset()
            state = AlfWorldAdapter._unpack(obs, infos)
            won = AlfWorldAdapter._won(state)
            final_reward = 0.0
            for cmd in commands[: self._max_steps]:
                obs, scores, dones, infos = gym_env.step([cmd])
                state = AlfWorldAdapter._unpack(obs, infos)
                final_reward = float(scores[0]) if scores else 0.0
                won = AlfWorldAdapter._won(state)
                if bool(dones[0]):
                    break
        finally:
            try:
                gym_env.close()
            except Exception:  # noqa: BLE001
                pass
        return ScoreRecord(
            task_id=task_id,
            domain="alfworld",
            metric_name="task_success",
            value=1.0 if won else 0.0,
            success=bool(won),
            details={
                "won": bool(won),
                "final_reward": final_reward,
                "n_commands": len(commands),
            },
            scorer_version=self.scorer_version,
            synthetic=False,
        )
