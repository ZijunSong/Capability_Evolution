"""WebShop / ALFWorld adapter+scorer contract tests with stubbed official envs.

These exercise the RSI-side contracts (tool surface, official-action mapping,
replay-prefix snapshot/fork, reward redaction, scorer privacy boundary and
official-metric pass-through) without loading the heavy official product
corpora / TextWorld games.  The real end-to-end chains are covered by the
integration smokes (skipped loudly when data/deps are absent).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rsi.environments.alfworld_env as alf_mod
import rsi.environments.webshop_env as web_mod
from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.errors import EnvironmentError_, ProtocolViolation
from rsi.contracts.observation import ArtifactRef
from rsi.contracts.task import PrivateTaskRef, PublicTask


# --------------------------------------------------------------------------
# WebShop stubs
# --------------------------------------------------------------------------
class _FakeServer:
    def __init__(self, goals, prices):
        self.goals = [dict(g) for g in goals]
        self.product_prices = dict(prices)
        self.weights = [1] * len(goals)
        self.cum_weights = list(range(len(goals) + 1))
        self.user_sessions = {}
        self.product_item_dict = {
            "ASIN000": {
                "asin": "ASIN000",
                "query": "q",
                "name": "product",
                "product_category": "pc",
                "Attributes": ["blue"],
                "Title": "product",
                "BulletPoints": [],
                "Description": "desc",
            }
        }


class _FakeWebShopEnv:
    def __init__(self, goals, prices):
        self.server = _FakeServer(goals, prices)
        self.session = ""
        self.actions = []

    def reset(self, session=None):
        self.session = str(session)
        self.server.user_sessions[self.session] = {"goal": self.server.goals[session], "asin": None, "options": {}}
        return f"page for goal {session}", None

    def step(self, action):
        self.actions.append(action)
        if action.startswith("click[buy now]"):
            self.server.user_sessions[self.session] = {"asin": "ASIN000", "options": {"color": "blue"}}
            return "Done page Reward = 0.75", 0.75, True, None
        return f"page after {action}", 0.0, False, None

    def close(self):
        pass


def _webshop_root(tmp_path: Path):
    goals = [
        {"asin": "ASIN000", "instruction_text": "buy the blue product", "attributes": ["blue"],
         "goal_options": {"color": "blue"}, "price_upper": 50.0, "weight": 1,
         "query": "q", "name": "product", "product_category": "pc", "category": "c"},
        {"asin": "ASIN001", "instruction_text": "buy another product", "attributes": ["red"],
         "goal_options": {"color": "red"}, "price_upper": 50.0, "weight": 1,
         "query": "q", "name": "other", "product_category": "pc", "category": "c"},
    ]
    payload = {"schema_version": "rsi_webshop_goals_v1", "goals": goals,
               "product_prices": {"ASIN000": 9.99, "ASIN001": 4.99}}
    gp = tmp_path / "frozen_goals.json"
    gp.write_text(json.dumps(payload), "utf-8")
    from rsi.contracts.common import digest_of_file

    (tmp_path / "frozen_manifest.json").write_text(
        json.dumps({
            "schema_version": "rsi_webshop_frozen_v1",
            "flavor": "small",
            "items_file": "data/items_shuffle_1000.json",
            "goals_file": "frozen_goals.json",
            "goals_sha256": digest_of_file(str(gp)),
            "panel": {"final_test_count": 1, "transfer_test_count": 1},
        }),
        "utf-8",
    )
    return tmp_path, goals


@pytest.fixture()
def webshop_adapter(tmp_path, monkeypatch):
    root, goals = _webshop_root(tmp_path)
    monkeypatch.setattr(web_mod, "_require_webshop_deps", lambda: None)
    monkeypatch.setattr(web_mod, "_require_webshop_importable", lambda root: None)
    monkeypatch.setattr(
        web_mod.WebShopAdapter, "_build_env", lambda self: _FakeWebShopEnv(goals, self._prices)
    )
    adapter = web_mod.WebShopAdapter(str(root))
    return adapter, goals


def _web_task(i: int, instruction: str) -> PublicTask:
    return PublicTask(
        task_id=f"webshop:g{i:05d}",
        domain="webshop",
        instruction=instruction,
        group_id=f"webshop:ASIN{i:03d}",
        metadata={"webshop_goal_index": i, "asin": f"ASIN{i:03d}"},
    )


def test_webshop_reset_selects_goal_by_index(webshop_adapter):
    adapter, goals = webshop_adapter
    obs = adapter.reset(_web_task(1, goals[1]["instruction_text"]), seed=7)
    assert "buy another product" in obs.content
    assert adapter._env.session == "1"  # int session -> official goals[idx]


def test_webshop_instruction_mismatch_is_loud(webshop_adapter):
    adapter, goals = webshop_adapter
    with pytest.raises(ProtocolViolation):
        adapter.reset(_web_task(0, "WRONG instruction"), seed=1)


def test_webshop_tool_mapping_and_reward_redaction(webshop_adapter):
    adapter, goals = webshop_adapter
    adapter.reset(_web_task(0, goals[0]["instruction_text"]), seed=1)
    res = adapter.step(ToolCall(name="search", arguments={"query": "blue product"}))
    assert adapter._actions == ["search[blue product]"]
    assert res.observation.ok
    res = adapter.step(ToolCall(name="click", arguments={"text": "buy now"}))
    assert res.done
    assert "0.75" not in res.observation.content  # official reward hidden from actor
    assert "<hidden>" in res.observation.content
    assert adapter._purchase["asin"] == "ASIN000"
    assert adapter._purchase["product"]["Attributes"] == ["blue"]


def test_webshop_snapshot_fork_replays(webshop_adapter):
    adapter, goals = webshop_adapter
    adapter.reset(_web_task(0, goals[0]["instruction_text"]), seed=1)
    adapter.step(ToolCall(name="search", arguments={"query": "blue"}))
    snap = adapter.snapshot()
    fork = adapter.fork(snap)
    assert fork._actions == adapter._actions
    assert fork.state_digest() == adapter.state_digest()
    fork.close()


def test_webshop_final_artifact_and_scorer_pass_through(webshop_adapter, monkeypatch):
    adapter, goals = webshop_adapter
    adapter.reset(_web_task(0, goals[0]["instruction_text"]), seed=1)
    adapter.step(ToolCall(name="click", arguments={"text": "buy now"}))
    art = adapter.final_artifact()
    assert art.path and Path(art.path).is_file()
    record = json.loads(Path(art.path).read_text("utf-8"))
    blob = json.dumps(record)
    assert "goal_options" not in blob and "price_upper" not in blob  # gold goal never in the artifact

    # stub the official metric function; the scorer must pass its value through
    captured = {}

    def fake_get_reward(product, goal, price=None, options=None, **kw):
        captured.update(product=product, goal=goal, price=price, options=options)
        return 0.75

    goal_mod = types.ModuleType("web_agent_site.engine.goal")
    goal_mod.get_reward = fake_get_reward
    site_mod = types.ModuleType("web_agent_site")
    engine_mod = types.ModuleType("web_agent_site.engine")
    monkeypatch.setitem(sys.modules, "web_agent_site", site_mod)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine.goal", goal_mod)
    monkeypatch.setattr(web_mod, "_require_webshop_deps", lambda: None)

    scorer = web_mod.WebShopScorer(str(adapter._root))
    scorer.load_private("webshop:g00000", goals[0])
    rec = scorer.evaluate(art, PrivateTaskRef(ref_id="webshop:g00000", domain="webshop"))
    assert rec.metric_name == "task_reward"
    assert rec.value == 0.75 and not rec.success  # harsh success only at 1.0
    assert captured["goal"]["attributes"] == ["blue"]
    assert captured["options"] == {"color": "blue"}


def test_webshop_no_purchase_scores_zero(webshop_adapter, monkeypatch):
    adapter, goals = webshop_adapter
    adapter.reset(_web_task(0, goals[0]["instruction_text"]), seed=1)
    art = adapter.final_artifact()
    monkeypatch.setattr(web_mod, "_require_webshop_deps", lambda: None)
    scorer = web_mod.WebShopScorer(str(adapter._root))
    scorer.load_private("webshop:g00000", goals[0])
    rec = scorer.evaluate(art, PrivateTaskRef(ref_id="webshop:g00000", domain="webshop"))
    assert rec.value == 0.0 and not rec.success


# --------------------------------------------------------------------------
# ALFWorld stubs
# --------------------------------------------------------------------------
class _FakeGym:
    """Mimics textworld.gym TextWorldBatchEnv (batch_size=1) shapes."""

    def __init__(self, won_after: int = 2):
        self.won_after = won_after
        self.commands = []
        self.closed = False

    def reset(self):
        return (
            ("Your task is to: put a clean apple on the table.",),
            {"won": [False], "admissible_commands": [["go to table 1", "help"]], "extra.gamefile": ["game.tw-pddl"]},
        )

    def step(self, commands):
        assert isinstance(commands, (list, tuple)), "Expected a list of commands."
        cmd = commands[0]
        self.commands.append(cmd)
        won = len(self.commands) >= self.won_after
        return (
            (f"After: {cmd}",),
            [1.0 if won else 0.0],
            [won],
            {"won": [won], "admissible_commands": [[]], "extra.gamefile": ["game.tw-pddl"]},
        )

    def close(self):
        self.closed = True


@pytest.fixture()
def alfworld_adapter(tmp_path, monkeypatch):
    game = tmp_path / "train" / "inst1" / "trial_T1" / "game.tw-pddl"
    game.parent.mkdir(parents=True)
    game.write_text("{}", "utf-8")
    monkeypatch.setattr(alf_mod, "_require_alfworld_deps", lambda: None)
    monkeypatch.setattr(alf_mod.AlfWorldAdapter, "_make_env", lambda self, files: _FakeGym())
    adapter = alf_mod.AlfWorldAdapter(str(tmp_path))
    task = PublicTask(
        task_id="alfworld:train/inst1/trial_T1",
        domain="alfworld",
        instruction="put a clean apple on the table",
        group_id="alfworld:inst1",
        metadata={"alfworld_game_file": "train/inst1/trial_T1/game.tw-pddl"},
    )
    return adapter, task


def test_alfworld_reset_and_command_tool(alfworld_adapter):
    adapter, task = alfworld_adapter
    obs = adapter.reset(task, seed=1)
    assert "put a clean apple" in obs.content
    assert obs.structured["admissible_commands"] == ["go to table 1", "help"]
    res = adapter.step(ToolCall(name="command", arguments={"text": "go to table 1"}))
    assert res.observation.ok and not res.done
    res = adapter.step(ToolCall(name="command", arguments={"text": "take apple from table 1"}))
    assert res.done  # stub wins after 2 commands
    assert adapter._commands == ["go to table 1", "take apple from table 1"]


def test_alfworld_snapshot_fork_replays(alfworld_adapter):
    adapter, task = alfworld_adapter
    adapter.reset(task, seed=1)
    adapter.step(ToolCall(name="command", arguments={"text": "look"}))
    snap = adapter.snapshot()
    fork = adapter.fork(snap)
    assert fork._commands == adapter._commands
    assert fork.state_digest() == adapter.state_digest()
    fork.close()


def test_alfworld_artifact_hides_win_state(alfworld_adapter):
    adapter, task = alfworld_adapter
    adapter.reset(task, seed=1)
    adapter.step(ToolCall(name="command", arguments={"text": "look"}))
    adapter.step(ToolCall(name="command", arguments={"text": "go to table 1"}))
    art = adapter.final_artifact()
    record = json.loads(Path(art.path).read_text("utf-8"))
    assert record["commands"] == ["look", "go to table 1"]
    assert "won" not in record and "reward" not in record


def test_alfworld_scorer_reexecutes_trajectory(alfworld_adapter, monkeypatch):
    adapter, task = alfworld_adapter
    adapter.reset(task, seed=1)
    adapter.step(ToolCall(name="command", arguments={"text": "look"}))
    adapter.step(ToolCall(name="command", arguments={"text": "go to table 1"}))
    art = adapter.final_artifact()
    monkeypatch.setattr(alf_mod, "_require_alfworld_deps", lambda: None)
    monkeypatch.setattr(alf_mod.AlfWorldAdapter, "_make_env", lambda self, files: _FakeGym(won_after=2))
    scorer = alf_mod.AlfWorldScorer(str(adapter._root))
    rec = scorer.evaluate(art, PrivateTaskRef(ref_id=task.task_id, domain="alfworld"))
    assert rec.metric_name == "task_success"
    assert rec.value == 1.0 and rec.success
    assert rec.details["n_commands"] == 2


def test_alfworld_scorer_zero_without_win(alfworld_adapter, monkeypatch):
    adapter, task = alfworld_adapter
    adapter.reset(task, seed=1)
    adapter.step(ToolCall(name="command", arguments={"text": "look"}))
    art = adapter.final_artifact()
    monkeypatch.setattr(alf_mod, "_require_alfworld_deps", lambda: None)
    monkeypatch.setattr(alf_mod.AlfWorldAdapter, "_make_env", lambda self, files: _FakeGym(won_after=99))
    scorer = alf_mod.AlfWorldScorer(str(adapter._root))
    rec = scorer.evaluate(art, PrivateTaskRef(ref_id=task.task_id, domain="alfworld"))
    assert rec.value == 0.0 and not rec.success


def test_alfworld_rejects_game_outside_root(alfworld_adapter):
    adapter, task = alfworld_adapter
    evil = PublicTask(
        task_id="x", domain="alfworld", instruction="i", group_id="g",
        metadata={"alfworld_game_file": "../escape/game.tw-pddl"},
    )
    with pytest.raises(EnvironmentError_):
        adapter.reset(evil, seed=1)


def test_alfworld_finish_action_step_result(alfworld_adapter):
    """Finish must carry done on StepResult only — Observation has no ``done``
    field (zqpvg 复盘: passing done=True raised TypeError, killing every task
    whose model emitted a Finish action)."""
    adapter, task = alfworld_adapter
    adapter.reset(task, seed=1)
    res = adapter.step(Finish())
    assert res.done is True
    assert res.observation.content == "finished"
    assert res.observation.ok is True


def test_alfworld_step_observation_has_no_done_kwarg(alfworld_adapter):
    """Observation dataclass exposes no done field; guard against regression."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(alf_mod.Observation)}
    assert "done" not in fields


# --------------------------------------------------------------------------
# extract_instruction schema coverage (official json_2.1.1 + legacy shapes)


def test_alfworld_extract_instruction_official_turk_dict_schema():
    """Official json_2.1.1: turk_annotations = {"anns": [{task_desc, ...}]}."""
    traj = {
        "task_type": "look_at_obj_in_light",
        "turk_annotations": {
            "anns": [
                {"assignment_id": "A1", "task_desc": "Hold the clock and turn on the lamp."},
                {"assignment_id": "A2", "task_desc": "Other wording."},
            ]
        },
    }
    assert (
        alf_mod.extract_instruction(traj)
        == "Hold the clock and turn on the lamp."
    )


def test_alfworld_extract_instruction_legacy_list_schema():
    """Older ALFRED dumps: turk_annotations as a bare list."""
    traj = {"turk_annotations": [{"task_desc": "put two spray bottles on the toilet."}]}
    assert alf_mod.extract_instruction(traj) == "put two spray bottles on the toilet."


def test_alfworld_extract_instruction_legacy_annots_fallback():
    traj = {"annots": [{"desc": "clean the lettuce"}]}
    assert alf_mod.extract_instruction(traj) == "clean the lettuce"


def test_alfworld_extract_instruction_placeholder_when_nothing_present():
    assert alf_mod.extract_instruction({"task_type": "x"}) == ""
