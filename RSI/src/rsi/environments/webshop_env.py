"""WebShop adapter — e-commerce search / navigation / purchase tasks.

Official source: https://github.com/princeton-nlp/WebShop (source tree frozen at
``data prepare`` via ``webshop_src_commit`` in the frozen manifest).  Official
metrics: per-task reward from ``web_agent_site.engine.goal.get_reward``
(attribute/option/price/type matching, scaled to [0, 1]) and harsh success
(``reward == 1``), matching the official 500-goal human test protocol
(``baseline_models/test.py`` iterates ``idx in range(500)``).

Frozen goal manifest: the official goal construction draws random price bounds
(``get_human_goals`` uses the unseeded global RNG) and the item prices
themselves (``generate_product_prices``), so a bare env is not deterministic
across processes.  ``scripts/prepare_webshop_data.py`` materializes
``frozen_goals.json`` (official ``get_goals`` output under a pinned RNG seed,
goals in the official post-shuffle order) plus ``frozen_manifest.json``
(source digests).  The adapter replaces ``SimServer.goals`` /
``SimServer.product_prices`` with the frozen copies so the official
``get_reward`` definition evaluates exactly the frozen benchmark.

Privacy boundary: the gold goal (attributes / goal_options / price_upper) lives
only in the evaluator side (``PrivateTaskRef`` / scorer private records).  The
Actor sees the instruction and the shop pages.  The purchase record needed for
scoring is written to a final artifact after the episode; the done-page reward
line is redacted from actor observations (``ScoreRecord`` is never a runtime
observation).

Layout (``RSI_WEBSHOP_ROOT``, default ``/mnt/songzijun/datasets/webshop``): a
complete official WebShop project tree — ``web_agent_site/``, ``search_engine/``
(with the built index), ``data/`` — plus the two frozen files.  The official
package resolves its data/index paths relative to itself, so the dataset root
*is* the project root (mirrors the official ``setup.sh`` layout) and is
installed editable from there by the prepare script (``pip install -e <root>``);
package code never mutates ``sys.path``.

Snapshot/fork: the world is deterministic given the frozen goals, frozen prices,
fixed product corpus and the fixed Lucene index — replay-prefix snapshots
(action log + digest verification) like the AppWorld adapter.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue, digest_of, digest_of_file
from rsi.contracts.errors import EnvironmentError_, NotImplementedCapability, ProtocolViolation
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

DEFAULT_WEBSHOP_ROOT = "/mnt/songzijun/datasets/webshop"

WEBSHOP_VERSION = "webshop-pinned+data-frozen-v1"

WEBSHOP_TOOL_SPEC: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search",
        description=(
            "Search the shop with free-text keywords (official action "
            "``search[keywords]``). Returns the search results page."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        capabilities=("web_search",),
    ),
    ToolSpec(
        name="click",
        description=(
            "Click a button/link/option visible on the current page by its "
            "exact text (official action ``click[value]``). ``buy now`` "
            "purchases the current product and ends the episode."
        ),
        parameters_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        capabilities=("ui_click",),
    ),
)

# Official done-page renders "Reward = <float>"; hidden from actor observations.
_REWARD_LINE_RE = re.compile(r"(?i)(reward\s*=\s*)[0-9]*\.?[0-9]+")


def _require_webshop_deps() -> Any:
    missing = []
    for mod in ("bs4", "thefuzz", "spacy", "pyserini"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"WebShop adapter requires optional packages {missing} "
            "(pip install -e '.[webshop]'). No fallback environment is used."
        )


def _require_webshop_importable(root: Path) -> Any:
    """The official package resolves data/index paths relative to itself, so it
    is installed from the prepared project tree (``pip install -e <root>`` by
    ``scripts/prepare_webshop_data.py``).  Package code never mutates
    ``sys.path`` — an uninstalled tree fails loudly here."""
    try:
        import web_agent_site  # noqa: F401
    except ImportError as e:
        raise NotImplementedCapability(
            f"web_agent_site not importable ({e!r}); the prepared project tree "
            f"at {root} must be installed (pip install -e {root}, done by "
            "scripts/prepare_webshop_data.py). No fallback environment is used."
        ) from e
    return web_agent_site


def _load_frozen(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    manifest_path = root / "frozen_manifest.json"
    if not manifest_path.is_file():
        raise NotImplementedCapability(
            f"WebShop frozen manifest missing: {manifest_path} — run "
            "scripts/prepare_webshop_data.py (materialized by data prepare)"
        )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    goals_path = root / str(manifest.get("goals_file") or "frozen_goals.json")
    if not goals_path.is_file():
        raise NotImplementedCapability(f"WebShop frozen goals missing: {goals_path}")
    expect = manifest.get("goals_sha256")
    actual = digest_of_file(str(goals_path))
    if expect and actual != expect:
        raise ProtocolViolation(
            f"WebShop frozen goals digest mismatch: {actual[:12]} != {str(expect)[:12]}"
        )
    payload = json.loads(goals_path.read_text("utf-8"))
    return manifest, list(payload["goals"]), dict(payload["product_prices"])


# Shared SimServer cache.  Building the official server is O(corpus): the
# flavor=all ``load_products`` json.loads the 5.5GB items file (peak >16GB RSS)
# and ``get_goals`` walks all 1.18M products — minutes per build.  The panel
# runners construct one adapter *per task* (run_panel / _panel_worker), so a
# per-adapter build would dominate wall-clock and re-materialize gigabytes per
# episode.  The SimServer keeps all mutable episode state inside
# ``user_sessions[session_id]`` (goal, page, cart, actions), so one server
# serves every task in the process as long as each episode uses its own
# session id (the adapter keys sessions by the int goal index).  Eval panels
# run one worker process per GPU, so this cache is bounded at one server
# (~its all_products footprint) per worker.
_WEBSHOP_SERVER_CACHE: dict[str, tuple[Any, str]] = {}
_WEBSHOP_SERVER_LOCK = __import__("threading").Lock()


def _redact_reward(text: str) -> str:
    return _REWARD_LINE_RE.sub(r"\1<hidden>", text)


class WebShopAdapter:
    """EnvironmentAdapter over the official WebAgentTextEnv."""

    name = "webshop"
    version = WEBSHOP_VERSION
    supports_shadow_fork = True

    def __init__(
        self,
        data_root: str | None = None,
        *,
        max_steps: int = 100,
        out_root: str | None = None,
    ):
        _require_webshop_deps()
        self._root = Path(data_root or os.environ.get("RSI_WEBSHOP_ROOT", DEFAULT_WEBSHOP_ROOT))
        self._manifest, self._goals, self._prices = _load_frozen(self._root)
        self._max_steps = int(max_steps)
        self._out_root = Path(out_root or os.environ.get("RSI_WEBSHOP_OUT") or self._root / "rsi_outputs")
        self._env: Any = None
        self._task_id = ""
        self._goal_index = -1
        self._session = ""
        self._actions: list[str] = []
        self._purchase: dict[str, Any] | None = None
        self._bought = False
        self._started = False

    def check_ready(self) -> tuple[bool, str]:
        """Panel-level capability probe: package importable + search index built."""
        try:
            _require_webshop_deps()
        except NotImplementedCapability as e:
            return False, str(e)
        try:
            _require_webshop_importable(self._root)
        except NotImplementedCapability as e:
            return False, str(e)
        indexes = self._root / "search_engine" / "indexes"
        flavor = self._manifest.get("flavor", "small")
        if flavor == "small" and (self._root / "search_engine" / "indexes_1k").is_dir():
            indexes = self._root / "search_engine" / "indexes_1k"
        if not indexes.is_dir() or not any(indexes.iterdir()):
            return False, f"search index not built under {indexes} (run scripts/prepare_webshop_data.py)"
        if self._manifest.get("flavor") == "all" and os.environ.get("RSI_WEBSHOP_FULL_ALLOW") != "1":
            return False, (
                "flavor=all (full 1.18M-product corpus) needs a large-memory node: the "
                "official load_products json.load peaks >16GB and dev-container memcg "
                "oom-kills it — run on CloudML with RSI_WEBSHOP_FULL_ALLOW=1 "
                "(see docs/runbook.md)"
            )
        try:
            import spacy

            spacy.load("en_core_web_sm")
        except Exception as e:  # noqa: BLE001
            return False, f"spacy model en_core_web_sm unavailable ({e!r}); needed by official get_reward"
        return True, "ok"

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        _require_webshop_importable(self._root)
        goal_index = int(public_task.metadata.get("webshop_goal_index", -1))
        if not (0 <= goal_index < len(self._goals)):
            raise EnvironmentError_(f"webshop goal index out of range: {goal_index}")
        goal = self._goals[goal_index]
        if public_task.instruction and public_task.instruction != goal["instruction_text"]:
            raise ProtocolViolation(
                "PublicTask instruction does not match the frozen goal — "
                "split manifest and frozen goals are out of sync"
            )
        self._task_id = public_task.task_id
        self._goal_index = goal_index
        self._actions = []
        self._purchase = None
        self._bought = False
        if self._env is None:
            self._env = self._build_env()
        self._pin_frozen_state()
        # Session id must be the int goal index: the official reset only passes
        # session_int (which selects goals[idx]) for int sessions.  A stale
        # same-id session entry would otherwise keep the previous episode goal.
        self._session = str(goal_index)
        self._env.server.user_sessions.pop(self._session, None)
        obs_text, _ = self._env.reset(session=goal_index)
        self._started = True
        return Observation(
            content=f"Task: {goal['instruction_text']}\n\n{obs_text}",
            provenance=ObservationProvenance(source_tool="reset", source_ref=self._task_id),
        )

    def _build_env(self) -> Any:
        try:
            from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        except Exception as e:  # noqa: BLE001
            raise NotImplementedCapability(
                f"WebAgentTextEnv not importable from {self._root} ({e!r}); "
                "the official project tree must be prepared under the dataset root"
            ) from e
        items = self._root / str(self._manifest.get("items_file") or "data/items_shuffle_1000.json")
        if not items.is_file():
            raise NotImplementedCapability(f"WebShop product data missing: {items}")
        num_products = self._manifest.get("num_products")
        server = self._shared_server(items, num_products)
        return WebAgentTextEnv(observation_mode="text", file_path=str(items), server=server)

    def _shared_server(self, items: Path, num_products: Any) -> Any:
        """Build-once-per-process SimServer shared across adapter instances.

        ``WebAgentTextEnv`` accepts an injected server (official
        ``web_agent_text_env.py`` ``server=`` parameter), and every mutable
        episode field on the server lives in ``user_sessions[session_id]`` —
        the adapter pins goals/prices onto the shared server and keys each
        episode by its goal index.  Cache key covers the items file and the
        product-count flavor so distinct datasets never share a server.
        """
        key = f"{items}|{num_products}"
        with _WEBSHOP_SERVER_LOCK:
            cached = _WEBSHOP_SERVER_CACHE.get(key)
            if cached is not None:
                server, digest = cached
                if digest == digest_of_file(str(items)):
                    return server
                _WEBSHOP_SERVER_CACHE.pop(key, None)
            from web_agent_site.envs.web_agent_text_env import SimServer

            server = SimServer(
                "http://127.0.0.1:3000",
                str(items),
                None,
                -1,
                num_products,
                True,
                False,
            )
            _WEBSHOP_SERVER_CACHE[key] = (server, digest_of_file(str(items)))
            return server

    def _pin_frozen_state(self) -> None:
        """Force the official server onto the frozen goals/prices (digest-pinned)."""
        server = self._env.server
        server.goals = [dict(g) for g in self._goals]
        server.product_prices = dict(self._prices)
        server.weights = [g.get("weight", 1) for g in server.goals]
        cum = [0.0]
        for w in server.weights:
            cum.append(cum[-1] + float(w))
        server.cum_weights = cum

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return WEBSHOP_TOOL_SPEC

    def step(self, action: Action) -> StepResult:
        self._require_started()
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(
                    content="finished",
                    provenance=ObservationProvenance(source_tool="finish", source_ref=self._task_id),
                ),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        if action.name == "search":
            query = action.arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return StepResult(
                    observation=Observation(content="query must be a non-empty string", ok=False,
                                            error_kind="model_invalid_action"),
                    done=False,
                )
            official = f"search[{query.strip()}]"
        elif action.name == "click":
            text = action.arguments.get("text")
            if not isinstance(text, str) or not text.strip():
                return StepResult(
                    observation=Observation(content="text must be a non-empty string", ok=False,
                                            error_kind="model_invalid_action"),
                    done=False,
                )
            official = f"click[{text.strip()}]"
        else:
            return StepResult(
                observation=Observation(content=f"unknown tool {action.name!r}", ok=False,
                                        error_kind="model_invalid_action"),
                done=False,
            )
        self._actions.append(official)
        try:
            obs_text, _reward, done, _info = self._env.step(official)
        except Exception as e:  # noqa: BLE001 - execution errors are observations
            return StepResult(
                observation=Observation(content=f"ENV_ERROR: {e!r}", ok=False, error_kind="tool_error"),
                done=False,
            )
        if done:
            self._bought = True
            self._record_purchase()
        obs = Observation(
            content=_redact_reward(str(obs_text)),
            provenance=ObservationProvenance(source_tool=action.name, source_ref=self._session),
        )
        if len(self._actions) >= self._max_steps:
            done = True
        return StepResult(observation=obs, done=bool(done), snapshot=self.snapshot(), cost={"tool_calls": 1})

    def _record_purchase(self) -> None:
        """Capture the scorer-side purchase record (official get_reward inputs).

        Product fields are environment state (the actor already saw the page);
        the *goal* (attributes/options/price bound) is NOT recorded here.
        """
        server = self._env.server
        session = server.user_sessions.get(self._env.session) or {}
        asin = session.get("asin")
        if not asin:
            self._purchase = None
            return
        product = dict(server.product_item_dict.get(asin) or {})
        keep = {
            k: product.get(k)
            for k in ("asin", "query", "name", "product_category", "Attributes", "Title", "BulletPoints", "Description")
        }
        self._purchase = {
            "asin": asin,
            "options": dict(session.get("options") or {}),
            "price": server.product_prices.get(asin),
            "product": keep,
        }

    def snapshot(self) -> SnapshotRef:
        self._require_started()
        return SnapshotRef(
            ref_id=f"replay:{len(self._actions)}",
            digest=self._replay_digest(self._actions),
            kind="replay_prefix",
        )

    def _replay_digest(self, actions: list[str]) -> str:
        return digest_of({"task": self._task_id, "goal_index": self._goal_index, "actions": list(actions)})

    def fork(self, snapshot: SnapshotRef) -> "WebShopAdapter":
        """Replay-prefix fork AT THE SNAPSHOT POSITION (PR-06, guide §6.3).

        The pre-fix implementation replayed the parent's FULL action log —
        forking an old decision silently produced the CURRENT state.
        ``ref_id`` is ``replay:<n>`` (see ``snapshot()``): replay exactly
        ``n`` actions and verify the digest."""
        child = WebShopAdapter(str(self._root), max_steps=self._max_steps, out_root=str(self._out_root))
        child._task_id = self._task_id
        child._goal_index = self._goal_index
        # With a shared SimServer the session key must be unique per live adapter:
        # the official reset keys episode state at user_sessions[session_id], and
        # reusing the parent's key would replay on top of the parent's live state
        # (receive() keeps an existing session).  Lend the parent's session out,
        # let the child reset cleanly on the int goal index (session_int selects
        # goals[idx]; a string session falls back to random_idx and would replay
        # the wrong goal), rename the child's fresh session to a fork-unique key,
        # then return the parent's session untouched.
        child._fork_gen = getattr(self, "_fork_gen", 0) + 1
        child._env = child._build_env()
        child._pin_frozen_state()
        parent_key = str(self._goal_index)
        child._session = f"{self._goal_index}f{child._fork_gen}"
        server = child._env.server
        lent: dict[str, Any] | None = None
        if parent_key in server.user_sessions:
            lent = server.user_sessions.pop(parent_key)
        try:
            child._env.reset(session=self._goal_index)
            if parent_key in server.user_sessions:
                server.user_sessions[child._session] = server.user_sessions.pop(parent_key)
            else:
                server.user_sessions.setdefault(child._session, {})
        finally:
            if lent is not None:
                server.user_sessions[parent_key] = lent
        child._env.session = child._session
        browser = getattr(child._env, "browser", None)
        if browser is not None:
            browser.session_id = child._session
        child._started = True
        # snapshot position (PR-06): explicit from the ref, falling back to
        # the parent cursor for legacy refs.
        replay_len = len(self._actions)
        if snapshot.ref_id.startswith("replay:"):
            try:
                replay_len = int(snapshot.ref_id.split(":", 1)[1])
            except ValueError:
                pass
        replay_len = max(0, min(replay_len, len(self._actions)))
        for act in self._actions[:replay_len]:
            child.step_raw(act)
        if snapshot.kind == "replay_prefix" and snapshot.digest:
            if child._replay_digest(child._actions) != snapshot.digest:
                raise EnvironmentError_(
                    "fork replay digest mismatch: state not reproduced "
                    f"({snapshot.digest[:12]})"
                )
        return child

    def step_raw(self, official_action: str) -> None:
        """Replay-only: execute one official action string (kept off the tool path)."""
        self._actions.append(official_action)
        _obs, _reward, done, _info = self._env.step(official_action)
        if done:
            self._bought = True
            self._record_purchase()

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        return self._replay_digest(self._actions)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._env = None
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "data_root": str(self._root),
            "items_file": self._manifest.get("items_file"),
            "goals_sha256": self._manifest.get("goals_sha256"),
            "webshop_src_commit": self._manifest.get("webshop_src_commit"),
            "tool_schema_hash": digest_of([t.digest() for t in WEBSHOP_TOOL_SPEC]),
            "supports_shadow_fork": True,
            "synthetic": False,
        }

    def final_artifact(self) -> ArtifactRef:
        """Write the scorer-side purchase record; return a digest-pinned ref."""
        record = {
            "task_id": self._task_id,
            "goal_index": self._goal_index,
            "actions": list(self._actions),
            "purchase": self._purchase,
            "bought": self._bought,
        }
        out_dir = self._out_root / (self._task_id.replace(":", "_").replace("/", "_") or "task")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "final_artifact.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False), "utf-8")
        return ArtifactRef(
            ref_id=f"webshop:{self._task_id}:{self._goal_index}",
            path=str(path),
            digest=digest_of(record),
            kind="purchase",
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("webshop adapter not reset")


class WebShopScorer:
    """Official WebShop evaluation (get_reward / harsh success) on the evaluator side.

    Calls the pinned official ``get_reward`` on the recorded purchase against
    the private goal.  Missing private record or missing purchase → 0 reward
    (the official protocol scores a non-purchase episode as 0), never a
    substituted metric.
    """

    scorer_version = f"webshop-get_reward-{WEBSHOP_VERSION}"

    def __init__(self, data_root: str | None = None):
        self._root = Path(data_root or os.environ.get("RSI_WEBSHOP_ROOT", DEFAULT_WEBSHOP_ROOT))
        self._private: dict[str, dict[str, Any]] = {}
        self._lazy_loaded = False

    def _lazy_load_private(self) -> None:
        """Pull the frozen goal gold into scorer memory on first evaluate.

        The panel runners never call ``load_private`` (that seam exists for
        callers that already hold provider records); the scorer lives on the
        evaluator side, so reading ``frozen_goals.json`` directly stays inside
        the privacy boundary — the gold never reaches the actor view.
        """
        if self._lazy_loaded:
            return
        try:
            _manifest, goals, _prices = _load_frozen(self._root)
        except Exception as e:  # noqa: BLE001 - surfaced by evaluate as loud error
            raise EnvironmentError_(f"scorer cannot load frozen goals from {self._root}: {e!r}") from e
        for i, g in enumerate(goals):
            self._private[f"webshop:g{i:05d}"] = dict(g)
        self._lazy_loaded = True

    def check_ready(self) -> tuple[bool, str]:
        try:
            _require_webshop_deps()
        except NotImplementedCapability as e:
            return False, str(e)
        try:
            import spacy

            spacy.load("en_core_web_sm")
        except Exception as e:  # noqa: BLE001
            return False, f"spacy model en_core_web_sm unavailable ({e!r})"
        if not (self._root / "frozen_goals.json").is_file():
            return False, "frozen_goals.json missing (run scripts/prepare_webshop_data.py)"
        return True, "ok"

    def load_private(self, task_id: str, goal_record: dict[str, Any]) -> None:
        """Evaluator-side only: gold attributes / options / price bound."""
        self._private[task_id] = goal_record

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        _require_webshop_deps()
        task_id = private_task_ref.ref_id
        if task_id not in self._private:
            self._lazy_load_private()
        if task_id not in self._private:
            raise EnvironmentError_(f"no private goal record loaded for {task_id}")
        goal = self._private[task_id]
        record: dict[str, Any] = {}
        if final_artifact.path and Path(final_artifact.path).is_file():
            record = json.loads(Path(final_artifact.path).read_text("utf-8"))
        purchase = record.get("purchase")
        if not purchase or not purchase.get("asin"):
            # official protocol: no purchase → reward 0
            return ScoreRecord(
                task_id=task_id,
                domain="webshop",
                metric_name="task_reward",
                value=0.0,
                success=False,
                details={"purchased": False},
                scorer_version=self.scorer_version,
                synthetic=False,
            )
        from web_agent_site.engine.goal import get_reward  # official metric

        reward = float(
            get_reward(
                purchase["product"],
                goal,
                price=purchase.get("price"),
                options=purchase.get("options") or {},
            )
        )
        value = max(0.0, min(1.0, reward))
        return ScoreRecord(
            task_id=task_id,
            domain="webshop",
            metric_name="task_reward",
            value=value,
            success=reward == 1.0,
            details={
                "purchased": True,
                "asin": purchase.get("asin"),
                "raw_reward": reward,
                "harsh_success": reward == 1.0,
            },
            scorer_version=self.scorer_version,
            synthetic=False,
        )
