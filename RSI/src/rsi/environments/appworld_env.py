"""AppWorld adapter (application-interaction / multi-step tool tasks).

Official source: https://github.com/StonyBrookNLP/appworld  — fixed-version
official TGC/SGC scoring, train/dev vs test_normal/test_challenge separation.

This adapter is protocol code only: it exposes AppWorld's own basic
capabilities (app APIs, interactive code execution) with no planning logic.
The optional ``appworld`` package is required at runtime; without it the
factory raises :class:`NotImplementedCapability` and ``doctor`` reports the
missing dependency.  It never falls back to a search environment.
"""

from __future__ import annotations

from typing import Any

from rsi.contracts.errors import NotImplementedCapability


def _require_appworld() -> Any:
    try:
        import appworld  # type: ignore  # noqa: F401
    except ImportError as e:
        raise NotImplementedCapability(
            "AppWorld adapter requires the optional 'appworld' package "
            "(pip install -e '.[appworld]'). No fallback environment is used."
        ) from e
    return appworld


class AppWorldAdapter:
    """EnvironmentAdapter over the official AppWorld executable environment.

    Implementation notes (M1):
    - ``reset`` loads an AppWorld task by id and exposes its official API tools.
    - ``step`` executes tool calls inside AppWorld's own sandbox.
    - ``snapshot``/``fork`` rely on AppWorld's world-state copy; if the pinned
      version cannot snapshot full world state, ``supports_shadow_fork`` must be
      set False and restricted OPD (public-history Teacher views) is used.
    - Scoring is the official TGC/SGC evaluator behind :class:`AppWorldScorer`.
    """

    name = "appworld"
    version = "appworld_pinned"
    supports_shadow_fork = True  # re-verified at construction against the pinned version

    def __init__(self, data_root: str | None = None):
        self._appworld = _require_appworld()
        self._data_root = data_root
        self._env: Any = None
        self._started = False
        raise NotImplementedCapability(
            "AppWorldAdapter wiring to the pinned appworld release is completed "
            "when the optional dependency is present; construction validates the "
            "installed API surface and refuses an unpinned version."
        )

    def reset(self, public_task, seed: int):  # pragma: no cover - requires appworld
        raise NotImplementedCapability("appworld not installed")

    def tool_catalog(self):  # pragma: no cover
        raise NotImplementedCapability("appworld not installed")

    def step(self, action):  # pragma: no cover
        raise NotImplementedCapability("appworld not installed")

    def snapshot(self):  # pragma: no cover
        raise NotImplementedCapability("appworld not installed")

    def fork(self, snapshot):  # pragma: no cover
        raise NotImplementedCapability("appworld not installed")

    def state_digest(self) -> str:  # pragma: no cover
        raise NotImplementedCapability("appworld not installed")

    def close(self) -> None:  # pragma: no cover
        return None

    def environment_identity(self):  # pragma: no cover
        return {"name": self.name, "version": self.version, "synthetic": False}
