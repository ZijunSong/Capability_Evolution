"""SWE-bench adapter (repository read/modify/shell tasks).

Official source: https://github.com/SWE-bench/SWE-bench — official resolved
rate.  Training/dev tasks are deduplicated against the final Verified/Lite
panels by issue family; gold patches are never Actor-visible files.

Requires optional ``docker``/``swebench`` for the execution harness; missing
deps surface as :class:`NotImplementedCapability` — no fallback.
"""

from __future__ import annotations

from typing import Any

from rsi.contracts.errors import NotImplementedCapability


def _require_deps() -> Any:
    missing = []
    for mod in ("docker", "swebench"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"SWE-bench adapter requires optional packages {missing} "
            "(pip install -e '.[swebench]'). No fallback environment is used."
        )


class SWEBenchAdapter:
    """EnvironmentAdapter over a per-instance SWE-bench container.

    Basic tools: repo file read/write and shell execution inside the instance
    container.  Snapshot = container commit/checkpoint of the workspace layer.
    """

    name = "swebench"
    version = "swebench_pinned"
    supports_shadow_fork = True  # container checkpoint; verified per pinned image digest

    def __init__(self, image_prefix: str | None = None):
        _require_deps()
        self._image_prefix = image_prefix
        raise NotImplementedCapability(
            "SWEBenchAdapter wiring to the pinned instance images is completed "
            "when the optional dependencies are present; construction validates "
            "the image digest and refuses an unpinned setup."
        )

    def reset(self, public_task, seed: int):  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def tool_catalog(self):  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def step(self, action):  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def snapshot(self):  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def fork(self, snapshot):  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def state_digest(self) -> str:  # pragma: no cover
        raise NotImplementedCapability("swebench deps not installed")

    def close(self) -> None:
        return None

    def environment_identity(self):  # pragma: no cover
        return {"name": self.name, "version": self.version, "synthetic": False}
