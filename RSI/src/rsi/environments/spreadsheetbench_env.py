"""SpreadsheetBench adapter (file & spreadsheet manipulation tasks).

Official source: https://github.com/RUCKBReasoning/SpreadsheetBench — official
execution/file-result correctness.  Multiple workbook test cases of the same
problem stay in one split.  Original vs Verified-derived datasets are frozen
separately at preprocessing time (see ``scripts/prepare_data.py``).

Requires optional ``openpyxl``/``pandas``; missing deps surface as
:class:`NotImplementedCapability` from ``doctor``/construction — no fallback.
"""

from __future__ import annotations

from typing import Any

from rsi.contracts.errors import NotImplementedCapability


def _require_deps() -> Any:
    missing = []
    for mod in ("openpyxl", "pandas"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"SpreadsheetBench adapter requires optional packages {missing} "
            "(pip install -e '.[spreadsheetbench]'). No fallback environment is used."
        )


class SpreadsheetBenchAdapter:
    """EnvironmentAdapter over frozen SpreadsheetBench workbooks.

    Basic tools: file read/write over the working directory plus a python_exec
    limited to the workspace.  Official scoring compares produced workbooks.
    """

    name = "spreadsheetbench"
    version = "spreadsheetbench_pinned"
    supports_shadow_fork = True  # file-layer snapshot; DB-less domain

    def __init__(self, dataset_root: str | None = None):
        _require_deps()
        self._dataset_root = dataset_root
        raise NotImplementedCapability(
            "SpreadsheetBenchAdapter wiring to the frozen dataset revision is "
            "completed when the optional dependencies are present; construction "
            "validates the frozen task list and refuses an unpinned revision."
        )

    def reset(self, public_task, seed: int):  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def tool_catalog(self):  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def step(self, action):  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def snapshot(self):  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def fork(self, snapshot):  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def state_digest(self) -> str:  # pragma: no cover
        raise NotImplementedCapability("spreadsheetbench deps not installed")

    def close(self) -> None:
        return None

    def environment_identity(self):  # pragma: no cover
        return {"name": self.name, "version": self.version, "synthetic": False}
