"""SpreadsheetBench adapter — file & spreadsheet manipulation tasks.

Official source: https://github.com/RUCKBReasoning/SpreadsheetBench (complete
912-question release ``all_data_912_v0.1``; Verified-400 is a separate frozen
panel, never mixed).  Official metric: OJ-style execution/file-result
correctness — a question is solved only if EVERY test-case workbook passes.

Data layout (frozen at ``data prepare``):
``all_data_912_v0.1/dataset.json`` (id, instruction, spreadsheet_path,
answer_position, answer_sheet, data_position) plus per-question workbooks
``<k>_<id>_input.xlsx`` / ``<k>_<id>_answer.xlsx``.

Privacy boundary: **answer workbooks are gold** — they live only on the
evaluator side.  The actor workspace receives only the ``*_input.xlsx`` files.
The scorer compares the edited inputs against the answers at
``answer_position`` on ``answer_sheet`` for every test case.

Basic tools: ``python_exec`` (openpyxl/pandas), ``read_file``/``write_file``/
``list_dir`` over the workspace, plus finish.  No domain strategy ships in the
adapter.  Snapshot/fork = workspace filesystem snapshot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
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

SSB_VERSION = "spreadsheetbench-912-v0.1"
DEFAULT_SSB_ROOT = "/mnt/songzijun/datasets/spreadsheetbench"

SSB_TOOL_SPEC: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="python_exec",
        description="Execute a Python snippet in the workspace (openpyxl/pandas available) to manipulate the workbooks.",
        parameters_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        capabilities=("python_exec", "fs_write"),
    ),
    ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the workspace.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        capabilities=("fs_read",),
    ),
    ToolSpec(
        name="write_file",
        description="Write UTF-8 text to a workspace file.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        capabilities=("fs_write",),
    ),
    ToolSpec(
        name="list_dir",
        description="List workspace entries.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        capabilities=("fs_read",),
    ),
)


def _require_deps() -> Any:
    missing = []
    for mod in ("openpyxl",):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"SpreadsheetBench adapter requires optional packages {missing} "
            "(pip install -e '.[spreadsheetbench]'). No fallback environment is used."
        )


def _safe_path(workdir: Path, rel: str) -> Path:
    p = (workdir / rel).resolve()
    try:
        p.relative_to(workdir.resolve())
    except ValueError as e:
        raise EnvironmentError_(f"path escapes workspace: {rel}") from e
    return p


def load_ssb_questions(root: Path) -> list[dict[str, Any]]:
    qfile = root / "all_data_912_v0.1" / "dataset.json"
    if not qfile.is_file():
        raise NotImplementedCapability(
            f"SpreadsheetBench dataset.json not found at {qfile} — run data prepare"
        )
    return list(json.loads(qfile.read_text("utf-8")))


class SpreadsheetBenchAdapter:
    """EnvironmentAdapter over one SpreadsheetBench question workspace."""

    name = "spreadsheetbench"
    version = SSB_VERSION
    supports_shadow_fork = True

    def __init__(self, dataset_root: str | None = None, *, shell_timeout_s: int = 120):
        _require_deps()
        self._root = Path(dataset_root or os.environ.get("RSI_SSB_ROOT", DEFAULT_SSB_ROOT))
        self._shell_timeout_s = int(shell_timeout_s)
        self._workdir: Path | None = None
        self._qid = ""
        self._snap_counter = 0
        self._started = False

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        qid = str(public_task.metadata.get("question_id") or public_task.task_id)
        src = self._root / "all_data_912_v0.1" / "spreadsheet" / qid
        if not src.is_dir():
            raise EnvironmentError_(f"question workbooks missing: {src}")
        self._qid = qid
        self._workdir = Path(tempfile.mkdtemp(prefix=f"rsi_ssb_{qid}_"))
        # Actor sees ONLY the input workbooks (answers are gold/private).
        for f in sorted(src.iterdir()):
            if f.name.endswith("_input.xlsx"):
                shutil.copy2(f, self._workdir / f.name)
        self._started = True
        return Observation(
            content=(
                f"Workspace ready with input workbooks: "
                f"{sorted(x.name for x in self._workdir.iterdir())}. "
                f"Task: {public_task.instruction}"
            ),
            provenance=ObservationProvenance(source_tool="reset", source_ref=qid),
        )

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return SSB_TOOL_SPEC

    def step(self, action: Action) -> StepResult:
        self._require_started()
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(content="finished"),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        try:
            if action.name == "python_exec":
                code = str(action.arguments.get("code", ""))
                proc = subprocess.run(
                    ["python", "-c", code], cwd=str(self._workdir), capture_output=True,
                    text=True, timeout=self._shell_timeout_s,
                )
                out = (proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")).strip()
                obs = Observation(
                    content=out[:20000] or "(no output)",
                    ok=proc.returncode == 0,
                    error_kind=None if proc.returncode == 0 else "tool_error",
                    provenance=ObservationProvenance(source_tool="python_exec", source_ref=self._qid),
                )
            elif action.name == "read_file":
                p = _safe_path(self._workdir, str(action.arguments.get("path", "")))
                obs = Observation(
                    content=p.read_text("utf-8")[:20000] if p.is_file() else f"no such file: {p.name}",
                    ok=p.is_file(),
                    error_kind=None if p.is_file() else "tool_error",
                    provenance=ObservationProvenance(source_tool="read_file", source_ref=str(p.name)),
                )
            elif action.name == "write_file":
                p = _safe_path(self._workdir, str(action.arguments.get("path", "")))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(action.arguments.get("content", "")), "utf-8")
                obs = Observation(
                    content=f"wrote {p.name}",
                    provenance=ObservationProvenance(source_tool="write_file", source_ref=str(p.name)),
                )
            elif action.name == "list_dir":
                p = _safe_path(self._workdir, str(action.arguments.get("path", "")) or ".")
                names = sorted(x.name for x in p.iterdir())[:200] if p.is_dir() else []
                obs = Observation(
                    content=str(names),
                    provenance=ObservationProvenance(source_tool="list_dir", source_ref=str(p.name)),
                )
            else:
                obs = Observation(content=f"unknown tool {action.name!r}", ok=False, error_kind="model_invalid_action")
        except Exception as e:  # noqa: BLE001
            obs = Observation(content=f"TOOL_ERROR: {e!r}", ok=False, error_kind="tool_error")
        return StepResult(observation=obs, done=False, snapshot=self.snapshot(), cost={"tool_calls": 1})

    def snapshot(self) -> SnapshotRef:
        self._require_started()
        self._snap_counter += 1
        tar_path = self._workdir / f"snap{self._snap_counter}.tar"
        with tarfile.open(tar_path, "w:gz") as tf:
            for f in sorted(self._workdir.iterdir()):
                if f.name.endswith(".xlsx"):
                    tf.add(f, arcname=f.name)
        return SnapshotRef(ref_id=str(tar_path), digest=digest_of(str(tar_path)), kind="full")

    def fork(self, snapshot: SnapshotRef) -> "SpreadsheetBenchAdapter":
        child = SpreadsheetBenchAdapter(str(self._root), shell_timeout_s=self._shell_timeout_s)
        child._qid = self._qid
        child._workdir = Path(tempfile.mkdtemp(prefix="rsi_ssb_fork_"))
        with tarfile.open(snapshot.ref_id, "r:gz") as tf:
            tf.extractall(child._workdir)
        child._started = True
        return child

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        return digest_of({"q": self._qid, "n": self._snap_counter})

    def close(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_root": str(self._root),
            "tool_schema_hash": digest_of([t.digest() for t in SSB_TOOL_SPEC]),
            "supports_shadow_fork": True,
            "synthetic": False,
        }

    def final_artifact(self) -> ArtifactRef:
        # The edited input workbooks ARE the artifact; digest them as a set and
        # hand the evaluator the workspace path (scored before env.close()).
        digests = {}
        if self._started:
            for f in sorted(self._workdir.iterdir()):
                if f.name.endswith(".xlsx"):
                    digests[f.name] = digest_of(f.read_bytes().hex())
        return ArtifactRef(
            ref_id=f"ssb:{self._qid}",
            kind="workbooks",
            path=str(self._workdir) if self._started else None,
            digest=digest_of(digests),
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("spreadsheetbench adapter not reset")


class SpreadsheetBenchScorer:
    """Official OJ-style workbook comparison (evaluator side only).

    For every test case k: compare ``answer_position`` cells on ``answer_sheet``
    between the actor's edited ``<k>_<id>_input.xlsx`` and the gold
    ``<k>_<id>_answer.xlsx``.  A question is solved only if ALL test cases
    match (official OJ metric); per-case results are recorded in details.
    """

    scorer_version = f"spreadsheetbench-oj-{SSB_VERSION}"

    def __init__(self, dataset_root: str | None = None):
        _require_deps()
        self._root = Path(dataset_root or os.environ.get("RSI_SSB_ROOT", DEFAULT_SSB_ROOT))
        self._questions: dict[str, dict[str, Any]] = {}
        for q in load_ssb_questions(self._root):
            self._questions[str(q["id"])] = q

    @staticmethod
    def _parse_spec(answer_position: str, answer_sheet: str | None) -> list[tuple[str | None, str]]:
        """Parse the official answer spec grammar.

        - ``"A3:D32"`` / ``"K6"`` — range/cell on ``answer_sheet`` (or the
          workbook's first sheet when the question carries no sheet)
        - ``"Sheet1'!A1:M237"`` / ``"'S2'!A:G"`` — sheet-qualified segments
        - comma-separated multi-segment specs are supported
        """
        out: list[tuple[str | None, str]] = []
        for part in answer_position.split(","):
            part = part.strip().strip('"')
            if "'!" in part:
                sheet, rng = part.split("'!", 1)
                out.append((sheet.strip("'") or None, rng.strip()))
            elif "!" in part:
                sheet, rng = part.split("!", 1)
                out.append((sheet.strip("'") or None, rng.strip()))
            else:
                out.append((answer_sheet or None, part))
        return out

    @staticmethod
    def _cells(path: Path, spec: list[tuple[str | None, str]]) -> list[list[str]]:
        from openpyxl import load_workbook

        wb = load_workbook(path, data_only=True)
        values: list[list[str]] = []
        for sheet, rng in spec:
            if sheet is not None and sheet not in wb.sheetnames:
                return []  # sheet mismatch == not solved
            ws = wb[sheet] if sheet is not None else wb.worksheets[0]
            try:
                rows = ws[rng]
            except Exception:  # noqa: BLE001
                return []
            if not isinstance(rows, tuple):
                rows = ((rows,),)
            elif rows and not isinstance(rows[0], tuple):
                rows = (rows,)
            for row in rows:
                values.append(
                    [("" if c.value is None else str(c.value).strip()) for c in row]
                )
        return values

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        qid = private_task_ref.ref_id.split(":", 1)[-1]
        meta = self._questions.get(qid)
        if meta is None:
            raise EnvironmentError_(f"unknown SpreadsheetBench question {qid}")
        ans_sheet = str(meta.get("answer_sheet") or "") or None
        ans_pos = str(meta.get("answer_position", ""))
        spec = self._parse_spec(ans_pos, ans_sheet)
        src = self._root / "all_data_912_v0.1" / "spreadsheet" / qid
        # Actor workspace path travels on the artifact (evaluator runs before
        # env.close(); the workdir is still on disk).
        workdir = Path(final_artifact.path or "")
        cases: list[bool] = []
        n_cases = 0
        for ans in sorted(src.glob("*_answer.xlsx")):
            k = ans.name.split("_", 1)[0]
            inp = workdir / f"{k}_{qid}_input.xlsx"
            if not inp.is_file():
                cases.append(False)
                n_cases += 1
                continue
            n_cases += 1
            got = self._cells(inp, spec)
            want = self._cells(ans, spec)
            cases.append(bool(got) and got == want)
        solved = bool(cases) and all(cases)
        return ScoreRecord(
            task_id=private_task_ref.ref_id,
            domain="spreadsheetbench",
            metric_name="execution_correctness",
            value=1.0 if solved else 0.0,
            success=solved,
            details={
                "n_test_cases": n_cases,
                "n_passed": sum(cases),
                "per_case": cases,
                "answer_position": ans_pos,
                "answer_sheet": ans_sheet,
            },
            scorer_version=self.scorer_version,
            synthetic=False,
        )
