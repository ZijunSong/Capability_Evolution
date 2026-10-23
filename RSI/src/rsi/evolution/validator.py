"""Candidate validation: schema, import, protocol, executability, resource and
immutable-boundary checks.

AST scanning is an auxiliary audit, not a Python security sandbox — real
isolation is the candidate worker process/container (``rsi.runtime.sandbox``).
Candidates cannot modify environments, splits, evaluators, rewards, checkpoint
selection rules or H0; ``editable_scope=bank_only`` is enforced by path checks
here and by mounts in the sandbox.

Dataset task ids may appear in *reports* for audit, never as constants in
candidate code (conditional branches / answer banks are forbidden).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue
from rsi.contracts.errors import CandidateInvalid

FORBIDDEN_MANIFEST_KEYS = (
    "environments",
    "splits",
    "evaluator",
    "reward",
    "checkpoint_policy",
    "h0",
    "scorer",
)

FORBIDDEN_IMPORT_PREFIXES = (
    "trim",
    "scape",
    "scope",
    "harness1",
)

IMMUTABLE_PATH_PREFIXES = (
    "environments/",
    "evaluation/",
    "training/",
    "runtime/",
    "contracts/",
    "harness/",
    "evolution/",
    "opd/",
    "campaign/",
    "providers/",
    "configs/",
    "data/",
)


@dataclass
class ValidationReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return {"ok": self.ok, "problems": self.problems, "checks": self.checks}


def _check_paths(paths: Sequence[str], problems: list[str]) -> None:
    for rel in paths:
        norm = rel.replace("\\", "/")
        if norm.startswith("/") or ".." in norm.split("/"):
            problems.append(f"path escapes bank root: {rel}")
            continue
        for prefix in IMMUTABLE_PATH_PREFIXES:
            if norm.startswith(prefix):
                problems.append(f"path in immutable scope: {rel}")
        if not (norm == "manifest.json" or norm.startswith("components/")):
            problems.append(f"path outside editable scope: {rel}")


def _check_task_id_constants(source: str, known_task_ids: Sequence[str], problems: list[str]) -> None:
    if not known_task_ids:
        return
    for tid in known_task_ids:
        if tid and tid in source:
            problems.append(f"dataset task id constant found in candidate code: {tid}")


def _check_imports_and_structure(source: str, filename: str, problems: list[str], resource_caps: Mapping[str, Any]) -> None:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        problems.append(f"{filename}: syntax error: {e}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_IMPORT_PREFIXES:
                    problems.append(f"{filename}: forbidden legacy import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in FORBIDDEN_IMPORT_PREFIXES:
                problems.append(f"{filename}: forbidden legacy import {node.module}")
        elif isinstance(node, ast.Call):
            f = node.func
            # Only bare-name dynamic execution is forbidden (re.compile etc. are fine).
            if isinstance(f, ast.Name) and f.id in {"eval", "exec", "compile", "__import__"}:
                problems.append(f"{filename}: forbidden dynamic execution call {f.id}()")
            elif isinstance(f, ast.Name) and f.id == "open":
                # open() is allowed for component-private state but flagged for review
                problems.append(f"{filename}: uses open() (review: component private state only)")
    # resource caps: max file bytes
    max_bytes = int(resource_caps.get("max_source_bytes", 200_000))
    if len(source.encode("utf-8")) > max_bytes:
        problems.append(f"{filename}: source exceeds resource cap ({max_bytes} bytes)")


def validate_candidate_files(
    *,
    files: Mapping[str, str],
    manifest_patch: Mapping[str, JSONValue],
    known_task_ids: Sequence[str] = (),
    resource_caps: Mapping[str, Any] | None = None,
) -> ValidationReport:
    problems: list[str] = []
    caps = dict(resource_caps or {})
    _check_paths(list(files.keys()), problems)
    for key in FORBIDDEN_MANIFEST_KEYS:
        if key in manifest_patch:
            problems.append(f"manifest_patch touches immutable key {key!r}")
    for rel, content in files.items():
        if rel == "manifest.json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                problems.append(f"manifest.json invalid: {e}")
            continue
        _check_imports_and_structure(content, rel, problems, caps)
        _check_task_id_constants(content, known_task_ids, problems)
    # entrypoint resolution sanity for single-component patches
    entry = manifest_patch.get("entrypoint") if isinstance(manifest_patch, dict) else None
    if isinstance(entry, str) and ":" in entry:
        mod = entry.split(":", 1)[0].replace(".", "/") + ".py"
        if mod not in files and not any(f.endswith(mod) for f in files):
            problems.append(f"manifest entrypoint module {mod!r} not in candidate files")
    return ValidationReport(ok=not problems, problems=problems, checks={"n_files": len(files)})


def validate_materialized(bank_dir: Path, *, known_task_ids: Sequence[str] = ()) -> ValidationReport:
    """Post-materialization audit of the resolved candidate tree."""
    problems: list[str] = []
    manifest_path = bank_dir / "manifest.json"
    if not manifest_path.is_file():
        return ValidationReport(ok=False, problems=["manifest.json missing"])
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for key in FORBIDDEN_MANIFEST_KEYS:
        if key in manifest:
            problems.append(f"manifest contains immutable key {key!r}")
    files: dict[str, str] = {}
    for comp in manifest.get("components", []):
        for rel in comp.get("files", []):
            p = bank_dir / rel
            if not p.is_file():
                problems.append(f"missing component file {rel}")
            else:
                files[rel] = p.read_text("utf-8")
    report = validate_candidate_files(
        files=files,
        manifest_patch={},
        known_task_ids=known_task_ids,
    )
    problems.extend(report.problems)
    return ValidationReport(ok=not problems, problems=problems, checks=report.checks)
