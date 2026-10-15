"""Path jail for bank files. AST checks are not a substitute for this jail."""

from __future__ import annotations

import ast
from pathlib import Path

from trim.harness_bank.loader import BankLoadError


_BANNED_CALLS = {"system", "popen", "exec", "eval", "spawn"}
_BANNED_MODULES = {"subprocess", "socket", "ctypes", "multiprocessing"}


def assert_inside(root: Path, path: Path) -> None:
    root_r = root.resolve()
    path_r = path.resolve()
    try:
        path_r.relative_to(root_r)
    except ValueError as exc:
        raise BankLoadError(f"path escapes bank sandbox: {path}") from exc
    if path.is_symlink() and not path_r.is_relative_to(root_r):
        raise BankLoadError(f"symlink escapes bank sandbox: {path}")


def scan_component_ast(source: str, *, label: str) -> None:
    """Reject obvious process and network escapes. This does not make exec safe."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise BankLoadError(f"{label} is not valid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BANNED_MODULES:
                    raise BankLoadError(f"{label} imports banned module {root}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _BANNED_MODULES:
                raise BankLoadError(f"{label} imports banned module {root}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _BANNED_CALLS:
                raise BankLoadError(f"{label} calls banned function {name}")


def reject_unjailed_exec(root: Path, source_path: Path) -> None:
    assert_inside(root, source_path)
    scan_component_ast(source_path.read_text(encoding="utf-8"), label=str(source_path))
