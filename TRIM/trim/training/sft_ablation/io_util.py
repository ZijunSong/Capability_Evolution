"""Small atomic IO helpers. No model or Harness imports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json_dumps(payload) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.exists():
        return "missing"
    rows: list[str] = []
    for file in sorted(item for item in target.rglob("*") if item.is_file()):
        rel = file.relative_to(target).as_posix()
        rows.append(f"{rel}:{sha256_file(file)}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
