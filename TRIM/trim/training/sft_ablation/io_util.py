"""Small atomic IO helpers. No model or Harness imports."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


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


def iter_json_array_items(path: Path, key: str, *, chunk_size: int = 1024 * 1024) -> Iterator[tuple[int, int, int, Any]]:
    """Yield array index, byte range, and one decoded item at a time."""
    decoder = json.JSONDecoder()
    needle = json.dumps(str(key), ensure_ascii=False) + ":"
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    base = 0
    index = 0
    array_found = False
    eof = False
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            buffer += utf8.decode(chunk, final=not chunk)
            eof = not chunk
            if not array_found:
                marker = buffer.find(needle)
                if marker < 0:
                    if eof:
                        raise ValueError(f"JSON array key not found: {key}")
                    keep = max(len(needle) + 2, 64)
                    if len(buffer) > keep:
                        consumed = buffer[:-keep]
                        base += len(consumed.encode("utf-8"))
                        buffer = buffer[-keep:]
                    continue
                cursor = marker + len(needle)
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if eof:
                        raise ValueError(f"JSON array key not found: {key}")
                    continue
                if buffer[cursor] != "[":
                    raise ValueError(f"JSON value for {key} is not an array")
                consumed = buffer[:cursor + 1]
                base += len(consumed.encode("utf-8"))
                buffer = buffer[cursor + 1:]
                array_found = True
            while True:
                leading = len(buffer) - len(buffer.lstrip())
                if leading:
                    base += len(buffer[:leading].encode("utf-8"))
                    buffer = buffer[leading:]
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    base += 1
                    buffer = buffer[1:]
                    continue
                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                raw = buffer[:end]
                start = base
                finish = base + len(raw.encode("utf-8"))
                yield index, start, finish, item
                index += 1
                base = finish
                buffer = buffer[end:]
            if eof:
                raise ValueError(f"unterminated JSON array: {key}")


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
