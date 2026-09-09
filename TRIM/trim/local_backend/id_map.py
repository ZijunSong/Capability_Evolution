"""Local corpus ID mapping so official IDs with underscores stay recoverable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


MAP_VERSION = "idmap_v1"


@dataclass
class IdMap:
    version: str = MAP_VERSION
    official_to_internal: dict[str, str] = field(default_factory=dict)
    internal_to_official: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "official_to_internal": dict(self.official_to_internal),
            "internal_to_official": dict(self.internal_to_official),
        }

    def register_official(self, official_id: str) -> str:
        official = str(official_id)
        if official in self.official_to_internal:
            return self.official_to_internal[official]
        if "_" not in official:
            internal = official
        else:
            internal = f"d{len(self.official_to_internal):06d}"
            while internal in self.internal_to_official:
                internal = f"d{len(self.official_to_internal) + len(self.internal_to_official):06d}"
        self.official_to_internal[official] = internal
        self.internal_to_official[internal] = official
        return internal

    def internal_of(self, official_id: str) -> str:
        official = str(official_id)
        if official in self.official_to_internal:
            return self.official_to_internal[official]
        return self.register_official(official)

    def official_of(self, token: str) -> str:
        text = str(token)
        if text in self.official_to_internal:
            return text
        if text in self.internal_to_official:
            return self.internal_to_official[text]
        root = text.rsplit("_", 1)[0] if "_" in text else text
        if root in self.internal_to_official:
            return self.internal_to_official[root]
        if root in self.official_to_internal:
            return root
        return text

    def to_official_list(self, ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in ids:
            official = self.official_of(str(item))
            if official not in seen:
                seen.add(official)
                out.append(official)
        return out

    def chunk_id(self, official_id: str, ordinal: int) -> str:
        return f"{self.internal_of(official_id)}_{int(ordinal)}"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None) -> "IdMap":
        if path is None or not Path(path).is_file():
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=str(payload.get("version") or MAP_VERSION),
            official_to_internal={str(k): str(v) for k, v in (payload.get("official_to_internal") or {}).items()},
            internal_to_official={str(k): str(v) for k, v in (payload.get("internal_to_official") or {}).items()},
        )

    @classmethod
    def from_official_ids(cls, official_ids: Iterable[str], existing: Mapping[str, str] | None = None) -> "IdMap":
        mapping = cls()
        if existing:
            for official, internal in existing.items():
                mapping.official_to_internal[str(official)] = str(internal)
                mapping.internal_to_official[str(internal)] = str(official)
        for official in official_ids:
            mapping.register_official(str(official))
        return mapping
