"""SHA256SUMS write / verify."""

from __future__ import annotations

from pathlib import Path

from scape.common.hashing import sha256_file


def write_sha256sums(root: Path, files: list[Path], *, out_name: str = "SHA256SUMS") -> Path:
    lines: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        rel = path.relative_to(root) if path.is_absolute() and root in path.parents else path.name
        lines.append(f"{sha256_file(path)}  {rel}")
    out = root / out_name
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def verify_sha256sums(sums_path: Path, *, root: Path | None = None) -> list[str]:
    """Return list of error strings; empty means OK."""
    base = root or sums_path.parent
    errors: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            digest, _, rel = line.partition(" ")
        path = base / rel.strip()
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        got = sha256_file(path)
        if got != digest.strip():
            errors.append(f"mismatch: {rel}")
    return errors
