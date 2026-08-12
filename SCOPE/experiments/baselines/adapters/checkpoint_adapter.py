"""Checkpoint path normalization for baselines (no silent Base fallback)."""

from __future__ import annotations

from pathlib import Path

from experiments.common.provenance import MissingAssetError, require_checkpoint


def resolve_checkpoint(path: str | None, *, allow_base: bool = False) -> str | None:
    resolved = require_checkpoint(path, allow_base=allow_base)
    return None if resolved is None else str(resolved)


def assert_not_overwriting_historical(path: Path) -> None:
    s = str(path)
    for banned in (
        "outputs/scope_round2",
        "outputs/scope_round3",
        "outputs/scope_round4",
        "outputs/scope_round5",
        "outputs/scope_round6",
        "outputs/scope_round7",
        "outputs/scope_round8",
        "outputs/scope_round9",
    ):
        if banned in s:
            raise MissingAssetError(f"refusing to write into historical outputs: {path}")
