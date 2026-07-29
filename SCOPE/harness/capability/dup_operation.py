"""Compact Dup capability operations (Round 2)."""

from __future__ import annotations

from enum import Enum


class DupOperation(str, Enum):
    KEEP_EVIDENCE = "KEEP_EVIDENCE"
    SKIP_DUPLICATE = "SKIP_DUPLICATE"
