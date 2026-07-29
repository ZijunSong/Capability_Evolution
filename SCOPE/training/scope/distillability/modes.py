"""E0 distillability probe modes (OFF / PROC / FULL)."""

from __future__ import annotations

from enum import Enum


class DistillabilityMode(str, Enum):
  OFF = "off"
  PROC = "proc"
  FULL = "full"
