"""Load the campaign YAML. Numbers are pilot bounds, not tuned optima."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED = (
    "schema_version",
    "campaign_id",
    "seed",
    "max_rounds",
    "base",
    "policy",
    "data",
    "evolution",
    "teacher_gate",
    "complexity",
    "training",
    "statistics",
    "promotion",
    "launch",
    "budget",
)


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RSI config must be a mapping")
    if payload.get("schema_version") != "trim_rsi_v1":
        raise ValueError("schema_version must be trim_rsi_v1")
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"RSI config missing keys: {missing}")
    if int(payload["max_rounds"]) < 1:
        raise ValueError("max_rounds must be positive")
    if payload["training"].get("method") != "trim":
        raise ValueError("RSI training.method must be trim")
    reward = payload["training"].get("rl_reward")
    if reward not in {"final-answer-recall", "harness1-composite"}:
        raise ValueError("RSI rl_reward must be an explicit public option")
    if payload["evolution"].get("seed_mode") not in {"empty", "legacy"}:
        raise ValueError("seed_mode must be empty or legacy")
    budget = payload["budget"]
    for key in ("max_total_rollout_episodes", "max_total_generated_tokens", "max_total_gpu_hours", "max_wall_clock_hours"):
        if key not in budget:
            raise ValueError(f"budget.{key} is required")
    return payload
