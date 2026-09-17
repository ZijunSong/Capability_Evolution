#!/usr/bin/env python3
"""Segmented wall-clock helper for a frozen training step (audit T2/T5).

This is a measurement entry point, not a second training engine. It records
monotonic spans for prepare / generate / tool / snapshot / train phases when
the caller supplies timestamps. Same-workload comparison belongs in T5.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


PHASE_KEYS = (
    "rollout_prepare_s",
    "rollout_wall_s",
    "prompt_render_s",
    "generate_s",
    "tool_s",
    "snapshot_s",
    "train_prepare_s",
    "actor_forward_backward_s",
    "opd_teacher_s",
    "optimizer_s",
    "weight_sync_s",
    "checkpoint_publish_s",
    "phase_transition_wait_s",
)


def empty_step_profile(*, step: int = 0, policy_version: str = "") -> dict[str, Any]:
    payload = {k: 0.0 for k in PHASE_KEYS}
    payload.update(
        {
            "step": int(step),
            "policy_version": policy_version,
            "wall_step_s": 0.0,
            "n_queries": 0,
            "n_episodes": 0,
            "n_decisions": 0,
            "n_rl_tokens": 0,
            "n_opd_tokens": 0,
            "n_optimizer_steps": 0,
            "clock": "monotonic",
        }
    )
    return payload


class StepProfiler:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.spans: dict[str, float] = {k: 0.0 for k in PHASE_KEYS}

    def add(self, name: str, seconds: float) -> None:
        if name not in self.spans:
            raise KeyError(f"unknown span {name}")
        self.spans[name] += float(seconds)

    def finish(self, **counts: Any) -> dict[str, Any]:
        out = empty_step_profile()
        out.update(self.spans)
        out["wall_step_s"] = time.perf_counter() - self.t0
        for key, value in counts.items():
            out[key] = value
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = empty_step_profile()
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
