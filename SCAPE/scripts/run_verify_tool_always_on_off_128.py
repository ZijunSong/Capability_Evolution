#!/usr/bin/env python3
"""V8D_VERIFY_TOOL always-on/off fork over the frozen 128-state cohort."""
from __future__ import annotations

from pathlib import Path

import run_adaptive_rerank_always_on_off_128 as runner


runner.COMPONENT = "verify_tool"
runner.SOURCE_COMPONENT = "adaptive_rerank_instruction"
runner.ARTIFACT_PREFIX = "VERIFY_TOOL_ALWAYS_ON_OFF"
runner.SHARD_PREFIX = "VERIFY_TOOL_ALWAYS_ON_OFF"
runner.RUNNER_NAME = "verify_tool_always_on_off_128"
runner.SCHEMA_VERSION = "verify_tool_always_on_off_v1"
runner.OUT_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "outputs/0821_verify_tool_always_on_off_128"
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
