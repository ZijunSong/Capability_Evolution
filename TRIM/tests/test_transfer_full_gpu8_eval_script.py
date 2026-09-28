"""Sanity checks for the transfer 3×3×2 (18-job) GPU8 eval launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_transfer_w_harness1_gpu8_eval.sh"


def _dry_run(env: dict[str, str] | None = None) -> str:
    merged = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_ID": "test_transfer_plan",
        "PY": "python3",
        **(env or {}),
    }
    return subprocess.check_output(["bash", str(SCRIPT)], env=merged, text=True)


def test_script_bash_syntax():
    subprocess.check_call(["bash", "-n", str(SCRIPT)])


def test_dry_run_emits_eighteen_jobs():
    out = _dry_run()
    starts = [line for line in out.splitlines() if line.startswith("JOB start ")]
    assert len(starts) == 18
    models = ("gpt-oss-20b", "harness-1", "Qwen3-4B-Instruct-2507")
    benches = ("longsealqa", "frames", "hotpotqa")
    components = ("zero", "all")
    expected = {
        f"model={model} component={component} benchmark={bench}"
        for model in models
        for component in components
        for bench in benches
    }
    got = set()
    for line in starts:
        parts = line.split()
        got.add(" ".join(parts[2:5]))
    assert got == expected


def test_harmony_jobs_use_reasoning_effort_and_4096():
    out = _dry_run()
    gpt_lines = [line for line in out.splitlines() if line.startswith("DRY_RUN ") and "gpt-oss-20b" in line]
    assert gpt_lines
    for line in gpt_lines:
        assert "--reasoning-effort medium" in line
        assert "--max-new-tokens 4096" in line
        assert "--component zero" in line or "--component all" in line
    qwen_lines = [
        line
        for line in out.splitlines()
        if line.startswith("DRY_RUN ") and "Qwen3-4B-Instruct-2507" in line
    ]
    assert qwen_lines
    for line in qwen_lines:
        assert "--reasoning-effort" not in line
        assert "--max-new-tokens 2048" in line


def test_all_jobs_pass_verifier_url():
    out = _dry_run()
    all_lines = [
        line
        for line in out.splitlines()
        if line.startswith("DRY_RUN ") and "--component all" in line
    ]
    assert len(all_lines) == 9
    for line in all_lines:
        assert "--verify-base-url http://127.0.0.1:8050/v1" in line
        assert "--verify-model harness-1-verifier" in line
    zero_lines = [
        line
        for line in out.splitlines()
        if line.startswith("DRY_RUN ") and "--component zero" in line
    ]
    assert len(zero_lines) == 9
    for line in zero_lines:
        assert "--verify-base-url" not in line


def test_zero_multigpu_emits_all_actor_urls():
    out = _dry_run(
        {
            "ZERO_ACTOR_GPUS": "0,3,4",
            "ZERO_ACTOR_PORTS": "8040,8042,8044",
            "TP": "12",
            "COMPONENTS": "zero",
            "MODELS": "gpt-oss-20b",
            "BENCHMARKS": "hotpotqa",
        }
    )
    dry = [line for line in out.splitlines() if line.startswith("DRY_RUN ")]
    assert len(dry) == 1
    assert "--api-base-url http://127.0.0.1:8040/v1,http://127.0.0.1:8042/v1,http://127.0.0.1:8044/v1" in dry[0]
    assert "--tp 12" in dry[0]


def test_skip_jobs_omits_named_job():
    out = _dry_run(
        {
            "SKIP_JOBS": "gpt-oss-20b/zero/frames",
            "MODELS": "gpt-oss-20b",
            "COMPONENTS": "zero",
        }
    )
    starts = [line for line in out.splitlines() if line.startswith("JOB start ")]
    skips = [line for line in out.splitlines() if line.startswith("JOB skip ")]
    assert len(starts) == 2
    assert len(skips) == 1
    assert "benchmark=frames" in skips[0]
    assert not any("benchmark=frames" in line for line in starts)


def test_models_filter_is_case_insensitive():
    out = _dry_run({"MODELS": "qwen3-4b-instruct-2507", "COMPONENTS": "zero", "BENCHMARKS": "hotpotqa"})
    starts = [line for line in out.splitlines() if line.startswith("JOB start ")]
    assert len(starts) == 1
    assert "Qwen3-4B-Instruct-2507" in starts[0]
    assert "component=zero" in starts[0]
    assert "benchmark=hotpotqa" in starts[0]
