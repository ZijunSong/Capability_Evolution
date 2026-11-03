"""Parallel panel evaluation: multi-process workers merge to the same
EvalRunResult as the serial path (same per-task seeds).

CPU workers (``gpu_ids=[None, None]``) prove the shard/merge logic; GPU pinning
is exercised by the 8-GPU entry script (``scripts/entry_evolve_8gpu.sh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.config import load_config
from rsi.evaluation.parallel import resolve_worker_plan, run_panel_mp
from rsi.evaluation.service import run_panel
from rsi.harness.loader import load_empty_bank
from rsi.providers.factory import build_policy
from rsi.contracts.errors import ProtocolViolation

CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "toy_smoke.yaml"


def _cfg(tmp_path):
    cfg = load_config(CFG_PATH)
    cfg.raw = dict(cfg.raw)
    cfg.raw["execution"] = {**cfg.section("execution"), "runs_root": str(tmp_path / "runs")}
    cfg.raw["runtime"] = {**cfg.section("runtime"), "temperature": 0.0}  # greedy: deterministic
    return cfg


def test_worker_plan_guards():
    # workers > visible GPUs is refused; CPU-only + require_gpu is refused
    with pytest.raises(ProtocolViolation):
        resolve_worker_plan(workers=4, gpu_ids=[0, 1], require_gpu=True)
    with pytest.raises(ProtocolViolation):
        resolve_worker_plan(workers=2, gpu_ids=[], require_gpu=True)
    n, plan = resolve_worker_plan(workers=3, gpu_ids=[0, 1, 2], require_gpu=True)
    assert n == 3 and plan == [0, 1, 2]
    n, plan = resolve_worker_plan(workers=2, gpu_ids=[], require_gpu=False)
    assert n == 2 and plan == [None, None]


def test_parallel_panel_matches_serial(tmp_path):
    cfg = _cfg(tmp_path)
    serial = run_panel(
        cfg,
        policy=build_policy(cfg, device="cpu"),
        bank=load_empty_bank(),
        role="harness_select",
        seed=17,
        max_per_domain=3,
    )
    parallel = run_panel_mp(
        cfg,
        bank_ref=None,
        role="harness_select",
        seed=17,
        max_per_domain=3,
        workers=2,
        gpu_ids=[None, None],  # CPU workers for the CI-sized test
        require_gpu=False,
    )
    assert set(serial.per_task) == set(parallel.per_task), (
        "parallel run must not drop or invent tasks"
    )
    for tid in serial.per_task:
        assert serial.per_task[tid]["value"] == parallel.per_task[tid]["value"], tid
        assert (
            serial.per_task[tid]["terminated_reason"]
            == parallel.per_task[tid]["terminated_reason"]
        )
    assert abs(serial.suite.macro_score - parallel.suite.macro_score) < 1e-12
    assert parallel.costs.n_tasks == serial.costs.n_tasks
