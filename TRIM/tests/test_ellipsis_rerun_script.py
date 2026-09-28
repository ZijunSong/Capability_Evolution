from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rerun_gptoss_all_ellipsis_queries.sh"


def test_script_bash_syntax():
    subprocess.check_call(["bash", "-n", str(SCRIPT)])


def test_dry_run_emits_eval_and_merge_for_both_benchmarks():
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_ID": "test_ellipsis_rerun",
        "PY": "python3",
    }
    out = subprocess.check_output(["bash", str(SCRIPT)], env=env, text=True)
    starts = [line for line in out.splitlines() if line.startswith("JOB start ")]
    assert len(starts) == 2
    assert any("benchmark=longsealqa" in line and "longseal-004" in line for line in starts)
    assert any("benchmark=hotpotqa" in line and "5a75f0ea5542994ccc91866c" in line for line in starts)
    dry = [line for line in out.splitlines() if line.startswith("DRY_RUN ")]
    assert any("--component all" in line and "--query-ids longseal-004" in line for line in dry)
    assert any("merge_upstream_api_rerun.py" in line and "hotpotqa" in line for line in dry)
    assert any("--verify-base-url http://127.0.0.1:8050/v1" in line for line in dry)
