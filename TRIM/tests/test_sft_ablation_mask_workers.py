from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trim.adapters.components import zero_mask
from trim.training.sft_ablation.env_worker import launch_env_worker
from trim.upstream_harness1.v8d_flags import all_enabled_mask


def test_extra_env_cannot_override_the_role_mask(tmp_path):
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="V8D_"):
        launch_env_worker(job, mask=zero_mask("Harness-1"), extra_env={"V8D_VERIFY_TOOL": "1"})


def _run(mask: dict, tmp_path: Path, name: str) -> dict:
    job = tmp_path / f"{name}.json"
    result = tmp_path / f"{name}-result.json"
    job.write_text(
        json.dumps({"task": "audit", "mask": mask, "result": str(result)}),
        encoding="utf-8",
    )
    before = os.environ.get("V8D_VERIFY_TOOL")
    launch_env_worker(job, mask=mask)
    assert os.environ.get("V8D_VERIFY_TOOL") == before
    return json.loads(result.read_text(encoding="utf-8"))


def test_full_and_zero_workers_import_the_requested_flags(tmp_path):
    full = _run(all_enabled_mask("Harness-1"), tmp_path, "full")
    zero = _run(zero_mask("Harness-1"), tmp_path, "zero")
    assert full["ok"] is True
    assert zero["ok"] is True
    assert full["ultra_core"]["V8D_VERIFY_TOOL"] is True
    assert full["train_rl"]["V8D_VERIFY_TOOL"] is True
    assert zero["ultra_core"]["V8D_VERIFY_TOOL"] is False
    assert "verify" in full["tool_names"]
    assert "verify" not in zero["tool_names"]
