from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from easyopd import EasyOPD
from easyopd.hook_dispatch import HookDispatcher
from easyopd.methods.scape_component_opd.teacher_sidecar import SCAPETeacherSidecar


ROOT = Path(__file__).resolve().parents[2]


def test_top_level_method_config_enables_dispatcher():
    dispatcher = HookDispatcher.from_config({"method": {"name": "scape_component_opd"}})
    assert dispatcher.enabled
    assert dispatcher.method_name == "scape_component_opd"
    assert dispatcher.hooks.has_loss


def test_scape_teacher_sidecar_scores_same_student_tokens():
    sidecar = SCAPETeacherSidecar("evidence_graph")
    batch = {
        "scape_snapshot": {
            "query_id": "q",
            "turn_id": 0,
            "documents": [{"id": "d0"}],
            "curated_ids": ["d0"],
            "component_masks": {"evidence_graph": False},
        },
        "student_response_token_ids": [11, 12, 13],
        "response_mask": [1, 1, 1],
    }
    payload = sidecar.teacher_forward(batch, teacher_model=object(), config={})
    assert payload["metadata"]["student_token_count"] == 3
    assert len(payload["teacher_logprobs"]) == 3
    assert payload["teacher_view"]["state_hashes"]["state_hash_student"] == payload["teacher_view"]["state_hashes"]["state_hash_teacher"]


def test_from_hparams_train_dry_run_builds_verl_command(tmp_path):
    inst = EasyOPD.from_hparams("scape_component_opd", auto_resolve_data=False)
    manifest = inst.train(dry_run=True, output_dir=str(tmp_path), extra_args={"component": {"name": "evidence_graph"}})
    assert manifest["dry_run"] is True
    assert manifest["command"][:3] == [sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "+method.name=scape_component_opd" in manifest["command"]
    assert "+component.name=evidence_graph" in manifest["command"]
    assert f"trainer.default_local_dir={tmp_path}" in manifest["command"]


def test_cli_train_dry_run_generates_command(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "scape_component_opd.py"),
            "train",
            "--component",
            "evidence_graph",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["command"][:3] == [sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "+component.name=evidence_graph" in payload["command"]
