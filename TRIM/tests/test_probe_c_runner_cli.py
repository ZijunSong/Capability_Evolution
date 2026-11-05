"""Runner CLI guardrails: refuse bad contracts before expensive stages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.probes.probe_c_contract import UNRESOLVED_REVISION
from trim.probes.probe_c_runner import main


def _cfg(tmp_path: Path, **over) -> Path:
    base = {
        "schema_version": "probe_c_v3",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": UNRESOLVED_REVISION,
        "pair_mode": "add_from_h0",
        "components": ["sentence_compress"],
        "state_source": "real_h0_rollout",
        "projection_mode": "direct_only",
        "probe_objective": "frozen_projected_gap",
        "loss_scale": 1.0,
        "splits": {"frozen": True},
        "git_sha": "deadbeef",
        "output_root": str(tmp_path / "out"),
    }
    base.update(over)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_audit_refuses_unresolved_revision(tmp_path: Path):
    p = _cfg(tmp_path)
    rc = main(["audit", "--config", str(p)])
    assert rc == 2


def test_audit_refuses_unfrozen_split(tmp_path: Path):
    p = _cfg(tmp_path, model_revision="rev-1", splits={"frozen": False})
    rc = main(["audit", "--config", str(p)])
    assert rc == 2


def test_expensive_stage_refuses_without_collect_hook(tmp_path: Path):
    p = _cfg(tmp_path, model_revision="rev-1", hooks={})
    rc = main(["collect", "--config", str(p)])
    assert rc == 2  # missing hook is a configuration error, not a fake run


def test_screen_refuses_before_cache(tmp_path: Path):
    p = _cfg(tmp_path, model_revision="rev-1")
    rc = main(["screen", "--config", str(p)])
    assert rc == 2  # missing student hooks / states — never a silent empty run


def test_audit_passes_with_loader_verification(tmp_path: Path, monkeypatch):
    mod = tmp_path / "hookmod.py"
    mod.write_text(
        "def build_hooks():\n"
        "    from trim.probes.probe_c_runner import RuntimeHooks\n"
        "    return RuntimeHooks(verify_model_loader=lambda cfg: {'supported': True})\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    p = _cfg(
        tmp_path,
        model_revision="rev-1",
        model_path="/models/qwen35-9b",
    )
    rc = main(["audit", "--config", str(p), "--hooks-module", "hookmod"])
    assert rc == 0
    out = next((tmp_path / "out").rglob("MANIFEST.json"))
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["model_revision"] == "rev-1"
    assert manifest["pairs"]["sentence_compress"]["diff_components"] == [
        "sentence_compress"
    ]
