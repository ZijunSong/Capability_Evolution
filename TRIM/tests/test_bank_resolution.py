from pathlib import Path

from trim.adapters.components import all_component_ids, zero_mask
from trim.adapters.role_masks import (
    resolve_harness_pair,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from trim.cli.launch import LaunchSpec
from trim.harness_bank.runtime import EpisodeBankRuntime
from trim.harness_bank.loader import load_bank
from tests.rsi_support import write_bank


def _spec(**kwargs) -> LaunchSpec:
    base = dict(
        harness="Harness-1",
        benchmark="BC+",
        model_name="toy",
        components=tuple(all_component_ids("Harness-1")),
        component_preset="all",
    )
    base.update(kwargs)
    return LaunchSpec(**base)


def test_legacy_masks_unchanged_without_bank():
    spec = _spec()
    pair = resolve_harness_pair(spec)
    assert pair["mode"] == "legacy"
    assert pair["student_mask"] == student_mask_for_ids(spec.components, harness="Harness-1")
    assert pair["teacher_mask"] == teacher_mask_for_ids(
        spec.components, harness="Harness-1", preset="all"
    )


def test_bank_zero_runs_no_hooks(tmp_path: Path):
    bank_dir = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    loaded = load_bank(bank_dir)
    spec = _spec(components=(), component_preset="zero", harness_bank=str(bank_dir))
    pair = resolve_harness_pair(spec, bank=loaded)
    assert pair["hooks_on_student"] is False
    assert pair["hooks_on_teacher"] is False
    runtime = EpisodeBankRuntime(loaded, role="zero")
    rendered = runtime.render({"query": "q"}, role="zero")
    assert "teacher_guidance" not in rendered
    assert runtime.components == []


def test_bank_all_does_not_enable_static_components(tmp_path: Path):
    bank_dir = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    loaded = load_bank(bank_dir)
    spec = _spec(components=(), component_preset="all", harness_bank=str(bank_dir))
    pair = resolve_harness_pair(spec, bank=loaded)
    assert pair["teacher_mask"] == zero_mask("Harness-1")
    assert pair["student_mask"] == zero_mask("Harness-1")
    assert pair["active_components"] == ["evo.query_repair"]
    assert pair["hooks_on_teacher"] is True
    runtime = EpisodeBankRuntime(loaded, role="teacher")
    rendered = runtime.render({"query": "q"}, role="teacher")
    assert rendered["teacher_guidance"]
