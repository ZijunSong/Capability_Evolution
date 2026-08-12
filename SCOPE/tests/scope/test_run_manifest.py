"""Run manifest tests."""

from __future__ import annotations

from experiments.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from experiments.common.spec import ExperimentSpec
from experiments.common.validation import validate_against_schema


def test_manifest_schema(tmp_path):
    spec = ExperimentSpec(
        experiment_id="m1",
        group="g",
        method="m",
        capability="duplicate_evidence",
        variant="v",
        changed_factor="none",
        base_model="Qwen2.5-7B-Instruct",
        dataset="browsecomp_plus",
        runtime_config="harness/configs/modules_minimal_v2.yaml",
        seed=42,
        output_dir=str(tmp_path / "out"),
    )
    man = build_run_manifest(spec, command=["echo", "hi"])
    man = finalize_run_manifest(man, exit_code=0, output_dir=tmp_path / "out")
    write_run_manifest(tmp_path / "out" / "run_manifest.json", man)
    errs = validate_against_schema(man, "run_manifest.schema.json")
    assert errs == []
    assert man["status"] == "completed"
    assert man["git"]["head"]
