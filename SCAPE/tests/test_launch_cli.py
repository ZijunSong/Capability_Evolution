from pathlib import Path

import pytest

from scape.adapters.components import all_component_ids
from scape.cli.launch import (
    LaunchError,
    canonical_component_ids,
    coalition_slug,
    discover_adapter_map,
    parse_eval_args,
    parse_train_args,
    student_mask_for_ids,
    teacher_mask_for_ids,
    train_method_to_mode,
)
from scape.training.four_cell_runtime import (
    component_ids_of,
    student_mask_for,
    teacher_for,
    validate_wiring,
)
from scape.training.rl_opd_types import (
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
)


def test_parse_component_list_space_and_comma():
    assert canonical_component_ids(["sentence_compress", "verify_tool"]) == [
        "sentence_compress",
        "verify_tool",
    ]
    assert canonical_component_ids(["sentence_compress,verify_tool"]) == [
        "sentence_compress",
        "verify_tool",
    ]
    assert canonical_component_ids(["AUTO", "GRAPH"]) == [
        "auto_populate_first_search",
        "evidence_graph",
    ]
    assert canonical_component_ids(["all"]) == list(all_component_ids())
    assert canonical_component_ids(["zero"]) == []
    assert canonical_component_ids(["ZERO"]) == []


def test_unknown_component_rejected():
    with pytest.raises(LaunchError):
        canonical_component_ids(["not_a_component"])
    with pytest.raises(LaunchError):
        canonical_component_ids(["zero", "sentence_compress"])


def test_train_method_mapping():
    assert train_method_to_mode("opd") == TRAINING_MODE_PURE_OPD
    assert train_method_to_mode("rl+opd") == TRAINING_MODE_RL_OPD
    assert train_method_to_mode("rl") == TRAINING_MODE_RL
    assert train_method_to_mode("rl_opd") == TRAINING_MODE_RL_OPD


def test_parse_train_cli():
    args, spec = parse_train_args(
        [
            "--harness",
            "Harness-1",
            "--benchmark",
            "BC+",
            "--model_name",
            "harness-1",
            "--train_method",
            "rl+opd",
            "--component",
            "sentence_compress,verify_tool",
            "--validate-only",
            "--out",
            "/tmp/scape-train-test",
        ]
    )
    assert spec.harness == "Harness-1"
    assert spec.benchmark == "BC+"
    assert spec.model_name == "harness-1"
    assert spec.train_method == "rl+opd"
    assert spec.training_mode == TRAINING_MODE_RL_OPD
    assert spec.components == ("sentence_compress", "verify_tool")
    assert args.component == "sentence_compress,verify_tool"
    assert args.n_queries == 664
    assert args.validate_only is True


def test_parse_eval_cli_space_separated_components():
    args, spec = parse_eval_args(
        [
            "--component",
            "sentence_compress",
            "token_budget_marker",
            "--out",
            "/tmp/scape-eval-test",
        ]
    )
    assert spec.components == ("sentence_compress", "token_budget_marker")
    assert args.component == "sentence_compress,token_budget_marker"
    assert spec.harness == "Harness-1"
    assert spec.benchmark == "BC+"


def test_parse_component_zero():
    args, spec = parse_train_args(
        [
            "--train_method",
            "rl",
            "--component",
            "zero",
            "--out",
            "/tmp/scape-train-zero",
        ]
    )
    assert spec.zero_components is True
    assert spec.components == ()
    assert spec.coalition == "zero"
    assert args.component == "zero"
    teacher = teacher_mask_for_ids(spec.components)
    student = student_mask_for_ids(spec.components)
    assert teacher == student
    assert all(enabled is False for enabled in teacher.values())
    assert set(teacher) == set(all_component_ids())


def test_masks_enable_listed_on_teacher_off_on_student():
    ids = ["adaptive_rerank_instruction", "sentence_compress"]
    teacher = teacher_mask_for_ids(ids)
    student = student_mask_for_ids(ids)
    assert teacher["adaptive_rerank_instruction"] is True
    assert teacher["sentence_compress"] is True
    assert student["adaptive_rerank_instruction"] is False
    assert student["sentence_compress"] is False
    assert teacher["verify_tool"] == student["verify_tool"]


def test_four_cell_coalition_mask_and_teacher():
    assert component_ids_of("sentence_compress,evidence_graph") == [
        "sentence_compress",
        "evidence_graph",
    ]
    mask = student_mask_for("sentence_compress,evidence_graph")
    assert mask["sentence_compress"] is False
    assert mask["evidence_graph"] is False
    assert teacher_for("sentence_compress,evidence_graph") is not None
    assert teacher_for("content_dedup") is not None
    assert component_ids_of("zero") == []
    zero_mask = student_mask_for("zero")
    assert all(enabled is False for enabled in zero_mask.values())
    assert teacher_for("zero") is not None


def test_validate_wiring_coalition():
    class A:
        component = "sentence_compress,verify_tool"
        n_queries = 8
        lambda_opd = 0.1
        group_size = 8
        max_turns = 6
        train_steps = 8
        opd_states_per_trajectory = 3
        seed = 42
        base_model = "/unused"
        sft_adapter = ""
        smoke = False
        query_manifest = None
        eval_manifest = None
        training_mode = "rl_opd"

    report = validate_wiring(A())
    assert report["ok"]
    assert report["component_ids"] == ["sentence_compress", "verify_tool"]
    assert report["official_test_is_166"]
    assert report["n_projected_steps"] >= 1
    assert report["teacher_leak_in_student_prefix"] is False


def test_validate_wiring_zero_components():
    class A:
        component = "zero"
        n_queries = 8
        lambda_opd = 0.1
        group_size = 8
        max_turns = 6
        train_steps = 8
        opd_states_per_trajectory = 3
        seed = 42
        base_model = "/unused"
        sft_adapter = ""
        smoke = False
        query_manifest = None
        eval_manifest = None
        training_mode = "rl"

    report = validate_wiring(A())
    assert report["ok"]
    assert report["component"] == "zero"
    assert report["component_ids"] == []
    assert report["n_projected_steps"] >= 1


def test_discover_adapter_map(tmp_path: Path):
    adapters = tmp_path / "seed42" / "adapters" / "rl_opd"
    adapters.mkdir(parents=True)
    (adapters / "adapter_config.json").write_text("{}", encoding="utf-8")
    found = discover_adapter_map(tmp_path)
    assert found["rl_opd"].endswith("seed42/adapters/rl_opd")
    assert coalition_slug(["sentence_compress", "verify_tool"]) == "sentence_compress+verify_tool"
    assert coalition_slug([]) == "zero"
