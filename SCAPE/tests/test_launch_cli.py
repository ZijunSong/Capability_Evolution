import json
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
    TRAINING_MODE_SCAPE_RL,
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
    assert train_method_to_mode("scape+rl") == TRAINING_MODE_SCAPE_RL
    assert train_method_to_mode("scape_rl") == TRAINING_MODE_SCAPE_RL


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
    assert args.opd_states_per_trajectory == 3
    assert args.opd_loss == "sr_opd_ce"
    assert args.lambda_opd == 0.1


def test_parse_scape_rl_defaults_all_actions_and_sampled_gap():
    args, spec = parse_train_args(
        [
            "--train_method",
            "scape+rl",
            "--component",
            "sentence_compress",
            "--out",
            "/tmp/scape-rl-test",
        ]
    )
    assert spec.train_method == "scape+rl"
    assert spec.training_mode == TRAINING_MODE_SCAPE_RL
    assert args.opd_states_per_trajectory == -1
    assert args.opd_loss == "sr_opd_sampled_gap"
    assert args.lambda_opd == 0.01
    assert args.opd_gate_beta == 5.0
    assert args.n_queries is None
    assert args.score_split == "bcplus_830"
    assert "harness-1-rl-data" in str(args.rl_data)
    assert "harness-1-sec-corpus" in str(args.sec_corpus_root)


def test_parse_scape_rl_can_override_k():
    args, _spec = parse_train_args(
        [
            "--train_method",
            "scape+rl",
            "--component",
            "zero",
            "--opd-states-per-trajectory",
            "5",
            "--out",
            "/tmp/scape-rl-k",
        ]
    )
    assert args.opd_states_per_trajectory == 5
    assert args.opd_loss == "sr_opd_sampled_gap"


def test_parse_scape_rl_can_override_lambda():
    args, _spec = parse_train_args(
        [
            "--train_method",
            "scape+rl",
            "--component",
            "zero",
            "--lambda-opd",
            "0.2",
            "--out",
            "/tmp/scape-rl-lam",
        ]
    )
    assert args.lambda_opd == 0.2
    assert args.opd_loss == "sr_opd_sampled_gap"


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
    assert args.max_turns == 40
    assert args.max_new_tokens == 2048
    assert args.temperature == 1.0
    assert args.search_k == 10
    assert args.max_model_len == 32768
    assert args.score_split == "bcplus_830"


def test_parse_eval_cli_score_split_830():
    args, _spec = parse_eval_args(
        [
            "--component",
            "zero",
            "--score-split",
            "bcplus_830",
            "--out",
            "/tmp/scape-eval-830",
        ]
    )
    assert args.score_split == "bcplus_830"


def test_parse_eval_cli_can_select_166():
    args, _spec = parse_eval_args(
        [
            "--component",
            "zero",
            "--score-split",
            "bcplus_test_166",
            "--out",
            "/tmp/scape-eval-166",
        ]
    )
    assert args.score_split == "bcplus_test_166"


def test_run_eval_detect_score_split_defaults_to_830():
    import argparse
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_entry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["run_eval_entry"] = module
    spec.loader.exec_module(module)
    assert module.detect_score_split(argparse.Namespace()) == "bcplus_830"
    assert module.detect_score_split(argparse.Namespace(score_split=None, run_dir=None)) == "bcplus_830"
    assert module.detect_score_split(argparse.Namespace(score_split="bcplus_test_166")) == "bcplus_test_166"


def test_parse_eval_cli_score_split_830():
    args, _spec = parse_eval_args(
        [
            "--component",
            "zero",
            "--score-split",
            "bcplus_830",
            "--out",
            "/tmp/scape-eval-830",
        ]
    )
    assert args.score_split == "bcplus_830"


def test_parse_eval_cli_can_override_horizon():
    args, _spec = parse_eval_args(
        [
            "--component",
            "zero",
            "--max-turns",
            "12",
            "--max-new-tokens",
            "512",
            "--temperature",
            "0",
            "--out",
            "/tmp/scape-eval-override",
        ]
    )
    assert args.max_turns == 12
    assert args.max_new_tokens == 512
    assert args.temperature == 0.0


def test_train_horizon_stays_short():
    args, _spec = parse_train_args(
        [
            "--train_method",
            "rl",
            "--component",
            "zero",
            "--out",
            "/tmp/scape-train-horizon",
        ]
    )
    assert args.max_turns == 6
    assert args.max_new_tokens == 384
    assert args.eval_max_turns == 40
    assert args.eval_max_new_tokens == 2048
    assert args.eval_temperature == 1.0


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


def test_run_train_entry_is_train_only(monkeypatch, tmp_path: Path):
    import importlib.util
    import sys

    captured: dict = {}

    def fake_run(args):
        captured["train_only"] = bool(args.train_only)
        captured["official_eval"] = bool(args.official_eval)
        captured["training_mode"] = args.training_mode
        from scape.training.four_cell_runtime import cells_for_mode

        captured["cells"] = cells_for_mode(args.training_mode, train_only=args.train_only)
        return {"ok": True, "train_only": True}

    monkeypatch.setattr("scape.training.four_cell_runtime.run_from_rl_opd_args", fake_run)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_train.py"
    spec = importlib.util.spec_from_file_location("run_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["run_train_entry"] = module
    spec.loader.exec_module(module)
    rc = module.main(
        [
            "--harness",
            "Harness-1",
            "--benchmark",
            "BC+",
            "--model_name",
            "harness-1",
            "--train_method",
            "rl",
            "--component",
            "all",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured["train_only"] is True
    assert captured["official_eval"] is False
    assert captured["training_mode"] == TRAINING_MODE_RL
    assert captured["cells"] == ("rl",)
    launch = json.loads((tmp_path / "LAUNCH.json").read_text(encoding="utf-8"))
    assert launch["train_only"] is True
    assert launch["official_eval"] is False
