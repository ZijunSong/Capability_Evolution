"""CLI, contract, projection, and gate checks for the three TRIM ablations."""

from __future__ import annotations

import pytest

from trim.cli.launch import LaunchError, parse_eval_args, parse_train_args, train_method_to_mode
from trim.training.dist_runtime import parse_dist_argv, torchrun_cmd
from trim.training.opd_train_contract import (
    assert_train_contract,
    control_variant_for_method,
    gap_variant_for_method,
    training_cell_for_method,
)
from trim.training.rl_opd_types import (
    OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    TRAINING_MODE_NO_GATE,
    TRAINING_MODE_NO_PROJECT,
    TRAINING_MODE_ONLY_OPD,
    TRAINING_MODE_PURE_OPD,
)
from trim.training.sr_opd_loss import action_gap_gate, gated_action_gap_per_token
from trim.training.tinker_opd_datum import TinkerOPDDatum, resolve_row_ablation


def _parse(method: str, *extra: str):
    argv = ["--train_method", method, "--out", f"/tmp/trim-{method}", *extra]
    return parse_train_args(argv)


def test_three_methods_are_independent_cells():
    expected = {
        "only-opd": TRAINING_MODE_ONLY_OPD,
        "no-project": TRAINING_MODE_NO_PROJECT,
        "no-gate": TRAINING_MODE_NO_GATE,
    }
    cells = set()
    for method, mode in expected.items():
        args, spec = _parse(method)
        assert spec.training_mode == mode
        assert train_method_to_mode(method) == mode
        assert training_cell_for_method(method) == mode
        assert args.component == "all" or spec.component_preset == "all"
        assert spec.component_preset == "all"
        assert args.opd_loss == OPD_LOSS_PROJECTED_GAP
        assert args.lambda_opd == 0.01
        assert args.opd_states_per_trajectory == -1
        assert args.rl_reward_effective == "trim-legacy-v1"
        cells.add(training_cell_for_method(method))
        contract = assert_train_contract(method, None, "hf_debug")
        assert contract["opd_loss"] == OPD_LOSS_PROJECTED_GAP
        assert gap_variant_for_method(method) == "projected"
        assert control_variant_for_method(method) is None
    assert cells == {"only_opd", "no_project", "no_gate"}
    only, _ = _parse("only-opd")
    project, _ = _parse("no-project")
    gate, _ = _parse("no-gate")
    assert only.rl_enabled is False
    assert only.projection_mode == "full"
    assert only.opd_gate_mode == "sigmoid"
    assert project.rl_enabled is True
    assert project.projection_mode == "native_only"
    assert gate.opd_gate_mode == "none"
    assert gate.projection_mode == "full"


def test_omitted_component_defaults_to_all_and_masks():
    args, spec = _parse("no-gate")
    assert spec.component_preset == "all"
    student = args.experiment_contract["student_mask"]
    teacher = args.experiment_contract["teacher_mask"]
    assert student and not any(student.values())
    assert teacher and all(teacher.values())
    assert args.experiment_contract["student_harness"] == "H_zero"
    assert args.experiment_contract["teacher_harness"] == "H_full"


def test_explicit_non_all_component_is_rejected():
    with pytest.raises(LaunchError):
        _parse("only-opd", "--component", "zero")
    with pytest.raises(LaunchError):
        _parse("no-project", "--harness", "Harness-G", "--component", "all")
    kept, spec = parse_train_args(
        ["--train_method", "trim", "--component", "sentence_compress", "--out", "/tmp/trim-keep"]
    )
    assert spec.train_method == "trim"
    assert "sentence_compress" in spec.components


def test_original_opd_defaults_stay_ce():
    args, spec = parse_train_args(
        ["--train_method", "opd", "--component", "sentence_compress", "--out", "/tmp/opd-keep"]
    )
    assert spec.training_mode == TRAINING_MODE_PURE_OPD
    assert args.opd_loss == OPD_LOSS_CE
    assert args.lambda_opd == 0.1
    assert args.opd_states_per_trajectory == 3


def test_new_methods_reject_ce_and_sampled_gap():
    for method in ("only-opd", "no-project", "no-gate"):
        for loss in (OPD_LOSS_CE, OPD_LOSS_SAMPLED_GAP):
            with pytest.raises(SystemExit):
                assert_train_contract(method, loss, "hf_debug")


def test_reward_choices_and_legacy_default():
    args, _ = _parse("trim", "--component", "all")
    assert args.rl_reward is None
    assert args.rl_reward_effective == "trim-legacy-v1"
    fa, _ = _parse("no-project", "--rl-reward", "final-answer-recall")
    assert fa.rl_reward_effective == "final-answer-recall"
    assert fa.experiment_contract["reward_used_for_optimization"] is True
    only, _ = _parse("only-opd", "--rl-reward", "final-answer-recall")
    assert only.experiment_contract["reward_used_for_optimization"] is False
    with pytest.raises(SystemExit):
        _parse("no-gate", "--rl-reward", "not-a-reward")


def test_dist_argv_keeps_method_and_reward():
    cfg, train_argv = parse_dist_argv(
        [
            "--nnodes",
            "1",
            "--nproc-per-node",
            "1",
            "--train_method",
            "no-project",
            "--rl-reward",
            "final-answer-recall",
            "--out",
            "/tmp/dist-ablation",
        ]
    )
    cmd = torchrun_cmd(script="TRIM/scripts/run_train.py", train_argv=train_argv, cfg=cfg, node_rank=0)
    text = " ".join(cmd)
    assert "no-project" in train_argv
    assert "final-answer-recall" in train_argv
    assert "no-project" in text
    assert "final-answer-recall" in text


def test_eval_parser_has_no_training_reward():
    with pytest.raises(SystemExit):
        parse_eval_args(["--component", "zero", "--rl-reward", "final-answer-recall", "--out", "/tmp/eval-r"])


def test_native_only_skips_mapping_and_keeps_later_anchor():
    from easyopd.methods.scape_component_opd.skip_to_anchor import project_events

    events = [
        {
            "event_id": "m1",
            "kind": "harness_mutation",
            "action_name": None,
            "state_delta": {"added": ["d1"], "removed": []},
            "projectable_target": {"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}},
            "payload": {"delta_add": ["d1"]},
            "accessible_doc_ids": ["d1"],
        },
        {
            "event_id": "r1",
            "kind": "model_action",
            "action_name": "read_document",
            "arguments": {"doc_id": "d1"},
            "accessible_doc_ids": ["d1"],
        },
    ]
    full = project_events(events, accessible_doc_ids=["d1"], first_only=True)
    native = project_events(events, accessible_doc_ids=["d1"], first_only=True, allow_explicit_mapping=False)
    assert full[0].actions[0].name == "curate"
    assert full[0].alignment_source == "explicit_mapping"
    assert native[0].actions[0].name == "read_document"
    assert native[0].alignment_source == "native_anchor_after_skip"
    only_map = project_events(events[:1], accessible_doc_ids=["d1"], allow_explicit_mapping=False)
    assert only_map[0].kind == "skip"


def test_native_curate_ignores_inconsistent_delta_and_horizon():
    from easyopd.methods.scape_component_opd.skip_to_anchor import project_events

    native_curate = {
        "event_id": "c1",
        "kind": "model_action",
        "action_name": "curate",
        "arguments": {"add_ids": ["kept"], "remove_ids": []},
        "state_delta": {"added": ["other"], "removed": []},
        "accessible_doc_ids": ["kept", "other"],
    }
    native = project_events([native_curate], accessible_doc_ids=["kept", "other"], allow_explicit_mapping=False)
    full = project_events([native_curate], accessible_doc_ids=["kept", "other"], allow_explicit_mapping=True)
    assert native[0].actions[0].arguments["add_ids"] == ["kept"]
    assert full[0].actions[0].arguments["add_ids"] == ["other"]
    hidden = {
        "event_id": "h",
        "kind": "tool_observation",
        "visible_to_student": False,
        "action_name": "verify",
    }
    late = {
        "event_id": "late",
        "kind": "model_action",
        "action_name": "end_search",
        "arguments": {},
    }
    events = [hidden] * 8 + [late]
    blocked = project_events(events, accessible_doc_ids=[], max_anchor_scan_events=8, allow_explicit_mapping=False)
    assert blocked[0].kind == "skip"
    again = project_events([native_curate], accessible_doc_ids=["kept", "other"], allow_explicit_mapping=True)
    assert again[0].actions[0].arguments["add_ids"] == full[0].actions[0].arguments["add_ids"]


def test_gate_none_is_one_and_beta_zero_is_half():
    torch = pytest.importorskip("torch")
    student = torch.tensor([0.2, -0.4], requires_grad=True)
    teacher = torch.tensor([0.5, 0.1], requires_grad=True)
    none_gap = gated_action_gap_per_token(student, teacher, gate_beta=0.0, gate_mode="none")
    sig_gap = gated_action_gap_per_token(student, teacher, gate_beta=0.0, gate_mode="sigmoid")
    none_gap.sum().backward()
    assert torch.allclose(student.grad, torch.tensor([-1.0, -1.0]))
    assert teacher.grad is None
    delta = (teacher.detach() - student.detach())
    half = action_gap_gate(delta, gate_beta=0.0, gate_mode="sigmoid")
    ones = action_gap_gate(delta, gate_beta=5.0, gate_mode="none")
    assert torch.allclose(half, torch.full_like(half, 0.5))
    assert torch.allclose(ones, torch.ones_like(ones))
    assert not torch.allclose(none_gap.detach(), sig_gap.detach())
    with pytest.raises(ValueError):
        action_gap_gate(delta, gate_mode="off")


def test_gate_mode_survives_datum_roundtrip_and_merge_key():
    datum = TinkerOPDDatum(
        model_input="s",
        prompt_token_ids=[1],
        target_tokens=[0, 2],
        weights=[0.0, 1.0],
        policy_version="v",
        n_supervised_tokens=1,
        teacher_prompt_token_ids=[3],
        opd_loss="sr_opd_projected_gap",
        metadata={
            "gate_mode": "none",
            "projection_mode": "full",
            "ablation_variant": "no_gate",
            "ablation_contract_version": "trim_ablation_v1",
            "lambda_opd": 0.01,
        },
    )
    restored = TinkerOPDDatum.from_dict(datum.to_dict())
    assert restored.metadata["gate_mode"] == "none"
    merged = resolve_row_ablation(restored.metadata, restored.to_dict())
    assert merged["gate_mode"] == "none"
    with pytest.raises(ValueError):
        resolve_row_ablation({"gate_mode": "sigmoid", "ablation_contract_version": "trim_ablation_v1"}, {})
