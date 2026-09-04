"""CLI / mask wiring for --harness Harness-G. Default Harness-1 must stay intact."""

from __future__ import annotations

from trim.adapters.components import all_component_ids, zero_mask
from trim.adapters.harness_g_components import COMPONENT_TAXONOMY as G_TAXONOMY
from trim.adapters.harness_g_components import RUNTIME_TOOLS as G_RUNTIME
from trim.adapters.harness_profiles import infer_harness_from_ids, is_harness_g, normalize_harness
from trim.cli.launch import (
    LaunchError,
    canonical_component_ids,
    parse_eval_args,
    parse_train_args,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from trim.training.four_cell_runtime import student_mask_for, teacher_mask_for
from trim.training.tool_mask import legal_tool_names

import pytest


def test_default_all_is_still_harness1():
    ids = canonical_component_ids(["all"])
    assert ids == list(all_component_ids())
    assert "evidence_graph" in ids
    assert "answer_with" not in ids


def test_overlapping_alias_neighbors_stays_harness1():
    assert canonical_component_ids(["neighbors"]) == ["chunk_neighbors"]
    assert infer_harness_from_ids(["neighbors"]) == "Harness-1"
    assert infer_harness_from_ids(["dedup"]) == "Harness-1"


def test_unambiguous_g_id_infers_harness_g():
    assert infer_harness_from_ids(["answer_with"]) == "Harness-G"
    assert infer_harness_from_ids("bridge_entities,snc_frontier") == "Harness-G"
    assert infer_harness_from_ids(["zero"]) == "Harness-1"
    assert infer_harness_from_ids(["all"]) == "Harness-1"


def test_canonical_g_all_requires_harness_flag():
    ids = canonical_component_ids(["all"], harness="Harness-G")
    assert ids == list(G_TAXONOMY)
    assert "answer_with" in ids
    assert "evidence_graph" not in ids


def test_canonical_g_zero_and_single():
    assert canonical_component_ids(["zero"], harness="Harness-G") == []
    assert canonical_component_ids(["answer_with"], harness="Harness-G") == ["answer_with"]
    assert canonical_component_ids(["harvest", "snc"], harness="Harness-G") == [
        "answer_with",
        "snc_frontier",
    ]
    assert canonical_component_ids(["dedup"], harness="Harness-G") == ["lookup_dedup"]
    assert canonical_component_ids(["neighbors"], harness="Harness-G") == ["sentence_neighbors"]


def test_h1_rejects_g_component():
    with pytest.raises(LaunchError, match="unknown"):
        canonical_component_ids(["answer_with"], harness="Harness-1")


def test_parse_train_harness_g_all():
    args, spec = parse_train_args(
        [
            "--harness",
            "Harness-G",
            "--train_method",
            "trim",
            "--component",
            "all",
            "--out",
            "/tmp/trim-hg-all",
        ]
    )
    assert spec.harness == "Harness-G"
    assert args.harness == "Harness-G"
    assert set(spec.components) == set(G_TAXONOMY)
    assert "answer_with" in spec.components
    student = student_mask_for_ids(spec.components, harness=spec.harness)
    teacher = teacher_mask_for_ids(spec.components, harness=spec.harness)
    assert set(student) == set(G_TAXONOMY)
    assert all(v is False for v in student.values())
    assert teacher["answer_with"] is True
    assert teacher["snc_frontier"] is True
    runtime = student_mask_for(args.component, harness=args.harness)
    assert runtime == student
    assert is_harness_g(mask=runtime)
    tools = legal_tool_names(harness_mask=runtime)
    assert set(G_RUNTIME) <= set(tools)
    assert "answer_with" not in tools
    assert "search_corpus" not in tools


def test_parse_train_harness_g_zero():
    args, spec = parse_train_args(
        [
            "--harness",
            "hg",
            "--train_method",
            "trim",
            "--component",
            "zero",
            "--out",
            "/tmp/trim-hg-zero",
        ]
    )
    assert spec.harness == "Harness-G"
    assert spec.components == ()
    mask = student_mask_for("zero", harness="Harness-G")
    assert mask == zero_mask("Harness-G")
    assert all(v is False for v in mask.values())
    assert teacher_mask_for("zero", harness="Harness-G") == mask


def test_parse_train_harness_g_answer_with():
    args, spec = parse_train_args(
        [
            "--harness",
            "Harness-G",
            "--train_method",
            "trim",
            "--component",
            "answer_with",
            "--out",
            "/tmp/trim-hg-aw",
        ]
    )
    assert spec.components == ("answer_with",)
    student = student_mask_for_ids(spec.components, harness=spec.harness)
    teacher = teacher_mask_for_ids(spec.components, harness=spec.harness)
    assert student["answer_with"] is False
    assert teacher["answer_with"] is True
    assert student["bridge_entities"] is True
    assert args.component == "answer_with"


def test_parse_eval_harness_g():
    args, spec = parse_eval_args(
        [
            "--harness",
            "Harness-G",
            "--component",
            "all",
            "--out",
            "/tmp/trim-hg-eval",
        ]
    )
    assert spec.harness == "Harness-G"
    assert args.harness == "Harness-G"
    assert "hybrid_init_retrieve" in spec.components


def test_default_train_cli_stays_harness1():
    args, spec = parse_train_args(
        [
            "--train_method",
            "trim",
            "--component",
            "all",
            "--out",
            "/tmp/trim-h1-default",
        ]
    )
    assert spec.harness == "Harness-1"
    assert normalize_harness(None) == "Harness-1"
    assert "evidence_graph" in spec.components
    assert "answer_with" not in spec.components
