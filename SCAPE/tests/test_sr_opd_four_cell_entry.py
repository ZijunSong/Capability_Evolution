import json
from pathlib import Path

from scape.eval.official_query_pool import load_official_384
from scape.training.four_cell_runtime import TEACHER_REGISTRY, build_manifest, validate_wiring


def test_sentence_compress_teacher_registered():
    assert "sentence_compress" in TEACHER_REGISTRY


def test_component_runners_default_to_vllm_scheme_a(monkeypatch):
    import importlib.util
    import sys

    scripts = Path(__file__).parents[1] / "scripts"
    runners = (
        "run_sentence_compress_sr_opd_four_cell.py",
        "run_adaptive_rerank_sr_opd_four_cell.py",
        "run_auto_populate_sr_opd_four_cell.py",
        "run_token_budget_marker_sr_opd_four_cell.py",
        "run_verify_tool_sr_opd_four_cell.py",
    )
    for filename in runners:
        name = filename[:-3]
        spec = importlib.util.spec_from_file_location(name, scripts / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[name] = module
        spec.loader.exec_module(module)
        monkeypatch.setattr(sys, "argv", [name, "--out", "/tmp/runner-test", "--base-model", "model"])
        args = module.parse_args()
        assert args.rollout_backend == "vllm"
        assert args.gpu_schedule == "scheme_a"
        assert args.on_policy_refresh is True
        assert args.enforce_eager is True


def test_adaptive_rerank_teacher_registered():
    assert "adaptive_rerank_instruction" in TEACHER_REGISTRY


def test_official_384_pool_present():
    rows, meta = load_official_384()
    assert meta["official_384"]
    assert len(rows) == 384
    assert rows[0]["query"] and rows[0]["query"] != rows[0]["query_id"]


def test_validate_wiring_sentence_compress(tmp_path: Path):
    class A:
        component = "sentence_compress"
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
        training_mode = "four_cell"

    report = validate_wiring(A())
    assert report["ok"]
    assert report["eval_is_official_384"]
    assert report["official_test_is_76"]
    assert report["official_test_count"] == 76
    assert report["n_train_queries"] == 8
    assert report["teacher_leak_in_student_prefix"] is False
    assert report["n_projected_steps"] >= 1


def test_manifest_marks_new_loss():
    class A:
        training_mode = "four_cell"
        component = "sentence_compress"
        lambda_opd = 0.1
        group_size = 8
        max_turns = 6
        train_steps = 8
        n_queries = 64
        opd_states_per_trajectory = 3
        seed = 42
        base_model = "x"
        sft_adapter = ""
        smoke = False

    man = build_manifest(A())
    assert man["opd_loss"] == "sr_opd_ce"
    assert man["rl_loss_fn"] == "cispo"
    assert man["legacy_tool_token_kl_hook_used"] is False
    assert man["protocol_complete_rl_opd"] is True
    assert man["backend"] == "vllm_rollout+hf_train"
    assert man["train_backend"] == "hf_debug"
    assert man["gpu_schedule"] == "scheme_a"
    assert man["on_policy_refresh"] is True
    assert man["harmony_encoding"] == "o200k_harmony"
    assert man["stop_token_ids"] == [200012, 200002]
    json.dumps(man)
