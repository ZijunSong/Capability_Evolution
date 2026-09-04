from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from trim.cli.launch import parse_sft_args
from trim.training.sft_data import (
    EXPECTED_N_TRAJECTORIES,
    HF_SFT_REPO,
    assert_train_sft_ready,
    load_sft_trajectories,
    materialize_sft_data_dir,
    pack_sft_tar,
    trajectory_filename,
    unwrap_sft_record,
)
from trim.training.sft_runtime import (
    HARNESS1_SFT_BATCH_SIZE,
    HARNESS1_SFT_EVAL_EVERY,
    HARNESS1_SFT_LEARNING_RATE,
    HARNESS1_SFT_LORA_RANK,
    HARNESS1_SFT_MAX_LENGTH,
    HARNESS1_SFT_MIN_RECALL,
    HARNESS1_SFT_MODEL_NAME,
    HARNESS1_SFT_NUM_EPOCHS,
    HARNESS1_SFT_SAVE_EVERY,
    HARNESS1_SFT_V8D_ENV,
    HARNESS1_TRAIN_SFT,
    canonical_sft_model_name,
    train_sft_argv,
)


def _traj(qid: str = "q1", dataset: str = "sec", recall: float = 0.5) -> dict:
    return {
        "format_version": "ultra_v3",
        "query_id": qid,
        "query_text": "How many days?",
        "dataset_name": dataset,
        "normalize_ids": False,
        "num_turns": 1,
        "final_recall": recall,
        "turn_history": [
            {
                "turn_idx": 1,
                "tool_name": "search_corpus",
                "params": {"query": "days"},
                "reasoning": "search",
                "observation": "doc_1: 45 days",
            }
        ],
        "doc_store": {"doc_1": {"snippet": "45 days"}},
        "stage": "sft",
    }


def test_unwrap_payload_json_and_skip_rl():
    traj = _traj()
    rec = {
        "query_id": "q1",
        "query": "How many days?",
        "dataset_name": "sec",
        "stage": "sft",
        "payload_json": json.dumps(traj),
    }
    out = unwrap_sft_record(rec)
    assert out is not None
    assert out["query_text"] == "How many days?"
    assert out["turn_history"][0]["tool_name"] == "search_corpus"
    assert out["stage"] == "sft"
    assert unwrap_sft_record({"stage": "rl", "payload_json": json.dumps(traj)}) is None


def test_pack_and_load_tar(tmp_path: Path):
    trajs = [_traj("a", "sec"), _traj("b", "web")]
    archive = pack_sft_tar(tmp_path / "harness-1-sft-data.tar.gz", trajs)
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
    assert "harness-1-sft-data/MANIFEST.json" in names
    assert any(n.endswith("ultra_v3_sec_a.json") for n in names)
    loaded, meta = load_sft_trajectories(archive, allow_hf=False)
    assert [t["query_id"] for t in loaded] == ["a", "b"]
    assert meta["n_trajectories"] == 2
    assert meta["n_rl_skipped"] == 0
    assert meta["datasets"] == {"sec": 1, "web": 1}


def test_materialize_json_dir_for_train_sft(tmp_path: Path):
    jsonl = tmp_path / "sft_trajectories.jsonl"
    rows = [
        {"stage": "sft", "payload_json": json.dumps(_traj("100", "browsecompplus"))},
        {"stage": "rl", "query_id": "rl1", "payload_json": "{}"},
    ]
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    dest = tmp_path / "extracted"
    data_dir, meta = materialize_sft_data_dir(
        jsonl, dest=dest, write_pack=tmp_path / "pack.tar.gz", allow_hf=False
    )
    assert meta["n_trajectories"] == 1
    assert meta["n_rl_skipped"] == 1
    assert data_dir.name == "trajectories"
    n = assert_train_sft_ready(data_dir)
    assert n == 1
    name = trajectory_filename(_traj("100", "browsecompplus"))
    assert (data_dir / name).is_file()
    sample = json.loads((data_dir / name).read_text(encoding="utf-8"))
    assert sample["query_text"]
    assert sample["turn_history"]


def test_parse_sft_defaults_match_harness1():
    args = parse_sft_args(["--out", "/tmp/trim-sft"])
    assert args.model_name == HARNESS1_SFT_MODEL_NAME
    assert args.num_epochs == HARNESS1_SFT_NUM_EPOCHS == 3
    assert args.batch_size == HARNESS1_SFT_BATCH_SIZE == 128
    assert args.learning_rate == HARNESS1_SFT_LEARNING_RATE == 5e-6
    assert args.lora_rank == HARNESS1_SFT_LORA_RANK == 32
    assert args.max_length == HARNESS1_SFT_MAX_LENGTH == 32768
    assert args.min_recall == HARNESS1_SFT_MIN_RECALL == 0.1
    assert args.save_every == HARNESS1_SFT_SAVE_EVERY == 50
    assert args.eval_every == HARNESS1_SFT_EVAL_EVERY == 50
    assert canonical_sft_model_name("gpt-oss-20b") == "openai/gpt-oss-20b"
    smoke = parse_sft_args(["--smoke", "--out", "/tmp/trim-sft-smoke"])
    assert smoke.num_epochs == 1
    assert smoke.batch_size == 4
    assert smoke.out.name == "trim-sft-smoke"


def test_train_sft_argv_points_at_harness1():
    argv = train_sft_argv(data_dir="/data/sft", log_path="/tmp/sft", python="python3")
    assert str(HARNESS1_TRAIN_SFT) in argv
    assert HARNESS1_TRAIN_SFT.is_file()
    assert argv[argv.index("--model-name") + 1] == "openai/gpt-oss-20b"
    assert argv[argv.index("--num-epochs") + 1] == "3"
    assert argv[argv.index("--batch-size") + 1] == "128"
    assert argv[argv.index("--lora-rank") + 1] == "32"
    assert argv[argv.index("--min-recall") + 1] == "0.1"
    assert argv[argv.index("--save-every") + 1] == "50"
    assert HARNESS1_SFT_V8D_ENV["V8D_VERIFY_TOOL"] == "1"
    assert HARNESS1_SFT_V8D_ENV["V8D_EVIDENCE_GRAPH"] == "1"
    assert "V8D_CHUNK_NEIGHBORS" not in HARNESS1_SFT_V8D_ENV
    assert EXPECTED_N_TRAJECTORIES == 899
    assert HF_SFT_REPO == "pat-jj/harness-1-train-data"


def test_run_sft_dry_run(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sft.py"
    spec = importlib.util.spec_from_file_location("trim_run_sft", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    (traj_dir / "ultra_v3_sec_q1.json").write_text(json.dumps(_traj()) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--sft-data",
            str(traj_dir),
            "--out",
            str(out),
            "--dry-run",
            "--n-trajectories",
            "1",
        ]
    )
    assert rc == 0
    launch = json.loads((out / "LAUNCH.json").read_text(encoding="utf-8"))
    assert launch["model_name"] == "openai/gpt-oss-20b"
    assert launch["framework"].startswith("tinker")
    assert launch["n_trajectory_json"] == 1
    assert launch["dry_run"] is True
    assert str(HARNESS1_TRAIN_SFT) == launch["entrypoint"]
