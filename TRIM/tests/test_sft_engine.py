"""CPU checks for the v2 SFT engine: plans, sealed rows, launch topology, rollout ACK."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from argparse import Namespace
from pathlib import Path

from trim.training.sft_ablation.cli import parse_sft_args
from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow, fingerprint
from trim.training.sft_ablation.dist_launch import launch_sft
from trim.training.sft_ablation.fixed_train import select_batch_indices, take_batch
from trim.training.sft_ablation.io_util import sha256_tree
from trim.training.sft_engine.action_loss import supervised_positions
from trim.training.sft_engine.contract import build_v2_contract, require_exact_resume
from trim.training.sft_engine.data import load_sealed, seal_dataset, split_query_ids
from trim.training.sft_engine.planner import assert_slot_plan, plan_slots
from trim.training.student_onpolicy_sft.rollout_service import (
    RolloutPool,
    assert_endpoints_disjoint,
    serve,
)


class _Row:
    def __init__(self, n: int) -> None:
        self.prompt_ids = [1] * n
        self.response_ids = [2]


def _sft_row(experiment: str, row_id: str, query: str, *, weight: float = 1.0, token: int = 3) -> SFTBatchRow:
    profile = PROFILES[experiment]
    return SFTBatchRow(
        row_id=row_id,
        experiment=experiment,
        query_id=query,
        episode_id=query,
        decision_id=row_id,
        prompt_ids=[1, 2],
        response_ids=[token],
        response_mask=[True],
        weight=weight,
        policy_version="p",
        prompt_owner=profile.prompt_owner,
        target_source=profile.supervision_provider,
    )


def test_plan_covers_uneven_batches_without_duplicating_rows() -> None:
    assert plan_slots([], 32) == [[] for _ in range(32)]
    for n_rows in (1, 31, 32, 33, 128, 255):
        rows = [_Row(i + 1) for i in range(n_rows)]
        ranks = plan_slots(rows, 32)
        assert_slot_plan(ranks, n_rows, 32)
        assert len({len(rank) for rank in ranks}) == 1


def test_supervised_positions_follow_the_causal_shift() -> None:
    assert supervised_positions(4, [True, False, True]) == [3, 5]


def test_take_batch_matches_cached_indices() -> None:
    rows = [_sft_row("teacher_sft", "b", "q"), _sft_row("teacher_sft", "a", "q"), _sft_row("teacher_sft", "c", "q")]
    first, cursor, _uses = take_batch(rows, cursor=0, batch_size=5, seed=3)
    logical, cursor2, _index_uses = select_batch_indices(3, cursor=0, batch_size=5, seed=3)
    ordered = sorted(rows, key=lambda row: row.row_id)
    assert [row.row_id for row in first] == [ordered[index].row_id for index in logical]
    assert cursor == cursor2 == 5
    second, cursor3, _uses = take_batch(rows, cursor=cursor, batch_size=2, seed=3)
    logical2, cursor4, _index_uses = select_batch_indices(3, cursor=5, batch_size=2, seed=3)
    assert [row.row_id for row in second] == [ordered[index].row_id for index in logical2]
    assert cursor3 == cursor4


def test_query_split_keeps_a_query_on_one_side() -> None:
    train, val = split_query_ids(["q1", "q1", "q2", "q3", "q4"], fraction=0.5, seed=1)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == {"q1", "q2", "q3", "q4"}


def test_seal_refuses_the_wrong_experiment_and_duplicate_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "CONTRACT.json"
        source.write_text(json.dumps({"experiment": "teacher_sft"}) + "\n", encoding="utf-8")
        rows = [
            _sft_row("teacher_sft", "b", "q2"),
            _sft_row("teacher_sft", "a", "q1"),
            _sft_row("teacher_sft", "c", "q1"),
        ]
        out = root / "sealed"
        manifest = seal_dataset(
            rows,
            out_dir=out,
            experiment="teacher_sft",
            source_contract_path=source,
            tokenizer_digest="tok",
            vocab_size=32,
            max_length=32,
            val_fraction=0.5,
            seed=1,
        )
        assert manifest["n_rows"] == 3
        assert (out / "READY.json").is_file()
        loaded = load_sealed(manifest_path=out / "DATASET.json", expected_experiment="teacher_sft")
        assert [loaded.get(index).row_id for index in range(3)] == ["a", "b", "c"]
        try:
            load_sealed(manifest_path=out / "DATASET.json", expected_experiment="projected_teacher_sft")
        except ValueError as exc:
            assert "refusing" in str(exc)
        else:
            raise AssertionError("projected load should have been refused")
        bad = root / "dup"
        try:
            seal_dataset(
                [_sft_row("teacher_sft", "a", "q"), _sft_row("teacher_sft", "a", "q")],
                out_dir=bad,
                experiment="teacher_sft",
                source_contract_path=source,
                tokenizer_digest="tok",
                vocab_size=32,
                max_length=32,
            )
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate row_id should have been refused")
        assert (bad / "DUPLICATES.json").is_file()
        try:
            seal_dataset(
                [_sft_row("projected_teacher_sft", "a", "q", weight=0.5)],
                out_dir=root / "cross",
                experiment="teacher_sft",
                source_contract_path=source,
                tokenizer_digest="tok",
                vocab_size=32,
                max_length=32,
            )
        except ValueError as exc:
            assert "belongs" in str(exc)
        else:
            raise AssertionError("cross-experiment row should have been refused")


def test_batch_alias_and_v2_launch_topology() -> None:
    try:
        parse_sft_args(
            "teacher_sft",
            ["--model_name", "m", "--out", "/tmp/sft-out", "--batch-size", "64", "--global-batch-size", "128"],
        )
    except SystemExit as exc:
        assert "disagree" in str(exc)
    else:
        raise AssertionError("disagreeing batch sizes should fail")
    args = parse_sft_args(
        "teacher_sft",
        [
            "--model_name",
            "m",
            "--out",
            "/tmp/sft-out",
            "--ce-backend",
            "accelerate_ddp",
            "--stage",
            "train",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "64",
        ],
    )
    assert args.batch_size == 64
    assert args.global_batch_size == 64
    assert args.queries_per_round == 64

    def worker(parsed, dist):
        raise AssertionError(f"dry-run started a worker {parsed} {dist}")

    code = launch_sft(
        script=Path(__file__),
        experiment="teacher_sft",
        worker=worker,
        argv=["--model_name", "m", "--out", "/tmp/teacher-sft", "--dry-run"],
    )
    assert code == 0
    try:
        launch_sft(
            script=Path(__file__),
            experiment="teacher_sft",
            worker=worker,
            argv=[
                "--nnodes",
                "1",
                "--nproc-per-node",
                "2",
                "--model_name",
                "m",
                "--out",
                "/tmp/x",
                "--dry-run",
            ],
        )
    except SystemExit as exc:
        assert "nproc-per-node" in str(exc)
    else:
        raise AssertionError("legacy nproc 2 should still be refused")
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = launch_sft(
            script=Path(__file__),
            experiment="teacher_sft",
            worker=worker,
            argv=[
                "--ce-backend",
                "accelerate_ddp",
                "--stage",
                "train",
                "--micro-batch-size",
                "1",
                "--nnodes",
                "4",
                "--nproc-per-node",
                "8",
                "--model_name",
                "m",
                "--out",
                "/tmp/teacher-v2",
                "--dry-run",
            ],
        )
    assert code == 0
    payload = json.loads(buffer.getvalue())
    assert payload["ce_backend"] == "accelerate_ddp"
    assert payload["gradient_ranks"] == 32
    assert payload["nproc_per_node"] == 8
    assert payload["pin_local_cuda_device"] is False
    assert payload["process_group"] == "accelerate"
    assert payload["rl_enabled"] is False


def test_exact_resume_refuses_a_different_world() -> None:
    args = Namespace(
        experiment="teacher_sft",
        global_batch_size=128,
        micro_batch_size=1,
        learning_rate=5e-6,
        weight_decay=0.0,
        grad_clip=1.0,
        optimizer="adamw",
        scheduler="constant",
        seed=1,
        max_model_len=32,
        max_new_tokens=8,
        logits_mode="action_positions",
        gradient_checkpointing="off",
        model_precision="bf16",
        model_name="base",
        init_adapter="init",
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target="q_proj",
        attn_implementation="",
    )
    left = build_v2_contract(
        args,
        dataset_digest="d",
        source_contract_digest="s",
        tokenizer_digest="t",
        gradient_ranks=8,
        nnodes=1,
        nproc_per_node=8,
        init_adapter_digest="a",
    )
    right = build_v2_contract(
        args,
        dataset_digest="d",
        source_contract_digest="s",
        tokenizer_digest="t",
        gradient_ranks=32,
        nnodes=4,
        nproc_per_node=8,
        init_adapter_digest="a",
    )
    require_exact_resume(left, dict(left))
    try:
        require_exact_resume(left, right)
    except ValueError as exc:
        assert "gradient_ranks" in str(exc)
    else:
        raise AssertionError("world-size change should refuse exact resume")


def test_endpoints_must_not_share_ce_gpus() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "endpoints.json"
        path.write_text(
            json.dumps(
                {
                    "workers": [
                        {
                            "worker_id": "w0",
                            "host": "127.0.0.1",
                            "port": 1,
                            "cuda_visible_devices": "4,5",
                            "hostname": "box",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert_endpoints_disjoint(path, train_visible="0,1", hostname="box")
        try:
            assert_endpoints_disjoint(path, train_visible="4,6", hostname="box")
        except SystemExit as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("overlapping GPUs should be refused")


class _FakeSession:
    def __init__(self) -> None:
        self.loaded = None
        self.inflight = 0

    def load_policy(self, version: str, adapter_path: str, adapter_digest: str) -> dict[str, str]:
        del adapter_path
        self.loaded = (version, adapter_digest)
        return {"loaded_version": version, "loaded_digest": adapter_digest, "probe_result": "fake"}

    def collect(self, request: dict) -> dict:
        if self.loaded is None or self.loaded[1] != request.get("expected_digest") or self.loaded[0] != request.get("expected_version"):
            raise RuntimeError("old digest rejected")
        return {"n_points": 1, "points": [{"adapter_digest": self.loaded[1]}], "loaded_digest": self.loaded[1]}

    def close(self) -> None:
        return None


def test_rollout_ack_rejects_a_stale_digest() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    thread = threading.Thread(target=serve, args=("127.0.0.1", port, _FakeSession(), "w0"), daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "endpoints.json"
        path.write_text(
            json.dumps(
                {"workers": [{"worker_id": "w0", "host": "127.0.0.1", "port": port, "cuda_visible_devices": "4"}]}
            ),
            encoding="utf-8",
        )
        pool = RolloutPool(path, timeout_s=5)
        pool.load_all("v1", "/adapter", "digest-a")
        try:
            pool.collect_all({"expected_version": "v0", "expected_digest": "digest-old", "queries": []})
        except RuntimeError as exc:
            assert "old digest" in str(exc) or "failed" in str(exc)
        else:
            raise AssertionError("stale policy should be rejected")
        shard = pool.collect_all({"expected_version": "v1", "expected_digest": "digest-a", "queries": [{"query_id": "q"}]})
        assert shard[0]["loaded_digest"] == "digest-a"
        pool.quiesce_all(0)
        pool.shutdown_all()
    thread.join(timeout=2)


def test_sealed_round_recovery_skips_a_committed_selection() -> None:
    from trim.training.sft_engine.online_runtime import inspect_sealed_round

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        adapter = root / "init"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        digest = sha256_tree(adapter)
        profile = PROFILES["student_onpolicy_sft"]
        row = SFTBatchRow(
            row_id="s1",
            experiment="student_onpolicy_sft",
            query_id="q",
            episode_id="e",
            decision_id="d",
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[True],
            weight=0.5,
            policy_version="student:init",
            prompt_owner=profile.prompt_owner,
            target_source=profile.supervision_provider,
        )
        round_dir = root / "rounds" / "r000000"
        round_dir.mkdir(parents=True)
        (round_dir / "targets.json").write_text(
            json.dumps({"policy": {"version": "v", "digest": digest}, "rows": [row.to_dict()]}),
            encoding="utf-8",
        )
        args = Namespace(
            init_adapter=str(adapter),
            out=str(root / "run"),
            max_model_len=32,
            global_batch_size=128,
            seed=1,
            num_updates=200,
        )
        decision = inspect_sealed_round(args, round_dir)
        assert decision["action"] == "update"
        assert decision["legacy_targets_without_ready"] is True
        checkpoint = Path(args.out) / "checkpoints" / "step-000001"
        checkpoint.mkdir(parents=True)
        (checkpoint / "COMMITTED.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "SFT_CHECKPOINT.json").write_text(
            json.dumps({"selection_digest": decision["selection_digest"], "adapter_digest": digest, "optimizer_step": 1}),
            encoding="utf-8",
        )
        again = inspect_sealed_round(args, round_dir)
        assert again["action"] == "skip"
        wrong = dict(json.loads((round_dir / "targets.json").read_text(encoding="utf-8")))
        wrong["policy"] = {"version": "v", "digest": "not-the-adapter"}
        (round_dir / "targets.json").write_text(json.dumps(wrong), encoding="utf-8")
        # The committed selection no longer matches because we only skip on digest equality
        # after the policy check. A mismatched policy is refused before the skip.
        refused = inspect_sealed_round(args, round_dir)
        assert refused["action"] == "refuse"
        assert fingerprint({"ok": True})


def _run() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    _run()
