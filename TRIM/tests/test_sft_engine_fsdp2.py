"""CPU checks for the FSDP2 SFT backend: routing, contract guards, dispatch, DCP round trip.

CUDA-level sharding (fully_shard, DTensor collectives, MoE forward) is only
proven by the multi-GPU smoke run; these tests pin the pure-Python seams.
"""

from __future__ import annotations

import json
import time
import threading
import socket
import tempfile
from argparse import Namespace
from pathlib import Path

from trim.training.sft_ablation.cli import parse_sft_args
from trim.training.sft_ablation.contracts import SFTBatchRow, fingerprint
from trim.training.sft_ablation.dist_launch import launch_sft, v2_train_requested
from trim.training.sft_ablation.io_util import sha256_tree
from trim.training.sft_ablation.contracts import PROFILES
from trim.training.sft_engine.accelerate_engine import backward_sync_context, replicated_scalar
from trim.training.sft_engine.contract import adapter_config_from, build_v2_contract, require_exact_resume
from trim.training.sft_engine.backend import ACCELERATED_BACKENDS, is_accelerated


def _base_args(**overrides):
    payload = dict(
        experiment="teacher_sft",
        global_batch_size=32,
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
    payload.update(overrides)
    return Namespace(**payload)


def test_accelerate_fsdp2_cli_and_launch_routing() -> None:
    assert "accelerate_fsdp2" in ACCELERATED_BACKENDS
    assert is_accelerated("accelerate_fsdp2") and is_accelerated("accelerate_ddp") and not is_accelerated("legacy")
    args = parse_sft_args(
        "teacher_sft",
        [
            "--model_name", "m",
            "--out", "/tmp/t",
            "--ce-backend", "accelerate_fsdp2",
            "--stage", "train",
            "--micro-batch-size", "1",
            "--gradient-checkpointing", "non_reentrant",
            "--global-batch-size", "64",
        ],
    )
    assert args.ce_backend == "accelerate_fsdp2"
    assert args.gradient_checkpointing == "non_reentrant"
    for bad in (
        ["--ce-backend", "accelerate_fsdp2", "--stage", "train", "--micro-batch-size", "2"],
        ["--ce-backend", "accelerate_fsdp2", "--stage", "all"],
        ["--ce-backend", "accelerate_fsdp2", "--stage", "train", "--logits-mode", "full"],
    ):
        try:
            parse_sft_args("teacher_sft", ["--model_name", "m", "--out", "/tmp/t", "--micro-batch-size", "1", *bad])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"should refuse: {bad}")

    assert v2_train_requested(["--ce-backend", "accelerate_fsdp2", "--stage", "train"], "teacher_sft")
    assert v2_train_requested(["--ce-backend", "accelerate_fsdp2"], "student_onpolicy_sft")

    import io
    from contextlib import redirect_stdout

    def worker(parsed, dist):
        raise AssertionError(f"dry-run started a worker {parsed} {dist}")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = launch_sft(
            script=Path(__file__),
            experiment="teacher_sft",
            worker=worker,
            argv=[
                "--ce-backend", "accelerate_fsdp2",
                "--stage", "train",
                "--micro-batch-size", "1",
                "--gradient-checkpointing", "off",
                "--nnodes", "4",
                "--nproc-per-node", "8",
                "--model_name", "m",
                "--out", "/tmp/teacher-fsdp2",
                "--dry-run",
            ],
        )
    assert code == 0
    plan = json.loads(buffer.getvalue())
    assert plan["ce_backend"] == "accelerate_fsdp2"
    assert plan["gradient_ranks"] == 32
    assert plan["gradient_checkpointing"] == "off"
    assert plan["pin_local_cuda_device"] is False
    assert plan["process_group"] == "accelerate"


def test_fsdp2_contract_fields_and_resume_guards() -> None:
    args = _base_args()
    ddp = build_v2_contract(
        args,
        dataset_digest="d",
        source_contract_digest="s",
        tokenizer_digest="t",
        gradient_ranks=32,
        nnodes=4,
        nproc_per_node=8,
        init_adapter_digest="a",
    )
    fsdp2 = build_v2_contract(
        args,
        dataset_digest="d",
        source_contract_digest="s",
        tokenizer_digest="t",
        gradient_ranks=32,
        nnodes=4,
        nproc_per_node=8,
        init_adapter_digest="a",
        backend="accelerate_fsdp2",
    )
    assert ddp["optimization"]["backend"] == "accelerate_ddp"
    assert fsdp2["optimization"]["backend"] == "accelerate_fsdp2"
    assert fsdp2["optimization"]["fsdp2"]["fsdp_version"] == 2
    assert fsdp2["optimization"]["fsdp2"]["wrap"] == "transformer_based_wrap"
    require_exact_resume(fsdp2, dict(fsdp2))
    # DDP↔FSDP2 is never an exact resume: the backend field refuses it.
    try:
        require_exact_resume(ddp, fsdp2)
    except ValueError as exc:
        assert "optimization.backend" in str(exc)
    else:
        raise AssertionError("backend change should refuse exact resume")
    # gradient_checkpointing is now part of the resume contract (no silent switch).
    other = build_v2_contract(
        _base_args(gradient_checkpointing="non_reentrant"),
        dataset_digest="d",
        source_contract_digest="s",
        tokenizer_digest="t",
        gradient_ranks=32,
        nnodes=4,
        nproc_per_node=8,
        init_adapter_digest="a",
        backend="accelerate_fsdp2",
    )
    try:
        require_exact_resume(fsdp2, other)
    except ValueError as exc:
        assert "gradient_checkpointing" in str(exc)
    else:
        raise AssertionError("checkpointing change should refuse exact resume")
    # v2-schema contracts (old DDP checkpoints) still exact-resume field-wise.
    old = dict(fsdp2)
    old["schema_version"] = "sft_run_contract_v2"
    require_exact_resume(old, dict(fsdp2))
    # adapter_config records PEFT facts when available.
    with tempfile.TemporaryDirectory() as tmp:
        adapter = Path(tmp) / "a"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 32, "target_modules": ["q_proj"]}), encoding="utf-8")
        config = adapter_config_from(adapter)
        assert config["r"] == 32 and config["target_modules"] == ["q_proj"]
        seeded = build_v2_contract(
            args,
            dataset_digest="d",
            source_contract_digest="s",
            tokenizer_digest="t",
            gradient_ranks=32,
            nnodes=4,
            nproc_per_node=8,
            init_adapter_digest="a",
            backend="accelerate_fsdp2",
            adapter_config=config,
        )
        assert seeded["model"]["adapter_config"]["r"] == 32


class _FakeAcc:
    def __init__(self, fsdp2: bool) -> None:
        self.is_fsdp2 = fsdp2

    def no_sync(self, model):
        del model
        return "ddp_no_sync"


def test_backward_sync_context_dispatch() -> None:
    from contextlib import nullcontext

    ddp = _FakeAcc(False)
    assert backward_sync_context(ddp, None, 0, 3) == "ddp_no_sync"  # non-final slots buffer
    assert backward_sync_context(ddp, None, 1, 3) == "ddp_no_sync"
    assert isinstance(backward_sync_context(ddp, None, 2, 3), nullcontext)  # last slot syncs
    fsdp2 = _FakeAcc(True)
    for slot in range(3):
        # FSDP2 first version: reduce-scatter every micro-step, never no_sync.
        assert isinstance(backward_sync_context(fsdp2, None, slot, 3), nullcontext)
    import torch

    value = torch.tensor(3.5)
    assert float(replicated_scalar(value)) == 3.5


def test_sharded_train_state_cpu_roundtrip() -> None:
    import torch

    from trim.training.sft_engine.sharded_checkpoint import load_sharded_train_state, save_sharded_train_state

    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    saved = {name: param.detach().clone() for name, param in model.named_parameters()}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dcp"
        save_sharded_train_state(model, optimizer, path)
        # Poison the live state, then restore from the DCP artifact.
        with torch.no_grad():
            for param in model.parameters():
                param.add_(3.0)
        optimizer.zero_grad()
        report = load_sharded_train_state(model, optimizer, path)
        assert report["missing_keys"] == [] and report["unexpected_keys"] == []
        for name, param in model.named_parameters():
            assert torch.allclose(param, saved[name]), f"{name} not restored"


def test_resolve_retrieval_config_is_strict() -> None:
    from trim.training.projected_teacher_sft.runtime import resolve_retrieval_config

    for bad in (
        None,
        {},
        {"backend": "upstream"},
        {"backend": "upstream", "dataset": "d", "collection_split": "test", "reranker": "r", "surprise": 1},
    ):
        try:
            resolve_retrieval_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should refuse: {bad}")
    try:
        resolve_retrieval_config({"backend": "local_bm25", "dataset": "d", "collection_split": "test", "reranker": "none"})
    except ValueError as exc:
        assert "local_bm25" in str(exc) or "missing" in str(exc)
    else:
        raise AssertionError("local_bm25 without index/docstore/corpus must be refused")


def test_projected_replay_requires_query_text() -> None:
    from trim.training.projected_teacher_sft.runtime import job_env_for_trajectory

    def factory(query, meta, query_id=None):
        return {"query": query, "query_id": query_id, "ended": False}

    for bad in ({"query_id": "q1"}, {"query_id": "q1", "query": ""}, {"query_id": "q1", "query": "   "}):
        try:
            job_env_for_trajectory(bad, encoder=None, state_factory=factory, mask={})
        except ValueError as exc:
            assert "query text" in str(exc)
        else:
            raise AssertionError(f"bad query must be refused: {bad}")
    env = job_env_for_trajectory({"query_id": "q1", "query": "what is X?"}, encoder=None, state_factory=factory, mask={})
    assert env.row["query"] == "what is X?" and env.row["query_id"] == "q1"


def test_sealed_round_transaction_commit_repairs_marks() -> None:
    from trim.training.sft_engine.online_runtime import _repair_consume_marks, inspect_sealed_round

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
            weight=1.0,
            policy_version="student:init",
            prompt_owner=profile.prompt_owner,
            target_source=profile.supervision_provider,
        )
        round_dir = root / "rounds" / "r000007"
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
        # Simulate the crash window: the checkpoint is COMMITTED with full
        # transaction fields but CONSUMED/loop_state were never written.
        checkpoint = Path(args.out) / "checkpoints" / "step-000001"
        checkpoint.mkdir(parents=True)
        (checkpoint / "COMMITTED.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "SFT_CHECKPOINT.json").write_text(
            json.dumps(
                {
                    "selection_digest": decision["selection_digest"],
                    "source_digest": decision["source_rows_digest"],
                    "rollout_adapter_digest": decision["rollout_adapter_digest"],
                    "pending_round": decision["round_id"],
                    "adapter_digest": digest,
                    "optimizer_step": 1,
                }
            ),
            encoding="utf-8",
        )
        again = inspect_sealed_round(args, round_dir)
        assert again["action"] == "skip"
        assert again["reason"] == "round transaction already committed"
        assert again["consume"]["optimizer_step"] == 1
        _repair_consume_marks(round_dir, again, Path(args.out))
        assert (round_dir / "CONSUMED.json").is_file()
        state = json.loads((Path(args.out) / "loop_state.json").read_text(encoding="utf-8"))
        assert state["optimizer_step"] == 1 and state["round_id"] == 8
        # Repair is idempotent and never regresses a newer loop state.
        _repair_consume_marks(round_dir, again, Path(args.out))
        (Path(args.out) / "loop_state.json").write_text(json.dumps({"optimizer_step": 5, "round_id": 12, "empty_rounds": 0}), encoding="utf-8")
        _repair_consume_marks(round_dir, again, Path(args.out))
        state = json.loads((Path(args.out) / "loop_state.json").read_text(encoding="utf-8"))
        assert state["optimizer_step"] == 5


def test_fsdp2_flag_keeps_the_weighted_ce_math() -> None:
    """is_fsdp2 only changes the sync context; loss/grad must match the DDP run."""
    import copy

    import torch
    from torch import nn

    from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow
    from trim.training.sft_engine.accelerate_engine import update_global_batch

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(40, 8)
            self.out = nn.Linear(8, 40, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False, output_router_logits=False, logits_to_keep=None):
            del attention_mask, use_cache, output_router_logits
            hidden = self.out(self.emb(input_ids))
            selected = hidden.index_select(1, logits_to_keep.to(hidden.device).long())
            return type("Out", (), {"logits": selected})()

    class Acc:
        def __init__(self, fsdp2: bool) -> None:
            self.device = torch.device("cpu")
            self.num_processes = 1
            self.process_index = 0
            self.gradient_accumulation_steps = 1
            self.is_fsdp2 = fsdp2
            self.no_sync_calls = 0

        def no_sync(self, model):
            del model
            self.no_sync_calls += 1
            from contextlib import nullcontext

            return nullcontext()

        def autocast(self):
            from contextlib import nullcontext

            return nullcontext()

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, params, max_norm):
            return torch.nn.utils.clip_grad_norm_(params, max_norm)

    profile = PROFILES["teacher_sft"]

    def row(row_id: str, prompt: list[int], response: list[int], mask: list[bool], weight: float) -> SFTBatchRow:
        return SFTBatchRow(
            row_id=row_id,
            experiment="teacher_sft",
            query_id="q",
            episode_id="e",
            decision_id=row_id,
            prompt_ids=prompt,
            response_ids=response,
            response_mask=mask,
            weight=weight,
            policy_version="p",
            prompt_owner=profile.prompt_owner,
            target_source=profile.supervision_provider,
        )

    rows = [
        row("a", [1, 2], [3, 4], [True, True], 1.0),
        row("b", [5], [6, 7, 8], [True, False, True], 2.0),
        row("c", [9, 1], [2], [True], 0.5),
    ]
    torch.manual_seed(0)
    ddp_model = Tiny()
    fsdp2_model = Tiny()
    fsdp2_model.load_state_dict(copy.deepcopy(ddp_model.state_dict()))
    ddp_acc, fsdp2_acc = Acc(False), Acc(True)
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
    fsdp2_opt = torch.optim.SGD(fsdp2_model.parameters(), lr=1e-2)
    left = update_global_batch(ddp_acc, ddp_model, ddp_opt, None, rows, 32, grad_clip=1.0e9)
    right = update_global_batch(fsdp2_acc, fsdp2_model, fsdp2_opt, None, rows, 32, grad_clip=1.0e9)
    assert left["loss_ce"] == right["loss_ce"]
    assert left["weighted_tokens"] == right["weighted_tokens"]
    assert ddp_acc.no_sync_calls >= 1  # DDP buffered the non-final slots
    assert fsdp2_acc.no_sync_calls == 0  # FSDP2 never buffers
    for p_left, p_right in zip(ddp_model.parameters(), fsdp2_model.parameters()):
        assert torch.allclose(p_left, p_right)


def test_rollout_collect_is_concurrent_across_workers() -> None:
    from trim.training.student_onpolicy_sft.rollout_service import RolloutPool, serve

    class _SlowSession:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.loaded = None

        def load_policy(self, version, adapter_path, adapter_digest):
            del adapter_path
            self.loaded = (version, adapter_digest)
            return {"loaded_version": version, "loaded_digest": adapter_digest, "probe_result": "fake"}

        def collect(self, request):
            time.sleep(0.6)
            return {"n_points": 1, "points": [{"adapter_digest": self.loaded[1]}], "loaded_digest": self.loaded[1]}

        def close(self):
            return None

    ports = []
    threads = []
    for index in range(3):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()
        thread = threading.Thread(
            target=serve,
            args=("127.0.0.1", port, _SlowSession(f"w{index}"), f"w{index}"),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        ports.append(port)
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", ports[0]), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "endpoints.json"
        path.write_text(
            json.dumps(
                {
                    "workers": [
                        {"worker_id": f"w{i}", "host": "127.0.0.1", "port": port, "cuda_visible_devices": str(i)}
                        for i, port in enumerate(ports)
                    ]
                }
            ),
            encoding="utf-8",
        )
        pool = RolloutPool(path, timeout_s=5)
        pool.load_all("v1", "/adapter", "digest-a")
        started = time.time()
        shards = pool.collect_all(
            {"expected_version": "v1", "expected_digest": "digest-a", "queries": [{"query_id": f"q{i}"} for i in range(6)]}
        )
        wall = time.time() - started
        assert len(shards) == 3
        assert [shard["loaded_digest"] for shard in shards] == ["digest-a"] * 3
        # Serial would be >= 3 * 0.6s; concurrent must beat that by a margin.
        assert wall < 1.7, f"collect_all looks serial: {wall:.2f}s"
        pool.shutdown_all()
    for thread in threads:
        thread.join(timeout=2)
    assert fingerprint({"ok": True})
