"""Student on-policy SFT: fresh Zero states, TRIM targets, one CE update."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trim.training.dist_runtime import barrier, shard_for_rank
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_tree
from trim.training.sft_ablation.session import append_metrics, prepare_directory, seal_contract
from trim.training.student_onpolicy_sft.collector import decode_decision_point, encode_decision_point
from trim.training.student_onpolicy_sft.targets import build_legal_ce_targets


@dataclass
class LoopState:
    optimizer_step: int = 0
    round_id: int = 0
    empty_rounds: int = 0


@dataclass
class Policy:
    version: str
    digest: str


def run_online_ce(ports: Any, *, max_updates: int, max_empty_rounds: int = 3) -> dict[str, Any]:
    if max_updates < 1 or max_empty_rounds < 1:
        raise ValueError("invalid update/empty-round budget")
    state = ports.restore_or_initialize()
    while state.optimizer_step < max_updates:
        policy = ports.publish_and_verify_policy(state)
        points = ports.collect_fresh_student_points(policy, state.round_id)
        if any(point.policy_version != policy.version for point in points):
            raise ValueError("mixed rollout policy versions")
        rows, audit = ports.build_legal_ce_targets(points, policy)
        ports.persist_round_data(state, policy, rows, audit)
        if not rows:
            state.empty_rounds += 1
            ports.record_empty_round(state, audit)
            if state.empty_rounds >= max_empty_rounds:
                raise RuntimeError("consecutive rounds without legal SFT targets")
        else:
            result = ports.ce_optimizer_update(rows)
            if not result.updated:
                raise RuntimeError("nonempty legal batch produced no update")
            state.optimizer_step += 1
            state.empty_rounds = 0
            ports.save_and_publish_checkpoint(state, result)
        state.round_id += 1
        ports.persist_loop_state(state)
    return ports.summary(state)


class StudentPorts:
    """Collective ports. Every rank calls every method; rank 0 owns AdamW."""

    def __init__(self, args: Any, dist: Any) -> None:
        self.args = args
        self.dist = dist
        self.out = Path(args.out)
        self.queries: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.contract: dict[str, Any] = {}
        self.trainer = None
        self._uses = 0

    def restore_or_initialize(self) -> LoopState:
        self.queries, self.info = prepare_directory(self.args, self.dist)
        self.contract = seal_contract(self.args, self.dist, query_info=self.info, archive_digest="none")
        state = LoopState()
        if int(self.dist.rank) == 0:
            from trim.training.sft_ablation.checkpoint import latest_committed, load_committed
            from trim.training.sft_ablation.trainer import build_hf_trainer

            self.trainer = build_hf_trainer(self.args)
            resume = Path(self.args.resume) if self.args.resume else latest_committed(self.out / "checkpoints")
            if resume is not None:
                manifest = load_committed(
                    resume,
                    model=self.trainer.model,
                    optimizer=self.trainer.optimizer,
                    scheduler=self.trainer.scheduler,
                    identity=self.contract["immutable"],
                )
                state.optimizer_step = int(manifest["optimizer_step"])
                state.round_id = int(manifest["next_round_id"])
            atomic_write_json(
                self.out / "loop_state.json",
                {
                    "optimizer_step": state.optimizer_step,
                    "round_id": state.round_id,
                    "empty_rounds": state.empty_rounds,
                },
            )
        barrier()
        if int(self.dist.rank) != 0 and (self.out / "loop_state.json").is_file():
            saved = read_json(self.out / "loop_state.json")
            state.optimizer_step = int(saved["optimizer_step"])
            state.round_id = int(saved["round_id"])
            state.empty_rounds = int(saved["empty_rounds"])
        return state

    def publish_and_verify_policy(self, state: LoopState) -> Policy:
        current = self.out / "adapters" / "current"
        if int(self.dist.rank) == 0 and self.trainer is not None and hasattr(self.trainer.model, "save_pretrained"):
            current.mkdir(parents=True, exist_ok=True)
            self.trainer.model.save_pretrained(current)
        barrier()
        if current.is_dir():
            digest = sha256_tree(current)
        elif str(self.args.init_adapter or "") and Path(self.args.init_adapter).exists():
            digest = sha256_tree(Path(self.args.init_adapter))
        else:
            digest = "base"
        observed = self.out / "policy_obs" / f"r{state.round_id}"
        observed.mkdir(parents=True, exist_ok=True)
        atomic_write_json(observed / f"{int(self.dist.rank)}.json", {"digest": digest})
        barrier()
        if int(self.dist.rank) == 0:
            seen = [read_json(observed / f"{rank}.json")["digest"] for rank in range(int(self.dist.world_size))]
            if len(set(seen)) != 1:
                raise RuntimeError(f"rollout adapter digest mismatch: {seen}")
        barrier()
        return Policy(version=f"student_onpolicy_sft:step{state.optimizer_step}:{digest[:16]}", digest=digest)

    def collect_fresh_student_points(self, policy: Policy, round_id: int) -> list[Any]:
        chosen = self._queries_for_round(round_id)
        owned = shard_for_rank(chosen, rank=int(self.dist.rank), world_size=int(self.dist.world_size))
        from trim.training.sft_ablation.replicas import run_replica_jobs

        partials = run_replica_jobs(
            self.args,
            self.dist,
            task="rollout",
            mask_name="zero",
            payload={
                "queries": owned,
                "policy_version": policy.version,
                "adapter": str(self.out / "adapters" / "current") if (self.out / "adapters" / "current").exists() else self.args.init_adapter,
                "adapter_digest": policy.digest,
            },
        )
        local = []
        for part in partials:
            if part.get("loaded_digest") not in {None, policy.digest}:
                raise RuntimeError("replica loaded a different adapter digest")
            local.extend(part.get("points") or [])
        shard = self.out / "rounds" / f"r{round_id:06d}" / "points" / str(int(self.dist.rank))
        shard.mkdir(parents=True, exist_ok=True)
        atomic_write_json(shard / "DONE.json", {"points": local, "query_ids": [row["query_id"] for row in owned]})
        barrier()
        merged = self.out / "rounds" / f"r{round_id:06d}" / "points.json"
        if int(self.dist.rank) == 0:
            points = []
            for rank in range(int(self.dist.world_size)):
                points.extend(read_json(self.out / "rounds" / f"r{round_id:06d}" / "points" / str(rank) / "DONE.json")["points"])
            atomic_write_json(merged, points)
        barrier()
        records = read_json(merged)
        return [
            decode_decision_point(record, adapter_digest=policy.digest, policy_version=policy.version)
            for record in records
        ]

    def build_legal_ce_targets(self, points: list[Any], policy: Policy) -> tuple[list[Any], dict[str, Any]]:
        if not points:
            return [], {"n_rows": 0, "rejected": [], "collection_semantics": "student_states_for_ce"}
        from trim.training.sft_ablation.encoding import load_bound_encoding

        encoder = load_bound_encoding(
            self.args.model_name,
            max_model_len=int(self.args.max_model_len),
            max_new_tokens=int(self.args.max_new_tokens),
        )
        return build_legal_ce_targets(
            points,
            policy_version=policy.version,
            adapter_digest=policy.digest,
            enc=encoder,
            max_length=int(self.args.max_model_len),
            per_trajectory=int(self.args.states_per_trajectory),
            seed=int(self.args.seed),
        )

    def persist_round_data(self, state: LoopState, policy: Policy, rows: list[Any], audit: dict[str, Any]) -> None:
        if int(self.dist.rank) != 0:
            return
        directory = self.out / "rounds" / f"r{state.round_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            directory / "targets.json",
            {"policy": {"version": policy.version, "digest": policy.digest}, "rows": [row.to_dict() for row in rows], "audit": audit},
        )

    def record_empty_round(self, state: LoopState, audit: dict[str, Any]) -> None:
        if int(self.dist.rank) == 0:
            append_metrics(
                self.out,
                {
                    "experiment_id": "student_onpolicy_sft",
                    "optimizer_step": state.optimizer_step,
                    "round_id": state.round_id,
                    "empty_round": True,
                    "n_rows": 0,
                    "audit": audit,
                },
            )

    def ce_optimizer_update(self, rows: list[Any]) -> Any:
        from trim.training.sft_ablation.fixed_train import take_batch
        from trim.training.sft_ablation.trainer import CEUpdateResult

        path = self.out / "last_update.json"
        if int(self.dist.rank) == 0:
            batch, _cursor, uses = take_batch(rows, cursor=0, batch_size=min(len(rows), int(self.args.batch_size)), seed=int(self.args.seed))
            self._uses += sum(uses.values())
            result = self.trainer.update(batch)
            atomic_write_json(path, result.payload)
        barrier()
        payload = read_json(path)
        return CEUpdateResult(payload)

    def save_and_publish_checkpoint(self, state: LoopState, result: Any) -> None:
        if int(self.dist.rank) == 0:
            from trim.training.sft_ablation.checkpoint import publish_checkpoint, write_latest
            from trim.training.sft_ablation.contracts import package_digest

            digest = self.trainer.adapter_digest()
            published = publish_checkpoint(
                self.out / "checkpoints" / f"step-{state.optimizer_step:06d}",
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler,
                identity=self.contract["immutable"],
                optimizer_step=state.optimizer_step,
                next_round_id=state.round_id + 1,
                policy_version=f"student_onpolicy_sft:step{state.optimizer_step}",
                adapter_digest=digest,
                source_digest=package_digest([]),
                sampler={"round_id": state.round_id + 1, "uses": self._uses},
            )
            write_latest(self.out, published)
            final = self.out / "adapters" / "final"
            final.mkdir(parents=True, exist_ok=True)
            if hasattr(self.trainer.model, "save_pretrained"):
                self.trainer.model.save_pretrained(final)
            append_metrics(
                self.out,
                {
                    "experiment_id": "student_onpolicy_sft",
                    "optimizer_step": state.optimizer_step,
                    "round_id": state.round_id,
                    "policy_version": f"student_onpolicy_sft:step{state.optimizer_step}",
                    "adapter_digest": digest,
                    "rollout_adapter_digest": digest,
                    "latest_committed_adapter_digest": digest,
                    "n_rows": result.n_rows,
                    "weighted_tokens": result.weighted_tokens,
                    "loss_ce": result.loss_ce,
                    "grad_norm": result.grad_norm,
                    "learning_rate": float(self.args.learning_rate),
                    "empty_rounds": state.empty_rounds,
                },
            )
        barrier()

    def persist_loop_state(self, state: LoopState) -> None:
        if int(self.dist.rank) == 0:
            atomic_write_json(
                self.out / "loop_state.json",
                {
                    "optimizer_step": state.optimizer_step,
                    "round_id": state.round_id,
                    "empty_rounds": state.empty_rounds,
                },
            )
        barrier()

    def summary(self, state: LoopState) -> dict[str, Any]:
        return {
            "experiment": "student_onpolicy_sft",
            "optimizer_step": state.optimizer_step,
            "round_id": state.round_id,
            "empty_rounds": state.empty_rounds,
        }

    def _queries_for_round(self, round_id: int) -> list[dict[str, Any]]:
        path = self.out / "rounds" / f"r{round_id:06d}" / "queries.json"
        if int(self.dist.rank) == 0:
            rng = random.Random(f"{int(self.args.seed)}:{round_id}")
            order = list(self.queries)
            rng.shuffle(order)
            width = min(len(order), max(1, int(self.args.batch_size)))
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, order[:width])
        barrier()
        return read_json(path)


def run(args: Any, dist: Any) -> int:
    if args.stage != "all":
        raise SystemExit("student_onpolicy_sft only accepts --stage all")
    run_online_ce(
        StudentPorts(args, dist),
        max_updates=int(args.num_updates),
        max_empty_rounds=int(args.max_empty_rounds),
    )
    return 0


def run_rollout_job(job: dict[str, Any]) -> dict[str, Any]:
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.adapters.components import zero_mask
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.student_onpolicy_sft.collector import encode_decision_point, student_collection_kwargs
    from trim.training.vllm_hybrid import VLLMGenerateClient

    adapter = str(job.get("adapter") or "")
    if adapter and Path(adapter).exists():
        digest = sha256_tree(Path(adapter))
    else:
        digest = "base"
    expected = str(job.get("adapter_digest") or "")
    if expected and expected != digest:
        raise RuntimeError("worker adapter digest does not match the published policy")
    enc = load_model_encoding(str(job["model_name"]), max_model_len=int(job["max_model_len"]), max_new_tokens=int(job["max_new_tokens"]))
    client = VLLMGenerateClient(
        model_path=str(job["model_name"]),
        session_dir=Path(job["session_dir"]),
        tensor_parallel_size=int(job.get("tensor_parallel_size") or 1),
        max_model_len=int(job["max_model_len"]),
        lora_path=adapter or None,
    )
    mask = zero_mask("Harness-1")
    try:
        client.start()
        groups = rollout_queries_batched(
            client.generate_batch,
            list(job.get("queries") or []),
            **student_collection_kwargs(
                student_zero_mask=mask,
                policy_version=str(job["policy_version"]),
                group_size=int(job["group_size"]),
                max_turns=int(job["max_turns"]),
                max_new_tokens=int(job["max_new_tokens"]),
                seed=int(job["seed"]),
                enc=enc,
                train_env=str(job["train_env"]),
            ),
        )
    finally:
        client.close()
    points = []
    for group in groups:
        for point in group.decision_points:
            points.append(encode_decision_point(point, adapter_digest=digest))
    return {"ok": True, "points": points, "loaded_digest": digest, "collection_semantics": "student_states_for_ce"}
