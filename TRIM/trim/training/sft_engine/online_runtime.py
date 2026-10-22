"""Student on-policy CE on every training rank.

``phase_jobs`` consumes one sealed round and exits, so rollout and training
do not share a GPU. ``split_pools`` keeps the CE model resident and talks to
an external rollout service on the other devices.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from trim.training.sft_ablation.contracts import SFTBatchRow, fingerprint, package_digest
from trim.training.sft_ablation.fixed_train import select_batch_indices
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_tree
from trim.training.sft_ablation.session import append_metrics
from trim.training.sft_engine.backend import backend_name
from trim.training.sft_engine.checkpoint import find_selection, latest_v2
from trim.training.sft_engine.contract import adapter_config_from, build_v2_contract
from trim.training.sft_engine.profile import summarize_progress
from trim.training.student_onpolicy_sft.runtime import LoopState, Policy, run_online_ce


def run_student_v2(args: Any) -> int:
    if str(args.experiment) != "student_onpolicy_sft":
        raise SystemExit(f"student engine got experiment {args.experiment}")
    layout = str(getattr(args, "online_layout", "phase_jobs") or "phase_jobs")
    if layout == "phase_jobs":
        return run_phase_job(args)
    if layout == "split_pools":
        if not str(getattr(args, "rollout_endpoints", "") or ""):
            raise SystemExit("split_pools requires --rollout-endpoints")
        from trim.training.student_onpolicy_sft.rollout_service import assert_endpoints_disjoint

        assert_endpoints_disjoint(args.rollout_endpoints)
        run_online_ce(
            SplitPoolPorts(args),
            max_updates=int(args.num_updates),
            max_empty_rounds=int(args.max_empty_rounds),
        )
        return 0
    raise SystemExit(f"unknown online layout {layout}")


def run_phase_job(args: Any) -> int:
    """One CE update from a sealed round, then exit."""
    if not str(getattr(args, "resume_round", "") or ""):
        raise SystemExit("phase_jobs requires --resume-round pointing at a sealed targets directory")
    from trim.training.sft_engine.backend import (
        create_training_context,
        make_training_accelerator,
        save_training_checkpoint,
        update,
    )
    from trim.training.sft_engine.fixed_runtime import _now
    from trim.training.sft_engine.distributed import broadcast_ints

    accelerator = make_training_accelerator(args)
    out = Path(args.out)
    decision_path = out / f"phase_decision_{Path(args.resume_round).name}.json"
    if int(accelerator.process_index) == 0:
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(decision_path, inspect_sealed_round(args, Path(args.resume_round)))
    accelerator.wait_for_everyone()
    decision = read_json(decision_path)
    if decision["action"] == "skip":
        if int(accelerator.process_index) == 0:
            _repair_consume_marks(Path(args.resume_round), decision, out)
        accelerator.wait_for_everyone()
        return 0
    if decision["action"] != "update":
        raise RuntimeError(decision.get("reason") or "sealed student round was refused")
    rows = _ordered_round_rows(Path(args.resume_round), int(args.max_model_len))
    logical, _cursor, _uses = select_batch_indices(
        len(rows),
        cursor=0,
        batch_size=min(len(rows), int(args.global_batch_size)),
        seed=int(args.seed),
    )
    indices = logical if int(accelerator.process_index) == 0 else None
    indices = broadcast_ints(indices, device=accelerator.device, src=0)
    batch = [rows[index] for index in indices]
    selection = fingerprint([row.row_id for row in batch])
    if selection != decision["selection_digest"]:
        raise RuntimeError("rank disagreed on the sealed-round selection")
    contract = _student_contract(args, accelerator)
    _write_contract(out, contract, accelerator)
    ctx = create_training_context(
        args,
        adapter_path=str(decision["adapter"]),
        contract=contract,
        resume_dir=Path(decision["resume"]) if decision.get("resume") else None,
        accelerator=accelerator,
    )
    started = _now()
    result = update(
        ctx,
        batch,
        int(args.max_model_len),
        grad_clip=float(args.grad_clip),
        sync_cuda=True,
    )
    if not result["updated"]:
        raise RuntimeError("sealed student round produced no update")
    step = int(decision["optimizer_step"]) + 1
    published = save_training_checkpoint(
        ctx,
        out / "checkpoints" / f"step-{step:06d}",
        contract=contract,
        optimizer_step=step,
        policy_version=f"student_onpolicy_sft:step{step}",
        source_digest=package_digest(batch),
        rollout_adapter_digest=str(decision["rollout_adapter_digest"]),
        sampler={"cursor": 0, "round_id": int(decision["round_id"]) + 1, "row_ids": [row.row_id for row in batch]},
        selection_digest=selection,
        pending_round=int(decision["round_id"]),
    )
    if int(accelerator.process_index) == 0:
        manifest = read_json(published / "SFT_CHECKPOINT.json")
        append_metrics(
            out,
            _metric(args, step, decision["round_id"], result, manifest, decision["rollout_adapter_digest"], _now() - started),
        )
        atomic_write_json(
            out / "loop_state.json",
            {"optimizer_step": step, "round_id": int(decision["round_id"]) + 1, "empty_rounds": 0},
        )
        atomic_write_json(out / "PERF_REPORT.json", summarize_progress(out))
        atomic_write_json(Path(args.resume_round) / "CONSUMED.json", {"selection_digest": selection, "checkpoint": str(published)})
    accelerator.wait_for_everyone()
    return 0


def inspect_sealed_round(args: Any, round_dir: Path) -> dict[str, Any]:
    targets_path = round_dir / "targets.json"
    if not targets_path.is_file():
        return {"action": "refuse", "reason": f"missing {targets_path}"}
    try:
        targets = read_json(targets_path)
        rows = [SFTBatchRow.from_dict(item) for item in targets.get("rows") or []]
        for row in rows:
            row.validate(int(args.max_model_len))
            if row.experiment != "student_onpolicy_sft":
                return {"action": "refuse", "reason": f"row {row.row_id} is {row.experiment}"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"action": "refuse", "reason": f"incomplete targets: {exc}"}
    if not rows:
        return {"action": "refuse", "reason": "sealed round has no supervised rows"}
    policy = dict(targets.get("policy") or {})
    policy_digest = str(policy.get("digest") or "")
    if not policy_digest:
        return {"action": "refuse", "reason": "sealed round has no source policy digest"}
    ordered = sorted(rows, key=lambda row: row.row_id)
    logical, _cursor, _uses = select_batch_indices(
        len(ordered),
        cursor=0,
        batch_size=min(len(ordered), int(args.global_batch_size)),
        seed=int(args.seed),
    )
    chosen = [ordered[index] for index in logical]
    selection = fingerprint([row.row_id for row in chosen])
    source_rows_digest = package_digest(chosen)
    round_id = _round_id_from_path(round_dir)
    # 1. A committed checkpoint for THIS transaction means the update already
    #    ran: only the CONSUMED/loop_state marks may be missing (crash between
    #    commit and bookkeeping). Do not step a second time (11.6).
    committed = find_round_transaction(
        Path(args.out) / "checkpoints",
        round_id=round_id,
        policy_digest=policy_digest,
        source_rows_digest=source_rows_digest,
        selection_digest=selection,
    )
    if committed is not None:
        manifest = read_json(committed / "SFT_CHECKPOINT.json")
        return {
            "action": "skip",
            "reason": "round transaction already committed",
            "selection_digest": selection,
            "checkpoint": str(committed),
            "optimizer_step": int(manifest.get("optimizer_step") or 0),
            "consume": {
                "selection_digest": selection,
                "checkpoint": str(committed),
                "optimizer_step": int(manifest.get("optimizer_step") or 0),
                "round_id": round_id,
            },
        }
    init_digest = sha256_tree(Path(args.init_adapter)) if str(args.init_adapter or "") and Path(args.init_adapter).exists() else ""
    latest = latest_v2(Path(args.out) / "checkpoints")
    if latest is None:
        if policy_digest != init_digest:
            return {"action": "refuse", "reason": "round policy digest does not match the initial adapter"}
        adapter = str(args.init_adapter)
        resume = None
        step = 0
    else:
        manifest = read_json(latest / "SFT_CHECKPOINT.json")
        if policy_digest != str(manifest.get("adapter_digest") or ""):
            return {"action": "refuse", "reason": "round policy digest does not match the latest committed adapter"}
        adapter = str(latest / "adapter")
        resume = str(latest)
        step = int(manifest["optimizer_step"])
    if step >= int(args.num_updates):
        return {"action": "skip", "reason": "update budget already reached", "optimizer_step": step}
    # 2. Legacy selection-only match (pre-transaction checkpoints).
    existing = find_selection(Path(args.out) / "checkpoints", selection)
    if existing is not None:
        return {
            "action": "skip",
            "reason": "selection already committed",
            "selection_digest": selection,
            "checkpoint": str(existing),
            "optimizer_step": step,
            "consume": {
                "selection_digest": selection,
                "checkpoint": str(existing),
                "optimizer_step": step,
                "round_id": round_id,
            },
        }
    return {
        "action": "update",
        "adapter": adapter,
        "resume": resume,
        "optimizer_step": step,
        "selection_digest": selection,
        "source_rows_digest": source_rows_digest,
        "rollout_adapter_digest": policy_digest,
        "round_id": round_id,
        "legacy_targets_without_ready": not (round_dir / "READY.json").is_file(),
        "n_rows": len(chosen),
    }


def _repair_consume_marks(round_dir: Path, decision: dict[str, Any], out: Path) -> None:
    """Finish the COMMITTED→CONSUMED bookkeeping an interrupted run left behind."""
    consume = dict(decision.get("consume") or {})
    if not consume:
        return
    consumed = round_dir / "CONSUMED.json"
    if not consumed.is_file():
        atomic_write_json(consumed, {"selection_digest": consume.get("selection_digest"), "checkpoint": consume.get("checkpoint")})
    state_path = out / "loop_state.json"
    want_step = int(consume.get("optimizer_step") or 0)
    want_round = int(consume.get("round_id") or 0) + 1
    if state_path.is_file():
        saved = read_json(state_path)
        if int(saved.get("optimizer_step") or 0) >= want_step:
            return  # never regress a newer loop state
    atomic_write_json(state_path, {"optimizer_step": want_step, "round_id": want_round, "empty_rounds": 0})


def find_round_transaction(
    root: Path,
    *,
    round_id: int,
    policy_digest: str,
    source_rows_digest: str,
    selection_digest: str,
) -> Path | None:
    """Locate a COMMITTED checkpoint written for exactly this round transaction."""
    root = Path(root)
    if not root.is_dir() or not selection_digest:
        return None
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        manifest_path = path / "SFT_CHECKPOINT.json"
        if not manifest_path.is_file() or not (path / "COMMITTED.json").is_file():
            continue
        manifest = read_json(manifest_path)
        if str(manifest.get("selection_digest") or "") != selection_digest:
            continue
        if str(manifest.get("rollout_adapter_digest") or "") != policy_digest:
            continue
        if str(manifest.get("source_digest") or "") != source_rows_digest:
            continue
        if int(manifest.get("pending_round") if manifest.get("pending_round") is not None else -1) != int(round_id):
            continue
        return path
    return None


class SplitPoolPorts:
    """All CE ranks call every method. Rank 0 owns rollout I/O and target building."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.out = Path(args.out)
        self.acc = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.contract: dict[str, Any] = {}
        self.queries: list[dict[str, Any]] = []
        self.source_policy: Policy | None = None
        self._round_rows: list[SFTBatchRow] = []
        self.fail_path = self.out / "FAILED.json"

    @property
    def rank(self) -> int:
        return int(self.acc.process_index)

    def restore_or_initialize(self) -> LoopState:
        from trim.training.sft_engine.backend import create_training_context, make_training_accelerator

        self.acc = make_training_accelerator(self.args)
        if self.rank == 0:
            self.out.mkdir(parents=True, exist_ok=True)
            from trim.training.sft_ablation.query_pool import load_train_queries

            self.queries, _info = load_train_queries(self.args)
        self.contract = _student_contract(self.args, self.acc)
        _write_contract(self.out, self.contract, self.acc)
        latest = latest_v2(self.out / "checkpoints")
        adapter = str(latest / "adapter") if latest is not None else str(self.args.init_adapter)
        ctx = create_training_context(
            self.args,
            adapter_path=adapter,
            contract=self.contract,
            resume_dir=latest,
            accelerator=self.acc,
        )
        self.model = ctx.model
        self.optimizer = ctx.optimizer
        self.scheduler = ctx.scheduler
        state = LoopState()
        if latest is not None:
            manifest = ctx.resume_manifest
            if manifest is None:
                raise RuntimeError("committed checkpoint was not restored")
            state.optimizer_step = int(manifest["optimizer_step"])
            state.round_id = int(manifest["sampler"].get("round_id") or manifest["optimizer_step"])
        if self.rank == 0:
            atomic_write_json(self.out / "loop_state.json", _state_dict(state))
        self.acc.wait_for_everyone()
        return state

    def publish_and_verify_policy(self, state: LoopState) -> Policy:
        pointer = self.out / "policy_current.json"
        ack = self.out / "policy_ack.json"
        staging = self.out / "policies" / f".tmp-step-{state.optimizer_step:06d}"
        # Adapter export is collective (FSDP2 DTensor aggregation), so every
        # rank enters it; only the pointer/ACK work stays on rank 0.
        self._guard(lambda: self._export_policy(state, staging))
        self.acc.wait_for_everyone()
        if self.rank == 0:
            self._guard(lambda: self._publish(state, staging, pointer, ack))
        self.acc.wait_for_everyone()
        self._wait_value(ack, "optimizer_step", int(state.optimizer_step))
        payload = read_json(pointer)
        if int(payload["optimizer_step"]) != int(state.optimizer_step):
            raise RuntimeError("published policy is for a different optimizer step")
        policy = Policy(version=str(payload["version"]), digest=str(payload["digest"]))
        self.source_policy = policy
        return policy

    def collect_fresh_student_points(self, policy: Policy, round_id: int) -> list[Any]:
        if self.rank != 0:
            return []
        from trim.training.student_onpolicy_sft.collector import decode_decision_point
        from trim.training.student_onpolicy_sft.rollout_service import RolloutPool

        def _collect() -> list[Any]:
            pool = RolloutPool(self.args.rollout_endpoints)
            current = read_json(self.out / "policy_current.json")
            pool.load_all(policy.version, str(current["path"]), policy.digest)
            queries = self._queries_for_round(round_id)
            shards = pool.collect_all(
                {
                    "round_id": int(round_id),
                    "expected_version": policy.version,
                    "expected_digest": policy.digest,
                    "queries": queries,
                    "sampling_config": {
                        "group_size": int(self.args.group_size),
                        "max_turns": int(self.args.max_turns),
                        "seed": int(self.args.seed),
                    },
                    "shard_dir": str(self.out / "rounds" / f"r{round_id:06d}" / "shards"),
                }
            )
            pool.quiesce_all(round_id)
            points = []
            for shard in shards:
                if shard.get("loaded_digest") != policy.digest:
                    raise RuntimeError("rollout shard came from a different adapter")
                for record in shard.get("points") or []:
                    points.append(decode_decision_point(record, adapter_digest=policy.digest, policy_version=policy.version))
            return points

        return self._guard(_collect)

    def build_legal_ce_targets(self, points: list[Any], policy: Policy) -> tuple[list[Any], dict[str, Any]]:
        ready = self.out / "rounds" / f"r{self._current_round():06d}" / "READY.json"
        if self.rank == 0:
            def _build() -> None:
                rows, audit = self._label(points, policy)
                self._write_round(self._current_round(), policy, rows, audit)
            self._guard(_build)
        self._wait(ready)
        payload = read_json(ready.parent / "targets.json")
        rows = [SFTBatchRow.from_dict(item) for item in payload.get("rows") or []]
        self._round_rows = rows
        return rows, dict(payload.get("audit") or {})

    def persist_round_data(self, state: LoopState, policy: Policy, rows: list[Any], audit: dict[str, Any]) -> None:
        del state, policy, rows, audit
        ready = self.out / "rounds" / f"r{self._current_round():06d}" / "READY.json"
        if not ready.is_file():
            raise RuntimeError("round targets were not sealed")

    def record_empty_round(self, state: LoopState, audit: dict[str, Any]) -> None:
        if self.rank == 0:
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
        from trim.training.sft_ablation.trainer import CEUpdateResult
        from trim.training.sft_engine.accelerate_engine import update_global_batch
        from trim.training.sft_engine.distributed import broadcast_ints
        from trim.training.sft_engine.fixed_runtime import _now

        ordered = sorted(rows, key=lambda row: row.row_id)
        width = min(len(ordered), int(self.args.global_batch_size))
        logical, _cursor, _uses = select_batch_indices(len(ordered), cursor=0, batch_size=width, seed=int(self.args.seed))
        indices = logical if self.rank == 0 else None
        indices = broadcast_ints(indices, device=self.acc.device, src=0)
        batch = [ordered[index] for index in indices]
        self._batch = batch
        started = _now()
        result = update_global_batch(
            self.acc,
            self.model,
            self.optimizer,
            self.scheduler,
            batch,
            int(self.args.max_model_len),
            grad_clip=float(self.args.grad_clip),
            sync_cuda=False,
        )
        result["update_wall_s"] = _now() - started
        result["selection_digest"] = fingerprint([row.row_id for row in batch])
        return CEUpdateResult(result)

    def save_and_publish_checkpoint(self, state: LoopState, result: Any) -> None:
        from trim.training.sft_engine.backend import export_policy_adapter, save_training_checkpoint

        if self.source_policy is None:
            raise RuntimeError("missing rollout policy for this update")
        batch = getattr(self, "_batch", [])
        published = save_training_checkpoint(
            self._ctx(),
            self.out / "checkpoints" / f"step-{state.optimizer_step:06d}",
            contract=self.contract,
            optimizer_step=state.optimizer_step,
            policy_version=f"student_onpolicy_sft:step{state.optimizer_step}",
            source_digest=package_digest(batch),
            rollout_adapter_digest=self.source_policy.digest,
            sampler={"cursor": 0, "round_id": state.round_id + 1, "row_ids": [row.row_id for row in batch]},
            selection_digest=str(result.payload.get("selection_digest") or ""),
            pending_round=state.round_id,
        )
        # Final adapter export is collective under FSDP2 (DTensor aggregation).
        export_policy_adapter(self._ctx(), self.out / "adapters" / "final")
        if self.rank == 0:
            manifest = read_json(published / "SFT_CHECKPOINT.json")
            append_metrics(
                self.out,
                _metric(
                    self.args,
                    state.optimizer_step,
                    state.round_id,
                    result.payload,
                    manifest,
                    self.source_policy.digest,
                    float(result.payload.get("update_wall_s") or 0.0),
                ),
            )
        self.acc.wait_for_everyone()

    def _ctx(self) -> Any:
        from types import SimpleNamespace

        from trim.training.sft_engine.backend import backend_name

        return SimpleNamespace(
            backend=backend_name(self.args),
            accelerator=self.acc,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            contract=self.contract,
        )

    def persist_loop_state(self, state: LoopState) -> None:
        path = self.out / "loop_state.json"
        if self.rank == 0:
            atomic_write_json(path, _state_dict(state))
        self._wait_value(path, "round_id", int(state.round_id))

    def summary(self, state: LoopState) -> dict[str, Any]:
        if self.rank == 0:
            atomic_write_json(self.out / "PERF_REPORT.json", summarize_progress(self.out))
        return {
            "experiment": "student_onpolicy_sft",
            "optimizer_step": state.optimizer_step,
            "round_id": state.round_id,
            "empty_rounds": state.empty_rounds,
            "online_layout": "split_pools",
        }

    def _export_policy(self, state: LoopState, staging: Path) -> None:
        import shutil

        from trim.training.sft_engine.backend import export_policy_adapter

        if self.rank == 0 and staging.exists():
            shutil.rmtree(staging)
        self.acc.wait_for_everyone()
        export_policy_adapter(self._ctx(), staging)

    def _publish(self, state: LoopState, staging: Path, pointer: Path, ack: Path) -> None:
        from trim.training.student_onpolicy_sft.rollout_service import RolloutPool

        if ack.is_file():
            ack.unlink()
        if not staging.is_dir() or not any(staging.iterdir()):
            raise RuntimeError("policy export produced no adapter files")
        digest = sha256_tree(staging)
        final = self.out / "policies" / f"step-{state.optimizer_step:06d}-{digest[:16]}"
        if final.exists():
            raise FileExistsError(f"policy directory already exists: {final}")
        staging.rename(final)
        version = f"student_onpolicy_sft:step{state.optimizer_step}:{digest[:16]}"
        atomic_write_json(
            pointer,
            {
                "path": str(final),
                "digest": digest,
                "version": version,
                "optimizer_step": int(state.optimizer_step),
            },
        )
        RolloutPool(self.args.rollout_endpoints).load_all(version, str(final), digest)
        atomic_write_json(ack, {"digest": digest, "version": version, "optimizer_step": int(state.optimizer_step)})

    def _label(self, points: list[Any], policy: Policy) -> tuple[list[SFTBatchRow], dict[str, Any]]:
        if not points:
            return [], {"n_rows": 0, "rejected": [], "collection_semantics": "student_states_for_ce"}
        from trim.training.sft_ablation.encoding import load_bound_encoding
        from trim.training.student_onpolicy_sft.targets import build_legal_ce_targets

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

    def _write_round(self, round_id: int, policy: Policy, rows: list[SFTBatchRow], audit: dict[str, Any]) -> None:
        directory = self.out / "rounds" / f"r{round_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            directory / "targets.json",
            {
                "policy": {"version": policy.version, "digest": policy.digest},
                "rows": [row.to_dict() for row in rows],
                "audit": audit,
                "query_ids": sorted({row.query_id for row in rows}),
            },
        )
        atomic_write_json(
            directory / "READY.json",
            {
                "ready": True,
                "n_rows": len(rows),
                "policy_digest": policy.digest,
                "rows_digest": package_digest(rows),
            },
        )

    def _queries_for_round(self, round_id: int) -> list[dict[str, Any]]:
        rng = random.Random(f"{int(self.args.seed)}:{int(round_id)}")
        order = list(self.queries)
        rng.shuffle(order)
        width = min(len(order), int(self.args.queries_per_round))
        chosen = order[:width]
        path = self.out / "rounds" / f"r{round_id:06d}" / "queries.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, chosen)
        return chosen

    def _current_round(self) -> int:
        if (self.out / "loop_state.json").is_file():
            return int(read_json(self.out / "loop_state.json")["round_id"])
        return 0

    def _guard(self, fn: Any) -> Any:
        try:
            return fn()
        except Exception as exc:
            atomic_write_json(self.fail_path, {"error": f"{type(exc).__name__}: {exc}"})
            raise

    def _wait(self, path: Path) -> None:
        self._wait_file(path)

    def _wait_file(self, path: Path) -> None:
        deadline = time.time() + 24 * 3600
        while not path.is_file():
            if self.fail_path.is_file():
                raise RuntimeError(str(read_json(self.fail_path).get("error") or "rank 0 failed"))
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for {path}")
            time.sleep(0.2)

    def _wait_value(self, path: Path, key: str, expected: int) -> None:
        deadline = time.time() + 24 * 3600
        while True:
            if path.is_file():
                try:
                    payload = read_json(path)
                except FileNotFoundError:
                    time.sleep(0.05)
                    continue
                if int(payload.get(key, -1)) == int(expected):
                    return
            if self.fail_path.is_file():
                raise RuntimeError(str(read_json(self.fail_path).get("error") or "rank 0 failed"))
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for {path} {key}={expected}")
            time.sleep(0.2)


def _ordered_round_rows(round_dir: Path, max_length: int) -> list[SFTBatchRow]:
    targets = read_json(round_dir / "targets.json")
    rows = [SFTBatchRow.from_dict(item) for item in targets.get("rows") or []]
    for row in rows:
        row.validate(max_length)
    return sorted(rows, key=lambda row: row.row_id)


def _round_id_from_path(path: Path) -> int:
    name = path.name
    if name.startswith("r") and name[1:].isdigit():
        return int(name[1:])
    return 0


def _student_contract(args: Any, accelerator: Any) -> dict[str, Any]:
    import os

    stable = fingerprint(
        {
            "queries_per_round": int(args.queries_per_round),
            "group_size": int(args.group_size),
            "max_turns": int(args.max_turns),
            "global_batch_size": int(args.global_batch_size),
            "seed": int(args.seed),
        }
    )
    from trim.training.sft_ablation.identity import tokenizer_digest

    init_digest = sha256_tree(Path(args.init_adapter)) if str(args.init_adapter or "") and Path(args.init_adapter).exists() else "missing"
    tokenizer = tokenizer_digest(str(args.model_name))
    if tokenizer in {"", "missing", "none"}:
        raise SystemExit("student v2 requires a model directory whose tokenizer files can be hashed")
    return build_v2_contract(
        args,
        dataset_digest="student_onpolicy_rounds_v1",
        source_contract_digest=stable,
        tokenizer_digest=tokenizer,
        gradient_ranks=int(accelerator.num_processes),
        nnodes=max(1, int(accelerator.num_processes) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE") or accelerator.num_processes))),
        nproc_per_node=int(os.environ.get("LOCAL_WORLD_SIZE") or accelerator.num_processes),
        init_adapter_digest=init_digest,
        backend=backend_name(args),
        adapter_config=adapter_config_from(args.init_adapter) if str(args.init_adapter or "") else None,
    )


def _write_contract(out: Path, contract: dict[str, Any], accelerator: Any) -> None:
    from trim.training.sft_engine.contract import require_exact_resume

    path = out / "CONTRACT_V2.json"
    if int(accelerator.process_index) == 0 and not path.is_file():
        atomic_write_json(path, contract)
    accelerator.wait_for_everyone()
    require_exact_resume(read_json(path), contract)


def _state_dict(state: LoopState) -> dict[str, int]:
    return {
        "optimizer_step": int(state.optimizer_step),
        "round_id": int(state.round_id),
        "empty_rounds": int(state.empty_rounds),
    }


def _metric(args: Any, step: int, round_id: int, result: dict[str, Any], manifest: dict[str, Any], rollout_digest: str, wall: float) -> dict[str, Any]:
    return {
        "experiment_id": "student_onpolicy_sft",
        "optimizer_step": int(step),
        "round_id": int(round_id),
        "policy_version": f"student_onpolicy_sft:step{int(step)}",
        "adapter_digest": manifest.get("adapter_digest"),
        "rollout_adapter_digest": rollout_digest,
        "latest_committed_adapter_digest": manifest.get("adapter_digest"),
        "source_digest": manifest.get("source_digest"),
        "selection_digest": manifest.get("selection_digest"),
        "n_rows": result.get("n_rows"),
        "weighted_tokens": result.get("weighted_tokens"),
        "loss_ce": result.get("loss_ce"),
        "grad_norm": result.get("grad_norm"),
        "learning_rate": float(args.learning_rate),
        "gradient_ranks": result.get("gradient_ranks"),
        "update_wall_s": wall,
        "input_tokens": result.get("input_tokens"),
        "supervised_tokens": result.get("supervised_tokens"),
        "dummy_rows": result.get("dummy_rows"),
    }
