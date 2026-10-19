"""Teacher SFT prepare and CE loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trim.training.dist_runtime import barrier, shard_for_rank
from trim.training.sft_ablation.contracts import SFTBatchRow, package_digest
from trim.training.sft_ablation.fixed_train import take_batch
from trim.training.sft_ablation.io_util import atomic_write_json, read_json
from trim.training.sft_ablation.session import append_metrics, prepare_directory, seal_contract
from trim.training.sft_ablation.teacher_archive import seal_archive
from trim.training.teacher_sft.dataset import build_teacher_sft_rows


def run(args: Any, dist: Any) -> int:
    queries, info = prepare_directory(args, dist)
    archive = _archive(args, dist, queries)
    digest = str(archive.get("digest") or sha256_tree(Path(args.teacher_traces)))
    contract = seal_contract(args, dist, query_info=info, archive_digest=digest)
    rows: list[SFTBatchRow] = []
    if args.stage in {"prepare", "all"}:
        rows, rejected = _rows(args, dist, archive, contract)
        if int(dist.rank) == 0:
            _write_rows(Path(args.out) / "sft_rows", rows, rejected)
    if args.stage in {"train", "all"}:
        if int(dist.rank) == 0 and not rows:
            rows = _read_rows(Path(args.out) / "sft_rows" / "rows.jsonl")
        _train(args, dist, rows, contract)
    barrier()
    return 0


def _archive(args: Any, dist: Any, queries: list[dict[str, Any]]) -> dict[str, Any]:
    out = Path(args.out) / "teacher_archive"
    merged = out / "archive.json"
    if str(args.teacher_traces or ""):
        path = Path(args.teacher_traces)
        payload = read_json(path / "archive.json") if path.is_dir() else read_json(path)
        return seal_archive(payload, contract_digest="external")
    if merged.is_file():
        return read_json(merged)
    if args.stage == "train":
        raise FileNotFoundError("Teacher SFT train requires a prepared archive or --teacher-traces")
    owned = shard_for_rank(queries, rank=int(dist.rank), world_size=int(dist.world_size))
    from trim.training.sft_ablation.replicas import run_replica_jobs

    partials = run_replica_jobs(
        args,
        dist,
        task="collect",
        mask_name="full",
        payload={"queries": owned, "policy_version": "teacher-init"},
    )
    local_rows = []
    for part in partials:
        local_rows.extend((part.get("archive") or {}).get("trajectories") or [])
    shard = Path(args.out) / "archive_shards" / str(int(dist.rank))
    shard.mkdir(parents=True, exist_ok=True)
    atomic_write_json(shard / "DONE.json", {"n": len(local_rows), "trajectories": local_rows})
    barrier()
    if int(dist.rank) == 0:
        trajectories = []
        for rank in range(int(dist.world_size)):
            part = read_json(Path(args.out) / "archive_shards" / str(rank) / "DONE.json")
            trajectories.extend(part["trajectories"])
        archive = seal_archive(
            {
                "schema_version": "full_teacher_archive_v1",
                "policy_version": "teacher-init",
                "mask": "all",
                "teacher_mode": False,
                "trajectories": trajectories,
            },
            contract_digest="generated",
        )
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(merged, archive)
    barrier()
    return read_json(merged)


def _rows(args: Any, dist: Any, archive: dict[str, Any], contract: dict[str, Any]) -> tuple[list[SFTBatchRow], list[dict[str, str]]]:
    from trim.training.sft_ablation.encoding import load_bound_encoding

    if int(dist.rank) != 0:
        return [], []
    encoder = load_bound_encoding(
        args.model_name,
        max_model_len=int(args.max_model_len),
        max_new_tokens=int(args.max_new_tokens),
    )
    return build_teacher_sft_rows(archive, encoder, contract)


def _write_rows(directory: Path, rows: list[SFTBatchRow], rejected: list[dict[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    (directory / "rows.jsonl").write_text(lines, encoding="utf-8")
    atomic_write_json(
        directory / "REJECTED.json",
        {"rejected": rejected, "digest": package_digest(rows), "n_rows": len(rows)},
    )


def _read_rows(path: Path) -> list[SFTBatchRow]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Teacher SFT rows: {path}")
    return [SFTBatchRow.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _train(args: Any, dist: Any, rows: list[SFTBatchRow], contract: dict[str, Any]) -> None:
    from trim.training.sft_ablation.checkpoint import latest_committed, load_committed, publish_checkpoint, write_latest
    from trim.training.sft_ablation.trainer import build_hf_trainer

    if int(dist.rank) != 0:
        barrier()
        return
    trainer = build_hf_trainer(args)
    root = Path(args.out) / "checkpoints"
    cursor = 0
    step = 0
    uses: dict[str, int] = {}
    identity = contract["immutable"]
    resume = latest_committed(root) if not args.resume else Path(args.resume)
    if resume is not None:
        manifest = load_committed(resume, model=trainer.model, optimizer=trainer.optimizer, scheduler=trainer.scheduler, identity=identity)
        step = int(manifest["optimizer_step"])
        cursor = int(manifest["sampler"]["cursor"])
        uses = dict(manifest["sampler"].get("uses") or {})
    while step < int(args.num_updates):
        batch, cursor, batch_uses = take_batch(rows, cursor=cursor, batch_size=int(args.batch_size), seed=int(args.seed))
        for key, count in batch_uses.items():
            uses[key] = uses.get(key, 0) + count
        result = trainer.update(batch)
        if not result.updated:
            raise RuntimeError("nonempty Teacher SFT batch produced no update")
        step += 1
        digest = trainer.adapter_digest()
        published = publish_checkpoint(
            root / f"step-{step:06d}",
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            identity=identity,
            optimizer_step=step,
            next_round_id=step,
            policy_version=f"teacher_sft:step{step}",
            adapter_digest=digest,
            source_digest=package_digest(rows),
            sampler={"cursor": cursor, "uses": uses},
        )
        write_latest(Path(args.out), published)
        append_metrics(
            Path(args.out),
            {
                "experiment_id": "teacher_sft",
                "optimizer_step": step,
                "round_id": step,
                "policy_version": f"teacher_sft:step{step}",
                "adapter_digest": digest,
                "n_rows": result.n_rows,
                "weighted_tokens": result.weighted_tokens,
                "loss_ce": result.loss_ce,
                "grad_norm": result.grad_norm,
                "learning_rate": float(args.learning_rate),
                "row_uses": sum(uses.values()),
            },
        )
    final = Path(args.out) / "adapters" / "final"
    final.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(final)
    barrier()
