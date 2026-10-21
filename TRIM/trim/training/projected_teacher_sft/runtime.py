"""Projected Teacher SFT: Zero replay, then the same CE update as Teacher SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trim.adapters.components import zero_mask
from trim.training.dist_runtime import barrier, shard_for_rank
from trim.training.opd_events import HarnessEvent
from trim.training.projected_teacher_sft.replay import (
    ReplayDecision,
    make_projector_adapter,
    replay_projected_trace,
    row_from_projected_step,
)
from trim.training.sft_ablation.contracts import SFTBatchRow, package_digest
from trim.training.sft_ablation.io_util import atomic_write_json, read_json
from trim.training.sft_ablation.session import prepare_directory, seal_contract
from trim.training.sft_ablation.teacher_archive import seal_archive
from trim.training.teacher_sft.runtime import _train


def run(args: Any, dist: Any) -> int:
    if getattr(args, "ce_backend", "legacy") == "accelerate_ddp" and getattr(args, "stage", "") == "train":
        from trim.training.sft_engine.fixed_runtime import run_fixed_sft

        if str(args.experiment) != "projected_teacher_sft":
            raise SystemExit(f"projected entry got experiment {args.experiment}")
        return run_fixed_sft(args)
    queries, info = prepare_directory(args, dist)
    archive = _load_archive(args, dist, queries)
    contract = seal_contract(args, dist, query_info=info, archive_digest=str(archive["digest"]))
    rows: list[SFTBatchRow] = []
    if args.stage in {"prepare", "all"}:
        rows = _replay_all(args, dist, archive, contract)
    if args.stage in {"train", "all"}:
        if int(dist.rank) == 0 and not rows:
            path = Path(args.out) / "sft_rows" / "rows.jsonl"
            rows = [
                SFTBatchRow.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        _train(args, dist, rows, contract)
        if int(dist.rank) == 0:
            _retarget_metrics(Path(args.out))
    barrier()
    return 0


def _retarget_metrics(out: Path) -> None:
    path = out / "METRICS.jsonl"
    if not path.is_file():
        return
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["experiment_id"] = "projected_teacher_sft"
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_archive(args: Any, dist: Any, queries: list[dict[str, Any]]) -> dict[str, Any]:
    if str(args.teacher_traces or ""):
        path = Path(args.teacher_traces)
        payload = read_json(path / "archive.json") if path.is_dir() else read_json(path)
        return seal_archive(payload, contract_digest="external")
    from trim.training.teacher_sft.runtime import _archive

    return _archive(args, dist, queries)


def _replay_all(args: Any, dist: Any, archive: dict[str, Any], contract: dict[str, Any]) -> list[SFTBatchRow]:
    del contract
    trajectories = list(archive.get("trajectories") or [])
    owned = shard_for_rank(trajectories, rank=int(dist.rank), world_size=int(dist.world_size))
    from trim.training.sft_ablation.replicas import run_replica_jobs

    partials = run_replica_jobs(
        args,
        dist,
        task="replay",
        mask_name="zero",
        payload={"records": owned, "policy_version": str(archive.get("policy_version") or "")},
    )
    rows = []
    audits = []
    for part in partials:
        rows.extend(SFTBatchRow.from_dict(item) for item in part.get("rows") or [])
        audits.extend(part.get("audits") or [])
    shard = Path(args.out) / "projection_audit" / str(int(dist.rank))
    shard.mkdir(parents=True, exist_ok=True)
    atomic_write_json(shard / "DONE.json", {"rows": [row.to_dict() for row in rows], "audits": audits})
    barrier()
    if int(dist.rank) != 0:
        return []
    merged: list[SFTBatchRow] = []
    all_audits = []
    for rank in range(int(dist.world_size)):
        part = read_json(Path(args.out) / "projection_audit" / str(rank) / "DONE.json")
        merged.extend(SFTBatchRow.from_dict(item) for item in part["rows"])
        all_audits.extend(part["audits"])
    ids = [row.row_id for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate projected row_id")
    merged.sort(key=lambda row: (row.query_id, row.episode_id, row.decision_id, row.row_id))
    directory = Path(args.out) / "sft_rows"
    directory.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in merged)
    (directory / "rows.jsonl").write_text(text, encoding="utf-8")
    atomic_write_json(
        Path(args.out) / "projection_audit" / "SUMMARY.json",
        {"audits": all_audits, "n_rows": len(merged), "digest": package_digest(merged)},
    )
    return merged


def replay_one(trajectory: dict[str, Any], *, encoder: Any, args: Any, contract: dict[str, Any]) -> tuple[list[SFTBatchRow], list[dict[str, Any]]]:
    from trim.training.opd_projection import StudentActionSpaceProjector

    events = [_event(item) for item in trajectory.get("events") or []]
    env = job_env_for_trajectory(trajectory, encoder=encoder, args=args)
    projector = StudentActionSpaceProjector()
    choose = make_projector_adapter(projector, audit=None)
    counter = {"n": 0}

    def make_row(before: ReplayDecision, step: Any, projection: Any) -> SFTBatchRow:
        counter["n"] += 1
        return row_from_projected_step(
            before,
            step,
            projection,
            enc=encoder,
            query_id=str(trajectory["query_id"]),
            episode_id=str(trajectory["trajectory_id"]),
            policy_version=str(trajectory["policy_version"]),
            max_model_len=int(contract["max_model_len"]),
            max_new_tokens=int(contract["max_new_tokens"]),
            row_id=f"projected_teacher_sft:{trajectory['trajectory_id']}:{projection.anchor_event_id}:{counter['n']}",
        )

    return replay_projected_trace(
        events,
        env,
        choose,
        make_row,
        scan=int(args.max_anchor_scan_events),
        max_steps=int(args.max_turns),
    )


def job_env_for_trajectory(trajectory: dict[str, Any], *, encoder: Any, args: Any) -> Any:
    """Open a fresh Zero environment. Call this only inside a Zero worker."""
    from trim.training.upstream_train_env import apply_train_action, new_state_fn, open_upstream_train_session

    mask = zero_mask("Harness-1")
    session = open_upstream_train_session(harness_mask=mask, max_turns=int(args.max_turns)) if args.train_env == "upstream" else None
    factory = new_state_fn(train_env=args.train_env, harness_mask=mask, session=session)
    query = str(trajectory.get("query") or trajectory["query_id"])
    state = factory(query, {}, query_id=str(trajectory["query_id"]))
    return LiveZeroEnv(state, query=query, query_id=str(trajectory["query_id"]), enc=encoder, mask=mask)


class LiveZeroEnv:
    def __init__(self, state: dict[str, Any], *, query: str, query_id: str, enc: Any, mask: dict[str, bool]) -> None:
        self.st = state
        self.row = {"query": query, "query_id": query_id}
        self.harness_mask = dict(mask)
        self.component_id = "zero"
        self.acts: list[Any] = []
        self.reasoning_effort = None
        self.enc = enc
        self.ended = bool(state.get("ended"))

    def decision(self) -> ReplayDecision:
        from trim.training.action_encoding import prompt_visible_doc_ids_from_prompt, visible_doc_ids_from_snapshot
        from trim.training.batched_env_rollout import _build_prompt_ids
        from trim.training.four_cell_runtime import snap_from_state

        prompt_ids = _build_prompt_ids(self, self.enc)
        snapshot = snap_from_state(self.row["query_id"], self.st, "zero", harness_mask=self.harness_mask)
        accessible = visible_doc_ids_from_snapshot(snapshot)
        visible = prompt_visible_doc_ids_from_prompt(prompt_ids=prompt_ids, accessible_ids=accessible, enc=self.enc)
        return ReplayDecision(
            snapshot=snapshot,
            prompt_ids=list(prompt_ids),
            metadata={"prompt_visible_doc_ids": visible, "accessible_doc_ids": accessible},
        )

    def execute(self, action: dict[str, Any]) -> Any:
        from trim.training.projected_teacher_sft.replay import ExecuteResult
        from trim.training.upstream_train_env import apply_train_action

        try:
            state, obs, ok = apply_train_action(self.st, action, True)
        except Exception as exc:
            return ExecuteResult(ok=False, reason=f"exception:{exc}")
        failed, reason = _tool_failed(ok, obs)
        if failed:
            return ExecuteResult(ok=False, reason=reason)
        self.st = state
        self.acts.append((action, obs))
        self.ended = bool(state.get("ended")) or str(action.get("name")) == "end_search"
        post = self.decision().snapshot
        return ExecuteResult(ok=True, post_state_hash=post.content_hash(), snapshot=post, ended=self.ended)


def _tool_failed(ok: bool, obs: Any) -> tuple[bool, str]:
    if not ok:
        return True, "exec_not_ok"
    text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)
    lowered = text.lower()
    if '"error"' in lowered or lowered.startswith("error"):
        return True, "tool_error_text"
    return False, ""


def _event(item: dict[str, Any]) -> HarnessEvent:
    payload = dict(item)
    payload.setdefault("component_id", payload.get("component"))
    payload.setdefault("kind", payload.get("kind") or "model_action")
    return HarnessEvent.from_dict(payload)


def run_replay_job(job: dict[str, Any]) -> dict[str, Any]:
    from argparse import Namespace

    args = Namespace(**{key: job[key] for key in ("model_name", "max_model_len", "max_new_tokens", "max_turns", "train_env", "max_anchor_scan_events")})
    from trim.training.sft_ablation.encoding import load_bound_encoding

    encoder = load_bound_encoding(args.model_name, max_model_len=int(args.max_model_len), max_new_tokens=int(args.max_new_tokens))
    contract = {"max_model_len": int(args.max_model_len), "max_new_tokens": int(args.max_new_tokens)}
    rows = []
    audits = []
    for trajectory in job.get("trajectories") or []:
        produced, audit = replay_one(trajectory, encoder=encoder, args=args, contract=contract)
        rows.extend(row.to_dict() for row in produced)
        audits.append(audit)
    return {"ok": True, "rows": rows, "audits": audits}
