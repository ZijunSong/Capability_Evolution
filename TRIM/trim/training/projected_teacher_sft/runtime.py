"""Projected Teacher SFT: Zero replay, then the same CE update as Teacher SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trim.adapters.components import zero_mask
from trim.training.dist_runtime import barrier
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
from trim.training.sft_ablation.teacher_archive import LazyArchive, seal_archive
from trim.training.teacher_sft.runtime import _train


def run(args: Any, dist: Any) -> int:
    from trim.training.sft_engine.backend import is_accelerated

    if is_accelerated(getattr(args, "ce_backend", "legacy")) and getattr(args, "stage", "") == "train":
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


def _load_archive(args: Any, dist: Any, queries: list[dict[str, Any]]) -> Any:
    del queries  # prepare consumes a published archive only; never re-collects
    if not str(args.teacher_traces or ""):
        raise FileNotFoundError(
            "Projected prepare requires an explicit published Teacher archive via --teacher-traces. "
            "It consumes collection output; it does not re-collect teacher trajectories."
        )
    path = Path(args.teacher_traces)
    archive_path = path / "archive.json" if path.is_dir() else path
    if not archive_path.is_file():
        raise FileNotFoundError(f"teacher archive not ready: {archive_path}")
    if archive_path.stat().st_size >= 1024 * 1024 * 1024:
        # verify_digest=False: skip the redundant full-rescan sha256 check (re-serializes
        # every trajectory via json.dumps to reproduce seal_archive's fingerprint). The
        # archive was already sealed+trusted by the Teacher SFT job (7rk9c); replay only
        # needs the byte-offset index (built regardless of this flag) to seek-and-read its
        # per-rank trajectories. Skipping the verify keeps replay data identical (get_item
        # reads raw bytes by offset, unaffected) while eliminating the 38GB full re-dump
        # that made evgbd stall ~35min in pure-Python _index.
        return LazyArchive(archive_path, verify_digest=False)
    payload = read_json(archive_path)
    return seal_archive(payload, contract_digest="external")


def _retrieval_config_from_args(args: Any) -> dict[str, Any] | None:
    """Map SFT CLI retrieval fields onto an explicit RetrievalConfig payload.

    Upstream replay refuses silent BrowseComp+ defaults: the corpus identity
    used at collection time must be written down and passed in (10.3/10.5).
    """
    if str(getattr(args, "train_env", "") or "") != "upstream":
        return None
    backend = str(getattr(args, "retrieval_backend", "") or "")
    if not backend:
        raise SystemExit(
            "upstream Projected replay requires --retrieval-backend (and --retrieval-dataset/"
            "--retrieval-collection-split/--retrieval-reranker) so the corpus identity is explicit"
        )
    config: dict[str, Any] = {"backend": backend}
    for key, field in (
        ("dataset", "retrieval_dataset"),
        ("collection_split", "retrieval_collection_split"),
        ("reranker", "retrieval_reranker"),
    ):
        value = str(getattr(args, field, "") or "")
        if not value:
            raise SystemExit(f"upstream Projected replay requires --{field.replace('_', '-')} (no default corpus guessing)")
        config[key] = value
    for key, field in (
        ("index_path", "bm25_index"),
        ("docstore_path", "docstore"),
        ("corpus_path", "corpus"),
    ):
        value = str(getattr(args, field, "") or "")
        if value:
            config[key] = value
    # Validate once in the controller too, so a bad config fails before workers.
    resolve_retrieval_config(config)
    return config


def _replay_all(args: Any, dist: Any, archive: Any, contract: dict[str, Any]) -> list[SFTBatchRow]:
    del contract
    from trim.training.sft_ablation.replicas import run_replica_jobs

    retrieval_config = _retrieval_config_from_args(args)
    if isinstance(archive, LazyArchive):
        replay_payload = {
            "archive_path": str(archive.path),
            "archive_ranges": [
                list(archive.offsets[index])
                for index in range(int(dist.rank), len(archive), max(1, int(dist.world_size)))
            ],
            "policy_version": str(archive.metadata.get("policy_version") or ""),
            "retrieval_config": retrieval_config,
        }
    else:
        trajectories = list(archive.get("trajectories") or [])
        owned = [item for i, item in enumerate(trajectories) if i % max(1, int(dist.world_size)) == int(dist.rank)]
        replay_payload = {
            "records": owned,
            "policy_version": str(archive.get("policy_version") or ""),
            "retrieval_config": retrieval_config,
        }

    n_traj = len(archive.offsets) if isinstance(archive, LazyArchive) else len(archive.get("trajectories") or [])
    print(f"REPLAY_START rank={int(dist.rank)} world={int(dist.world_size)} n_trajectories={n_traj} archive_type={'LazyArchive' if isinstance(archive, LazyArchive) else 'dict'}", flush=True)
    partials = run_replica_jobs(
        args,
        dist,
        task="replay",
        mask_name="zero",
        payload=replay_payload,
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


def replay_one(
    trajectory: dict[str, Any],
    *,
    encoder: Any,
    args: Any,
    contract: dict[str, Any],
    state_factory: Any = None,
    mask: dict[str, bool] | None = None,
) -> tuple[list[SFTBatchRow], list[dict[str, Any]]]:
    from trim.training.opd_projection import StudentActionSpaceProjector

    events = [_event(item) for item in trajectory.get("events") or []]
    if state_factory is None or mask is None:
        # Legacy call shape: build a one-off factory (opens its own session).
        state_factory, mask = build_replay_state_factory(
            train_env=str(args.train_env),
            retrieval_config=getattr(args, "retrieval_config", None),
            max_turns=int(args.max_turns),
        )
    env = job_env_for_trajectory(trajectory, encoder=encoder, state_factory=state_factory, mask=mask)
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


def build_replay_state_factory(*, train_env: str, retrieval_config: Any, max_turns: int) -> tuple[Any, dict[str, bool]]:
    """One tool session per worker; every trajectory still gets a fresh state.

    The retrieval configuration is explicit: SEC replay must not guess the
    corpus from ``RetrievalConfig()`` defaults (browsecompplus/test).
    """
    from trim.training.upstream_train_env import new_state_fn, open_upstream_train_session

    mask = zero_mask("Harness-1")
    session = None
    if train_env == "upstream":
        retrieval = resolve_retrieval_config(retrieval_config)
        session = open_upstream_train_session(retrieval=retrieval, harness_mask=mask, max_turns=int(max_turns))
    elif train_env != "local_legacy":
        raise ValueError(f"unknown train_env={train_env}")
    factory = new_state_fn(train_env=train_env, harness_mask=mask, session=session)
    return factory, mask


REQUIRED_RETRIEVAL_KEYS = ("backend", "dataset", "collection_split", "reranker")


def resolve_retrieval_config(config: Any) -> Any:
    """Strict RetrievalConfig construction: no silent defaults, no unknown keys."""
    from trim.upstream_harness1.retrieval import RETRIEVAL_LOCAL_BM25, RetrievalConfig

    if not isinstance(config, dict) or not config:
        raise ValueError(
            "upstream replay requires an explicit retrieval_config "
            f"with at least {list(REQUIRED_RETRIEVAL_KEYS)}; refusing RetrievalConfig() defaults"
        )
    missing = [key for key in REQUIRED_RETRIEVAL_KEYS if not str(config.get(key) or "")]
    if missing:
        raise ValueError(f"retrieval_config missing required keys: {missing}")
    known = {
        "backend",
        "dataset",
        "collection_split",
        "reranker",
        "index_path",
        "docstore_path",
        "corpus_path",
        "reranker_available",
        "reranker_base_url",
        "reranker_model",
    }
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"retrieval_config has unknown keys: {unknown}")
    if str(config.get("backend")) == RETRIEVAL_LOCAL_BM25:
        missing_paths = [key for key in ("index_path", "docstore_path", "corpus_path") if not str(config.get(key) or "")]
        if missing_paths:
            raise ValueError(f"local_bm25 retrieval_config missing {missing_paths}")
    retrieval = RetrievalConfig.from_mapping(config)
    retrieval.assert_ready()
    return retrieval


def job_env_for_trajectory(trajectory: dict[str, Any], *, encoder: Any, state_factory: Any, mask: dict[str, bool]) -> Any:
    """Open a fresh Zero episode from the shared worker session."""
    query = trajectory.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            f"replay requires the original nonempty query text for query_id={trajectory.get('query_id')!r}; "
            "query_id is not a substitute for the question"
        )
    query_id = str(trajectory["query_id"])
    # Each call creates a fresh episode. Only the tool session is shared.
    state = state_factory(query, {}, query_id=query_id)
    return LiveZeroEnv(state, query=query, query_id=query_id, enc=encoder, mask=mask)


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
            failure_class, retryable = _classify_failure(f"exception:{exc}")
            return ExecuteResult(ok=False, reason=f"exception:{exc}", failure_class=failure_class, retryable=retryable)
        failed, reason = _tool_failed(ok, obs)
        if failed:
            failure_class, retryable = _classify_failure(reason)
            return ExecuteResult(ok=False, reason=reason, failure_class=failure_class, retryable=retryable)
        self.st = state
        self.acts.append((action, obs))
        self._exec_count = int(getattr(self, "_exec_count", 0)) + 1
        if self._exec_count <= 3:
            print(f"REPLAY_EXECUTE n={self._exec_count} qid={self.query_id} action_name={action.get('name')} ok={ok} acts_len={len(self.acts)}", flush=True)
        self.ended = bool(state.get("ended")) or str(action.get("name")) == "end_search"
        post = self.decision().snapshot
        return ExecuteResult(ok=True, post_state_hash=post.content_hash(), snapshot=post, ended=self.ended)


INFRA_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "connect",
    "429",
    "rate limit",
    "ratelimit",
    "502",
    "503",
    "504",
    "5xx",
    "unavailable",
    "service",
    "temporarily",
    "reset by peer",
    "broken pipe",
)


def _classify_failure(reason: str) -> tuple[str, bool]:
    """Map an unstructured failure text to (failure_class, retryable).

    Real tools only return text; until the adapter layer raises typed errors,
    this is the boundary between infrastructure noise (retry the trajectory)
    and a semantic tool failure (keep the legal prefix).
    """
    lowered = str(reason or "").lower()
    if any(marker in lowered for marker in INFRA_MARKERS):
        return "infra", True
    return "semantic", False


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


def _read_range(handle: Any, start: int, end: int) -> bytes:
    handle.seek(int(start))
    return handle.read(int(end) - int(start))


def _load_query_map(archive_path: Path) -> dict[str, str]:
    """Build query_id -> query text from the sibling query_manifest.json.

    The Teacher archive (TEACHER_OUT/teacher_archive/archive.json) stores only
    query_id per trajectory; the original question text lives in the
    query_manifest.json that the Teacher SFT job wrote at TEACHER_OUT/query_manifest.json
    (i.e. the archive file's parent directory). Returns {} if no manifest is found
    (legacy archives whose trajectories already carry a "query" field need no map).
    """
    manifest_path = archive_path.parent / "query_manifest.json"
    if not manifest_path.is_file():
        # The Teacher job writes query_manifest.json at TEACHER_OUT root (archive is at
        # TEACHER_OUT/teacher_archive/archive.json), so look one level up too.
        manifest_path = archive_path.parent.parent / "query_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    queries = manifest.get("queries") or manifest.get("data") or manifest or []
    if isinstance(queries, dict):
        queries = queries.get("queries") or []
    out: dict[str, str] = {}
    for entry in queries or []:
        qid = str(entry.get("query_id") or entry.get("id") or "")
        text = str(entry.get("query") or entry.get("text") or "")
        if qid and text:
            out[qid] = text
    return out


def _inject_query_text(trajectory: dict[str, Any], query_map: dict[str, str]) -> None:
    """Set trajectory['query'] from the manifest map when the archive omitted it.

    job_env_for_trajectory reads trajectory.get('query'); the Teacher archive only has
    query_id, so look the text up by query_id. A trajectory already carrying a nonempty
    'query' is left untouched.
    """
    if isinstance(trajectory.get("query"), str) and trajectory["query"].strip():
        return
    qid = str(trajectory.get("query_id") or "")
    if qid in query_map:
        trajectory["query"] = query_map[qid]


def run_replay_job(job: dict[str, Any]) -> dict[str, Any]:
    from argparse import Namespace

    args = Namespace(
        **{key: job[key] for key in ("model_name", "max_model_len", "max_new_tokens", "max_turns", "train_env", "max_anchor_scan_events")}
    )
    from trim.training.sft_ablation.encoding import load_bound_encoding

    encoder = load_bound_encoding(args.model_name, max_model_len=int(args.max_model_len), max_new_tokens=int(args.max_new_tokens))
    contract = {"max_model_len": int(args.max_model_len), "max_new_tokens": int(args.max_new_tokens)}
    # One tool session per worker job; each trajectory still replays in a fresh
    # state/WM (job_env_for_trajectory). Retrieval identity is explicit.
    state_factory, mask = build_replay_state_factory(
        train_env=str(args.train_env),
        retrieval_config=job.get("retrieval_config"),
        max_turns=int(args.max_turns),
    )
    rows = []
    audits = []
    if job.get("archive_path"):
        ranges = [tuple(map(int, pair)) for pair in job.get("archive_ranges") or []]
        archive_path = Path(job["archive_path"])

        # The published Teacher archive stores only query_id per trajectory (no original
        # query text); job_env_for_trajectory needs the raw question to open a Zero env.
        # Load the query_id -> query map once from the sibling query_manifest.json that the
        # Teacher SFT job wrote next to the archive (TEACHER_OUT/query_manifest.json), then
        # inject the text into each trajectory before replay. Without this 7xq5g raised
        # ValueError "replay requires the original nonempty query text for query_id=...".
        query_map = _load_query_map(archive_path)

        def _read_trajectories():
            with archive_path.open("rb") as handle:
                for start, end in ranges:
                    trajectory = json.loads(_read_range(handle, start, end).decode("utf-8"))
                    _inject_query_text(trajectory, query_map)
                    yield trajectory

        trajectories = _read_trajectories()
    else:
        trajectories = job.get("trajectories") or []
    max_attempts = max(1, int(job.get("infra_retries") or 3))
    for traj_idx, trajectory in enumerate(trajectories):
        traj_id = str(trajectory.get("trajectory_id") or trajectory.get("query_id") or f"idx{traj_idx}")
        print(f"REPLAY_TRAJ_START idx={traj_idx} traj_id={traj_id} qid={trajectory.get('query_id')}", flush=True)
        produced, audit = _replay_with_infra_retries(
            trajectory,
            encoder=encoder,
            args=args,
            contract=contract,
            state_factory=state_factory,
            mask=mask,
            max_attempts=max_attempts,
        )
        print(f"REPLAY_TRAJ_DONE idx={traj_idx} traj_id={traj_id} n_rows={len(produced)}", flush=True)
        rows.extend(row.to_dict() for row in produced)
        audits.append(audit)
    return {
        "ok": True,
        "rows": rows,
        "audits": audits,
        "retrieval_identity": job.get("retrieval_config") or ({} if str(args.train_env) != "upstream" else None),
    }


def _replay_with_infra_retries(
    trajectory: dict[str, Any],
    *,
    encoder: Any,
    args: Any,
    contract: dict[str, Any],
    state_factory: Any,
    mask: dict[str, bool],
    max_attempts: int,
) -> tuple[list[SFTBatchRow], list[dict[str, Any]]]:
    """Infra failures retry the WHOLE trajectory from a fresh state and never
    commit a half-attempt; semantic stops keep their legal prefix (10.6)."""
    from trim.training.projected_teacher_sft.replay import RetryableReplayError

    last_reason = ""
    for attempt in range(1, max_attempts + 1):
        try:
            produced, audit = replay_one(
                trajectory,
                encoder=encoder,
                args=args,
                contract=contract,
                state_factory=state_factory,
                mask=mask,
            )
            audit = {"trajectory_id": str(trajectory.get("trajectory_id")), "attempt": attempt, "events": audit}
            return produced, audit
        except RetryableReplayError as exc:
            last_reason = str(exc)
    raise RuntimeError(
        f"infrastructure failure replaying trajectory {trajectory.get('trajectory_id')!r} "
        f"after {max_attempts} attempts: {last_reason}"
    )
