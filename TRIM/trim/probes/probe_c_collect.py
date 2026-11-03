"""Probe C real H0 state collection on the official train pool (Qwen3.5).

One collection, many components: Student rollouts under H0 (``zero`` mask,
frozen θ0) produce pre-action decision states shared by every candidate
coalition. Split assignment happens *before* sampling; states carry the exact
fitted prompt ids, native action ids, raw visible history and snapshot.

This is a real collection: no FakeSearchEnv, no synthetic states, no
query_id back-fill. Worker shards run one vLLM replica per GPU.

    # 1) resolve query pool + splits (CPU)
    python -m trim.probes.probe_c_collect prepare --model-path /tmp/qwen35-9b --out OUT
    # 2) per-GPU workers (entry loops ranks)
    python -m trim.probes.probe_c_collect worker --model-path /tmp/qwen35-9b --out OUT \
        --worker-rank R --workers 8 --max-turns 40 --max-new 2048
    # 3) strict validation + merge (CPU)
    python -m trim.probes.probe_c_collect merge --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_TRIM = Path(__file__).resolve().parents[2]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.probes.probe_c_contract import (  # noqa: E402
    SPLIT_AUDIT,
    SPLIT_SELECT,
    SPLIT_TASK_TEST,
    SPLIT_TRAIN,
    STATE_SOURCE_REAL,
)
from trim.probes.probe_c_data import (  # noqa: E402
    PROBE_C_STATE_SCHEMA_VERSION,
    ProbeCState,
    env_content_hash,
    split_audit,
    validate_state_row,
    write_split_audit,
    write_states_jsonl,
)
from trim.state.snapshot import stable_hash  # noqa: E402

COLLECTOR_VERSION = "probe_c_collect_v1"
SPLIT_ASSIGNMENT = "SPLIT_ASSIGNMENT.json"
STATES_RAW_DIR = "states_raw"
MAX_STATES_PER_QUERY = 4


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def model_identity(model_path: Path) -> dict[str, str]:
    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    archs = cfg.get("architectures") or []
    if "Qwen3_5ForConditionalGeneration" not in archs:
        raise RuntimeError(
            f"unsupported model loader: architectures={archs}; Qwen3.5 requires "
            "Qwen3_5ForConditionalGeneration (no silent fallback to another Qwen)"
        )
    tok_cfg = (model_path / "tokenizer_config.json")
    tok = (model_path / "tokenizer.json")
    return {
        "model_id": str(cfg.get("_name_or_path") or model_path),
        "architectures": ",".join(str(a) for a in archs),
        "config_hash": _sha16(model_path / "config.json"),
        "tokenizer_hash": _sha16(tok) if tok.is_file() else "",
        "chat_template_hash": stable_hash(
            json.loads(tok_cfg.read_text(encoding="utf-8")).get("chat_template") or ""
        )
        if tok_cfg.is_file()
        else "",
        "model_revision": stable_hash(
            {
                "config": _sha16(model_path / "config.json"),
                "tokenizer": _sha16(tok) if tok.is_file() else "",
            }
        )[:16],
    }


# ---------------------------------------------------------------------------
# query pool + split assignment (before sampling)
# ---------------------------------------------------------------------------


def load_train_pool(
    *,
    n_queries: int,
    train_data: str = "sec",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Official train pool only. Test-pool top-up is forbidden."""
    if train_data != "sec":
        raise RuntimeError(f"unsupported train_data={train_data!r} (sec only)")
    from trim.eval.official_query_pool import load_bcplus_830_full, overlap_ids
    from trim.eval.sec_corpus import (
        attach_sec_doc_stores,
        default_sec_corpus_root,
        default_sec_rl_data,
        load_sec_rl_queries,
    )

    rows, src = load_sec_rl_queries(
        default_sec_rl_data(),
        n_queries=n_queries or None,
        query_file=None,
        corpus_root=default_sec_corpus_root(),
    )
    src = dict(src)
    src["doc_store"] = attach_sec_doc_stores(
        rows, corpus_root=default_sec_corpus_root()
    )
    eval_rows, _eval_src = load_bcplus_830_full()
    overlap = overlap_ids(rows, eval_rows)
    if overlap:
        raise RuntimeError(
            f"train/eval query overlap ({len(overlap)} ids, e.g. {overlap[:5]}); "
            "Probe C states must not touch the official test pool"
        )
    meta = {
        "train_data": train_data,
        "n_train_pool": len(rows),
        "source": dict(src),
        "eval_overlap": 0,
        "note": "shortfall is reported as actual counts; no test-pool top-up",
    }
    return rows, meta


def assign_splits(
    query_ids: Sequence[str],
    *,
    n_train: int,
    n_select: int,
    n_audit: int,
    seed: int = 20261031,
) -> dict[str, str]:
    """Deterministic query-level split; groups never cross splits."""
    ids = sorted(set(str(q) for q in query_ids))
    order = sorted(ids, key=lambda q: (stable_hash(f"{seed}:{q}"), q))
    assignment: dict[str, str] = {}
    for i, qid in enumerate(order):
        if i < n_train:
            assignment[qid] = SPLIT_TRAIN
        elif i < n_train + n_select:
            assignment[qid] = SPLIT_SELECT
        elif i < n_train + n_select + n_audit:
            assignment[qid] = SPLIT_AUDIT
        else:
            assignment[qid] = ""  # pool remainder: not collected
    return assignment


# ---------------------------------------------------------------------------
# state serialization
# ---------------------------------------------------------------------------


def _raw_messages(point: Any, query: str) -> list[dict[str, Any]]:
    """Actual visible history: the decision-time message/tool record."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": str(query)}]
    meta = dict(point.pre_action_snapshot.metadata or {})
    acts = meta.get("_rollout_actions_obs") or []
    for item in acts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        action = item[0] if isinstance(item[0], dict) else {"name": str(item[0])}
        messages.append({"role": "assistant", "tool_call": dict(action)})
        messages.append({"role": "tool", "content": str(item[1])})
    return messages


def select_decision_points(points: Sequence[Any]) -> list[Any]:
    """≤4 decision points per episode: early / mid / pre-stop. Pre-action only."""
    pts = sorted(points, key=lambda p: int(p.turn_id))
    n = len(pts)
    if n <= MAX_STATES_PER_QUERY:
        return pts
    idxs = sorted({0, n // 3, (2 * n) // 3, n - 1})
    return [pts[i] for i in idxs]


def state_row_from_decision(
    point: Any,
    *,
    query: str,
    split: str,
    scenario_group_id: str,
    rollout_seed: int,
    identity: Mapping[str, str],
    rl_row: Mapping[str, Any] | None,
    max_model_len: int,
    max_new: int,
    tool_schema: Mapping[str, Any],
) -> dict[str, Any]:
    snap = point.pre_action_snapshot
    snap_dict = snap.to_dict()
    parse_result = {
        "structurally_valid": bool(point.structurally_valid),
        "executed_ok": point.executed_ok,
        "tool_names": list(point.action_tool_names or []),
        "action_text": str(point.student_action_text or ""),
    }
    from trim.adapters.components import zero_mask

    student_mask = zero_mask("Harness-1")
    return {
        "schema_version": PROBE_C_STATE_SCHEMA_VERSION,
        "source": STATE_SOURCE_REAL,
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "policy_checksum": stable_hash(
            {"theta0": True, "model_revision": identity["model_revision"]}
        )[:16],
        "adapter_checksum": "",
        "query_id": str(point.query_id),
        "query_text": str(query),
        "scenario_group_id": str(scenario_group_id),
        "split": str(split),
        "rollout_seed": int(rollout_seed),
        "turn_id": int(point.turn_id),
        "student_snapshot": snap_dict,
        "env_content_hash": env_content_hash(snap_dict),
        "snapshot_hash": snap.content_hash(),
        "env_restore_version": "local_legacy_train_env_v1",
        "raw_messages": _raw_messages(point, query),
        "effective_student_prompt_ids": [int(t) for t in point.student_prompt_token_ids],
        "effective_tool_schema": dict(tool_schema),
        "student_action_ids": [int(t) for t in point.student_action_tokens],
        "sampled_old_logprobs": [
            float(x) for x in ((rl_row or {}).get("token_logprobs") or [])
        ],
        "original_action_parse_result": parse_result,
        "student_mask": dict(student_mask),
        # Collection-time teacher mask is H0; per-candidate Teacher masks are
        # attached at cache time via HarnessPairSpec.
        "teacher_mask": dict(student_mask),
        "component_revision": stable_hash({"collector": COLLECTOR_VERSION})[:16],
        "information_provenance": [
            {
                "source": "student_rollout",
                "owner": "student_reduced",
                "policy_version": "theta0",
                "collector": COLLECTOR_VERSION,
            }
        ],
        "tokenizer_hash": identity["tokenizer_hash"],
        "chat_template_hash": identity["chat_template_hash"],
        "enable_thinking": False,
        "truncation_policy_hash": stable_hash(
            {
                "policy": "fit_prompt_ids_to_context",
                "max_model_len": int(max_model_len),
                "max_new": int(max_new),
            }
        ),
        "row_id": f"{point.query_id}:{point.rollout_idx}:{point.turn_id}",
    }


# ---------------------------------------------------------------------------
# prepare / worker / merge
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = model_identity(Path(args.model_path))
    rows, pool_meta = load_train_pool(
        n_queries=int(args.n_queries), train_data=str(args.train_data)
    )
    wanted = int(args.n_train) + int(args.n_select) + int(args.n_audit)
    assignment = assign_splits(
        [r["query_id"] for r in rows],
        n_train=int(args.n_train),
        n_select=int(args.n_select),
        n_audit=int(args.n_audit),
        seed=int(args.split_seed),
    )
    collected_ids = [q for q, s in assignment.items() if s]
    payload = {
        "collector_version": COLLECTOR_VERSION,
        "identity": identity,
        "assignment": assignment,
        "n_pool": len(rows),
        "n_assigned": len(collected_ids),
        "wanted": wanted,
        "shortfall": max(0, wanted - len(collected_ids)),
        "pool_meta": pool_meta,
        "max_states_per_query": MAX_STATES_PER_QUERY,
    }
    (out / SPLIT_ASSIGNMENT).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[probe_c_collect] prepare: pool={len(rows)} assigned={len(collected_ids)} "
        f"shortfall={payload['shortfall']} -> {out / SPLIT_ASSIGNMENT}"
    )
    return 0


def _load_assignment(out: Path) -> dict[str, Any]:
    path = out / SPLIT_ASSIGNMENT
    if not path.is_file():
        raise RuntimeError(f"missing {path}; run prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_worker(args: argparse.Namespace) -> int:
    import torch
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.eval.sec_corpus import (
        attach_sec_doc_stores,
        default_sec_corpus_root,
        default_sec_rl_data,
        load_sec_rl_queries,
    )
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.rl_opd_types import OPD_LOSS_PROJECTED_GAP
    from trim.training.vllm_hybrid import VLLMGenerateClient

    out = Path(args.out)
    payload = _load_assignment(out)
    assignment: dict[str, str] = payload["assignment"]
    identity = payload["identity"]
    rows, _src = load_sec_rl_queries(
        default_sec_rl_data(),
        n_queries=None,
        query_file=None,
        corpus_root=default_sec_corpus_root(),
    )
    attach_sec_doc_stores(rows, corpus_root=default_sec_corpus_root())
    # shard by query_id hash so worker count changes do not reshuffle queries
    shard = [
        r
        for r in rows
        if assignment.get(str(r["query_id"]))
        and int(stable_hash(str(r["query_id"]))[:8], 16) % int(args.workers)
        == int(args.worker_rank)
    ]
    print(
        f"[probe_c_collect] worker {args.worker_rank}/{args.workers}: "
        f"{len(shard)} queries",
        flush=True,
    )
    if not shard:
        (out / STATES_RAW_DIR).mkdir(parents=True, exist_ok=True)
        (out / STATES_RAW_DIR / f"shard_{args.worker_rank}.jsonl").write_text("")
        return 0

    enc = load_model_encoding(
        str(args.model_path), max_model_len=int(args.max_model_len)
    )
    searcher = None
    if str(args.searcher) == "sec":
        from trim.training.four_cell_runtime import open_train_retrieval

        searcher = open_train_retrieval(
            argparse.Namespace(train_data="sec", smoke=False, sec_corpus_root=None),
            train_rows=rows,
        )
    session_dir = out / "vllm_sessions" / f"collect_r{args.worker_rank}"
    client = VLLMGenerateClient(
        model_path=str(args.model_path),
        session_dir=session_dir,
        tensor_parallel_size=1,
        max_model_len=int(args.max_model_len),
        lora_path=None,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=True,
        startup_timeout_s=3600.0,
        generate_timeout_s=3600.0,
    )
    client.start()
    try:
        groups = rollout_queries_batched(
            client.generate_batch,
            shard,
            component_id="zero",
            group_size=int(args.group_size),
            max_turns=int(args.max_turns),
            max_new=int(args.max_new),
            policy_version="theta0",
            seed=int(args.seed),
            sample=bool(args.sample),
            enc=enc,
            searcher=searcher,
            teacher_mode=False,
            harness_mask=None,
            search_k=int(args.search_k),
            doc_store_k=int(args.doc_store_k),
            query_batch_size=int(args.query_batch_size or 0) or None,
            train_env=str(args.train_env),
            rollout_backend="vllm",
            collection_mode="rl+opd",
            opd_loss=OPD_LOSS_PROJECTED_GAP,
            rl_reward=None,
            enable_rl=False,
        )
    finally:
        client.close()

    from trim.training.tool_mask import legal_tool_names
    from trim.adapters.components import zero_mask

    tool_schema = {
        "tool_names": legal_tool_names(harness_mask=zero_mask("Harness-1")),
        "harness": "Harness-1",
        "mask": "zero",
    }
    seed_base = int(args.seed)
    rows_out: list[dict[str, Any]] = []
    for group in groups:
        qid = str(group.query_id)
        split = assignment.get(qid) or ""
        query = ""
        for r in shard:
            if str(r["query_id"]) == qid:
                query = str(r["query"])
                break
        rl_rows = list((group.trajectory_group or {}).get("rl_rows") or [])
        by_turn = {
            (str(rec.get("episode_id") or ""), int(rec.get("turn_id") or 0)): rec
            for rec in rl_rows
        }
        by_episode: dict[str, list[Any]] = {}
        for p in group.decision_points:
            by_episode.setdefault(str(p.episode_id), []).append(p)
        for episode_id, points in by_episode.items():
            for point in select_decision_points(points):
                rec = by_turn.get((episode_id, int(point.turn_id)))
                rows_out.append(
                    state_row_from_decision(
                        point,
                        query=query,
                        split=split,
                        scenario_group_id=qid,
                        rollout_seed=seed_base + int(point.rollout_idx),
                        identity=identity,
                        rl_row=rec,
                        max_model_len=int(args.max_model_len),
                        max_new=int(args.max_new),
                        tool_schema=tool_schema,
                    )
                )
    shard_dir = out / STATES_RAW_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{args.worker_rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"[probe_c_collect] worker {args.worker_rank}: wrote {len(rows_out)} states -> {path}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out)
    payload = _load_assignment(out)
    identity = payload["identity"]
    raw_rows: list[dict[str, Any]] = []
    shard_dir = out / STATES_RAW_DIR
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))
    states = [validate_state_row(r, row_index=i) for i, r in enumerate(raw_rows)]
    audit = split_audit(states)
    by_split: dict[str, list[ProbeCState]] = {}
    for st in states:
        by_split.setdefault(st.split, []).append(st)
    write_states_jsonl(states, out / "states" / "all_states.jsonl")
    for name, rows in by_split.items():
        write_states_jsonl(rows, out / "states" / f"{name}.jsonl")
    write_split_audit(states, out)
    manifest = {
        "collector_version": COLLECTOR_VERSION,
        "identity": identity,
        "n_states": len(states),
        "per_split": {
            name: {"n_states": len(rows), "n_queries": len({s.query_id for s in rows})}
            for name, rows in by_split.items()
        },
        "split_audit": audit,
        "max_states_per_query": MAX_STATES_PER_QUERY,
    }
    (out / "COLLECTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[probe_c_collect] merge: {len(states)} states; "
        f"per-split={manifest['per_split']}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe_c_collect")
    parser.add_argument("cmd", choices=["prepare", "worker", "merge"])
    parser.add_argument("--model-path", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-queries", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-select", type=int, default=64)
    parser.add_argument("--n-audit", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=20261031)
    parser.add_argument("--train-data", default="sec")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--search-k", type=int, default=10)
    parser.add_argument("--doc-store-k", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=0)
    parser.add_argument("--train-env", default="local_legacy")
    parser.add_argument("--searcher", choices=["sec", "none"], default="sec")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "worker":
        return cmd_worker(args)
    return cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
