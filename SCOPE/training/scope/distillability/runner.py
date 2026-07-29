"""E0 distillability rollout runner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from harness.agent import OpenAIAgentInferenceModel
from harness.capability.capability_id import CapabilityId, E0_PROBE_CAPABILITIES, parse_capability_id
from harness.harness_config import apply_harness_config, config_path, load_harness_config
from harness.llm_env import get_llm_client, get_llm_model_name
from training.chat_decision_driver import ChatDecisionDriver
from training.opd.browsecomp_queries import load_browsecomp_full_queries
from training.opd.env_factory import build_rollout_runtime
from training.opd.harness_rollout import check_retrieval_backend, load_completed_query_ids
from training.opd.rollout_worker import QueryRecord, load_query_records_from_json
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
from training.scope.distillability.metrics import (
    GLOBAL_METRICS,
    aggregate_episodes,
    capability_specific_metrics,
    enrich_episode_metrics,
    episodes_by_query,
)
from training.scope.distillability.modes import DistillabilityMode
from training.scope.distillability.proc_injector import ProcInjector
from training.scope.distillability.registry import (
    apply_probe_env,
    clear_active_probe,
    get_probe_spec,
    set_capability_mode,
)
from training.scope.distillability.schema import E0RunManifest, ProcAuditStats
from training.train_rl import MAX_TURNS, SlidingWindowSearchEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCOPE E0 Capability Distillability rollout")
    p.add_argument("--capability", required=True, help="Capability id to probe")
    p.add_argument(
        "--mode",
        required=True,
        choices=[m.value for m in DistillabilityMode],
    )
    p.add_argument("--output-dir", default="outputs/scope_e0_distillability")
    p.add_argument(
        "--queries-json",
        default="artifacts/datasets/e0_audit_100q/query_ids.json",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument(
        "--model-path",
        default="/data/ppnm/models/Qwen2.5-7B-Instruct",
    )
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--vllm-port", type=int, default=8776)
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--manage-vllm", action="store_true", default=True)
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--vllm-model-name", default="e0-harness-policy")
    p.add_argument("--bm25-index-path", default=None)
    p.add_argument("--retrieval", default="bm25", choices=["bm25"])
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--query-timeout-s", type=int, default=900)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--reuse-full-from",
        default="outputs/harness_rollout_browsecomp_full_v2/harness_rollouts.jsonl",
        help="Reuse FULL mode from existing Phase-0 rollout if manifest matches",
    )
    return p.parse_args()


def load_successful_query_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("error"):
                    done.add(str(row["query_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _sample_key(qid: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{qid}".encode()).hexdigest()


def load_e0_queries(args: argparse.Namespace) -> list[QueryRecord]:
    qpath = Path(args.queries_json)
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8"))
        qids = [str(x) for x in data.get("query_ids", data)]
        all_records = {
            r.query_id: r
            for r in load_browsecomp_full_queries(split="all", limit=0, download_if_missing=False)
        }
        records = [all_records[qid] for qid in qids if qid in all_records]
        if args.limit > 0:
            records = records[: args.limit]
        if records:
            return records
    records = load_browsecomp_full_queries(split="all", limit=0, download_if_missing=False)
    records = sorted(records, key=lambda r: _sample_key(r.query_id, args.seed))
    if args.limit > 0:
        records = records[: args.limit]
    return records


def export_query_ids(path: Path, records: list[QueryRecord], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n_queries": len(records),
        "query_ids": [r.query_id for r in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _manifest_matches(
  *,
  manifest: dict[str, Any],
  args: argparse.Namespace,
  query_ids: list[str],
  config_hash: str,
) -> bool:
    checks = {
        "model_path": args.model_path,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            return False
    if "modules_full_v2" not in str(manifest.get("harness_config", "")):
        return False
    if manifest.get("retrieval") not in {"bm25", None}:
        return False
    # Query set: full 830 contains our 100q subset
    return True


def reuse_full_mode(
    *,
    args: argparse.Namespace,
    capability: CapabilityId,
    query_ids: list[str],
    out_dir: Path,
) -> bool:
    src = Path(args.reuse_full_from)
    manifest_path = src.parent / "harness_rollout_manifest.json"
    if not src.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full_cfg = load_harness_config(config_path("modules_full_v2.yaml"))
    if not _manifest_matches(
        manifest=manifest,
        args=args,
        query_ids=query_ids,
        config_hash=full_cfg.config_hash(),
    ):
        print("[e0] FULL reuse rejected: manifest mismatch", flush=True)
        return False

    wanted = set(query_ids)
    episodes: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("query_id")) in wanted:
                episodes.append(enrich_episode_metrics(capability.value, row))

    if len(episodes) < len(query_ids):
        print(
            f"[e0] FULL reuse rejected: only {len(episodes)}/{len(query_ids)} queries found",
            flush=True,
        )
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for ep in sorted(episodes, key=lambda x: str(x.get("query_id"))):
            fh.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")

    summary = aggregate_episodes(episodes)
    summary["mode"] = "full_reused"
    summary["n_queries"] = len(episodes)
    summary["capability_id"] = capability.value
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_manifest = E0RunManifest(
        capability_id=capability,
        mode=DistillabilityMode.FULL,
        model_path=args.model_path,
        harness_config=str(config_path("modules_full_v2.yaml")),
        config_hash=full_cfg.config_hash(),
        query_ids=query_ids,
        seed=args.seed,
        bm25_index_path=str(manifest.get("bm25_index_path", "")),
        max_turns=args.max_turns,
        temperature=args.temperature,
        parallel=args.parallel,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                **run_manifest.to_dict(),
                "reused_from": str(src),
                "reuse_manifest": str(manifest_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    full_cfg.save_resolved(out_dir / "resolved_config.yaml")
    (out_dir / "errors.jsonl").write_text("", encoding="utf-8")
    print(f"[e0] FULL reused {len(episodes)} episodes from {src}", flush=True)
    return True


async def run_rollout_async(args: argparse.Namespace) -> None:
    capability = parse_capability_id(args.capability)
    mode = DistillabilityMode(args.mode)
    spec = get_probe_spec(capability)
    out_dir = Path(args.output_dir) / capability.value / mode.value
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_e0_queries(args)
    qpath = Path(args.queries_json)
    if not qpath.exists():
        export_query_ids(qpath, records, args.seed)
    query_ids = [r.query_id for r in records]

    if mode == DistillabilityMode.FULL:
        if reuse_full_mode(args=args, capability=capability, query_ids=query_ids, out_dir=out_dir):
            return

    if mode == DistillabilityMode.PROC and not spec.proc_supported:
        raise SystemExit(f"PROC not supported for {capability.value}")

    harness_cfg, env_overrides = set_capability_mode(capability, mode)
    apply_harness_config(harness_cfg)
    apply_probe_env(env_overrides)
    harness_cfg.save_resolved(out_dir / "resolved_config.yaml")

    index_path = check_retrieval_backend(
        args.retrieval, bm25_index_path=args.bm25_index_path, smoke=False
    )
    runtime = build_rollout_runtime(
        "browsecompplus",
        collection_split="test",
        reranker="none",
        retrieval=args.retrieval,
        bm25_index_path=index_path,
    )

    episodes_path = out_dir / "episodes.jsonl"
    events_path = out_dir / "events.jsonl"
    errors_path = out_dir / "errors.jsonl"
    if not args.resume:
        for p in (episodes_path, events_path, errors_path):
            p.write_text("", encoding="utf-8")
    else:
        # Drop failed rows so resume can retry them.
        if episodes_path.exists():
            kept = []
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("error"):
                    kept.append(line)
            episodes_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        done = load_successful_query_ids(episodes_path)

    done = load_successful_query_ids(episodes_path) if args.resume else set()
    pending = [r for r in records if r.query_id not in done]

    run_manifest = E0RunManifest(
        capability_id=capability,
        mode=mode,
        model_path=args.model_path,
        harness_config=str(config_path("modules_full_v2.yaml")),
        config_hash=harness_cfg.config_hash(),
        query_ids=query_ids,
        seed=args.seed,
        bm25_index_path=str(index_path),
        max_turns=args.max_turns,
        temperature=args.temperature,
        parallel=args.parallel,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(run_manifest.to_dict(), indent=2), encoding="utf-8"
    )

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
    os.environ["base_url"] = base_url
    os.environ["api_key"] = "EMPTY"
    os.environ["model_name"] = args.vllm_model_name
    # Clear pydantic settings cache so local vLLM overrides .env API defaults.
    from harness.llm_env import get_llm_settings

    get_llm_settings.cache_clear()

    proc_audit = ProcAuditStats()
    proc_audits: list[ProcAuditStats] = []
    all_events: list[dict[str, Any]] = []

    try:
        if args.manage_vllm and args.vllm_url is None:
            vllm_handle = start_vllm_server(
                model_path=args.model_path,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                served_model_name=args.vllm_model_name,
                log_path=str(out_dir / "vllm_server.log"),
            )
            base_url = vllm_handle.base_url
            os.environ["base_url"] = base_url
            get_llm_settings.cache_clear()

        client = get_llm_client()
        model_name = get_llm_model_name()
        inference = OpenAIAgentInferenceModel(
            openai_client=client,
            model=model_name,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            api_style="chat_completions",
        )

        sem = asyncio.Semaphore(args.parallel)
        write_lock = asyncio.Lock()
        completed = len(records) - len(pending)

        async def _one(record: QueryRecord) -> None:
            nonlocal completed
            ep_events: list[dict[str, Any]] = []
            try:
                async with sem:
                    env = SlidingWindowSearchEnv(
                        toolset=runtime.toolset,
                        search_tool=runtime.search_tool,
                        query_id=record.query_id,
                        query_text=record.query,
                        dataset=runtime.dataset,
                        text_token_counter=runtime.text_token_counter,
                        max_turns=args.max_turns,
                    )
                    driver_kwargs: dict[str, Any] = {
                        "env": env,
                        "inference": inference,
                        "max_turns": args.max_turns,
                    }
                    if mode == DistillabilityMode.OFF and capability == CapabilityId.STOP_DECISION:
                        driver_kwargs["min_turns_before_end"] = 0
                        driver_kwargs["min_curated_before_end"] = 0
                    elif mode == DistillabilityMode.FULL:
                        driver_kwargs["min_turns_before_end"] = 8
                        driver_kwargs["min_curated_before_end"] = 1

                    driver = ChatDecisionDriver(**driver_kwargs)
                    pre_hook = None
                    ep_audit = ProcAuditStats()
                    if mode == DistillabilityMode.PROC:
                        injector = ProcInjector(
                            capability_id=capability,
                            env=env,
                            audit=ep_audit,
                        )

                        def pre_hook(state, action):
                            return injector.maybe_inject(state, action)

                    result = await asyncio.wait_for(
                        driver.run(pre_step_hook=pre_hook),
                        timeout=float(args.query_timeout_s),
                    )
            except asyncio.TimeoutError:
                result = {
                    "query_id": record.query_id,
                    "error": True,
                    "turns": 0,
                    "recall": 0.0,
                }
                ep_events.append(
                    {"kind": "timeout", "query_id": record.query_id}
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "query_id": record.query_id,
                    "error": True,
                    "turns": 0,
                    "recall": 0.0,
                    "error_message": str(exc)[:500],
                }
                ep_events.append(
                    {"kind": "error", "query_id": record.query_id, "error": str(exc)[:500]}
                )

            ep = enrich_episode_metrics(capability.value, result)
            ep["query_id"] = record.query_id
            ep["capability_id"] = capability.value
            ep["mode"] = mode.value
            if mode == DistillabilityMode.PROC:
                proc_audits.append(ep_audit)

            async with write_lock:
                completed += 1
                with episodes_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
                if ep_events:
                    with events_path.open("a", encoding="utf-8") as fh:
                        for ev in ep_events:
                            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    all_events.extend(ep_events)
                if args.log_every and completed % args.log_every == 0:
                    print(
                        f"[e0] {capability.value}/{mode.value} "
                        f"{completed}/{len(records)} last={record.query_id} "
                        f"recall={ep.get('recall', 0):.3f}",
                        flush=True,
                    )

        await asyncio.gather(*[_one(r) for r in pending])

        if proc_audits:
            proc_audit = ProcAuditStats(
                visibility_violation_rate=sum(a.visibility_violation_rate for a in proc_audits)
                / len(proc_audits),
                new_observation_from_proc=sum(a.new_observation_from_proc for a in proc_audits),
                external_call_from_proc=sum(a.external_call_from_proc for a in proc_audits),
                hidden_field_access=sum(a.hidden_field_access for a in proc_audits),
                state_mutation_rate=sum(a.state_mutation_rate for a in proc_audits),
                n_proc_interventions=sum(a.n_proc_interventions for a in proc_audits),
                n_shadow_calls=sum(a.n_shadow_calls for a in proc_audits),
            )

        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = aggregate_episodes(episodes)
        summary.update(
            {
                "capability_id": capability.value,
                "mode": mode.value,
                "n_queries": len(episodes),
                "proc_audit": proc_audit.to_dict() if mode == DistillabilityMode.PROC else {},
                "capability_metrics": capability_specific_metrics(
                    capability.value, episodes, all_events
                ),
            }
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        clear_active_probe()
        if vllm_handle is not None:
            vllm_handle.stop()


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
    asyncio.run(run_rollout_async(parse_args()))


if __name__ == "__main__":
    main()
