#!/usr/bin/env python3
"""One-click Harness-1 / Harness-G closed-loop eval.

Harness-1 official path (``--evaluation-path upstream_api``): original
``SlidingWindowSearchEnv`` plus an OpenAI-compatible chat/completions adapter.
``--component`` is the actual v8d mask (all=10/10, default=8/10, zero=0/10,
or an exact list). ``--run-dir`` / ``--adapter`` / ``--eval-mode`` identify
weights only and never invert that mask.

``legacy_local`` keeps the old TRIM in-process env for historical replay. It
is never selected automatically when the official path fails.

Harness-G still uses its graph runtime.

Pass ``--api-base-url`` and ``--api-model`` for the served actor. Default retrieval
is ``--retrieval-backend upstream`` (original Chroma tools). Pass
``--retrieval-backend local_bm25`` with ``--index-path`` and a full-text corpus to
keep the original env while dropping Chroma. Do not use ``legacy_local``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.offline_credentials import ensure_local_offline_credentials

ensure_local_offline_credentials()

from trim.cli.launch import (
    LaunchError,
    discover_adapter_map,
    eval_mask_for_ids,
    parse_eval_args,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from trim.upstream_harness1.v8d_flags import (
    EVALUATION_PATH_LEGACY_LOCAL,
    EVALUATION_PATH_UPSTREAM_API,
)
from trim.eval.adapter_reload_audit import audit_saved_adapter
from trim.eval.official_query_pool import (
    SCORE_SPLIT_830,
    canonical_score_split,
    score_split_for_benchmark,
)
from trim.eval.sr_opd_four_cell_eval import write_eval_outputs
from trim.eval.transfer_benchmarks import load_eval_benchmark, score_split_for_eval_benchmark


def _adapter_map(args) -> dict[str, str | None]:
    if args.adapter_map:
        payload = json.loads(Path(args.adapter_map).read_text(encoding="utf-8"))
        return {str(k): (str(v) if v else None) for k, v in payload.items()}
    if args.adapter:
        return {"eval": str(args.adapter)}
    if args.run_dir:
        found = discover_adapter_map(args.run_dir)
        if found:
            return found
    return {}


def resolve_eval_mode(args) -> tuple[str, dict[str, str | None]]:
    """Model-artifact discovery only. Does not choose a component mask."""
    mapping = _adapter_map(args)
    mode = args.eval_mode
    if mode == "auto":
        mode = "adapter" if any(mapping.values()) else "harness"
    if mode == "harness":
        return "harness", mapping or {"harness": None}
    return "adapter", mapping or {"before": None}


def resolve_evaluation_path(args, spec) -> str:
    explicit = getattr(args, "evaluation_path", None)
    if explicit:
        return str(explicit)
    from trim.adapters.harness_profiles import is_harness_g

    if is_harness_g(harness=spec.harness, component_ids=spec.components):
        return EVALUATION_PATH_LEGACY_LOCAL
    return EVALUATION_PATH_UPSTREAM_API


def resolve_eval_mask(args, spec, *, evaluation_path: str) -> dict[str, bool]:
    """Official eval: --component is the live mask. Training complement is not used."""
    if evaluation_path == EVALUATION_PATH_UPSTREAM_API:
        return eval_mask_for_ids(
            spec.components, harness=spec.harness, preset=spec.component_preset
        )
    mode, _mapping = resolve_eval_mode(args)
    if mode == "harness":
        return teacher_mask_for_ids(
            spec.components, harness=spec.harness, preset=spec.component_preset
        )
    return student_mask_for_ids(spec.components, harness=spec.harness)


def detect_score_split(args) -> str:
    explicit = getattr(args, "score_split", None)
    if explicit:
        return canonical_score_split(str(explicit), default=SCORE_SPLIT_830) or SCORE_SPLIT_830
    implied = score_split_for_eval_benchmark(str(getattr(args, "benchmark", "") or "")) or score_split_for_benchmark(
        str(getattr(args, "benchmark", "") or "")
    )
    if implied:
        return implied
    return SCORE_SPLIT_830


def main(argv: list[str] | None = None) -> int:
    try:
        args, spec = parse_eval_args(argv)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc

    mode, adapter_map = resolve_eval_mode(args)
    score_split = detect_score_split(args)
    args.score_split = score_split
    evaluation_path = resolve_evaluation_path(args, spec)
    args.evaluation_path = evaluation_path
    harness_mask = resolve_eval_mask(args, spec, evaluation_path=evaluation_path)

    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    held_outer = False
    if not args.audit_only and int(getattr(args, "eval_replicas", 1)) <= 1:
        acquire_keepalive()
        held_outer = True

    spec.out.mkdir(parents=True, exist_ok=True)
    from trim.adapters.harness_profiles import is_harness_g
    from trim.eval.runtime_effect_audit import audit_harness_mask_or_raise

    wiring_audit = None
    if evaluation_path == EVALUATION_PATH_LEGACY_LOCAL and not is_harness_g(
        mask=harness_mask, component_ids=spec.coalition
    ):
        wiring_audit = audit_harness_mask_or_raise(harness_mask, out=spec.out)
    launch = {
        "harness": spec.harness,
        "benchmark": spec.benchmark,
        "model_name": spec.model_name,
        "component": spec.coalition,
        "component_ids": list(spec.components),
        "component_preset": spec.component_preset,
        "evaluation_path": evaluation_path,
        "eval_mode": mode,
        "base_model": str(spec.base_model),
        "adapter_map": adapter_map,
        "score_split": score_split,
        "harness_mask": harness_mask,
        "api_base_url": getattr(args, "api_base_url", None),
        "api_model": getattr(args, "api_model", None),
        "max_turns": 2 if args.smoke else int(args.max_turns),
        "max_new_tokens": min(int(args.max_new_tokens), 256) if args.smoke else int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "search_k": int(args.search_k),
        "max_model_len": int(args.max_model_len),
        "eval_replicas": int(getattr(args, "eval_replicas", 1)),
        "eval_gpus": getattr(args, "eval_gpus", None),
        "out": str(spec.out),
    }
    (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in launch.items() if k != "harness_mask"} | {"eval_mode": mode}, indent=2), flush=True)

    rows, pool_meta = load_eval_benchmark(spec.benchmark, score_split=score_split)
    audits = []
    for cell, path in adapter_map.items():
        if path:
            audits.append(audit_saved_adapter(Path(path), cell=str(cell)))
        else:
            audits.append({"cell": cell, "adapter_dir": None, "reload_ready": True})
    if args.audit_only:
        payload = write_eval_outputs(
            spec.out,
            component_id=spec.coalition,
            summaries=[
                {
                    "setting": "audit_only",
                    "n_queries": len(rows),
                    "legal_action_rate": None,
                    "test_evidence_recall_at_5": None,
                    "mean_tool_calls_per_query": None,
                    "tool_search_cost": None,
                    "note": "Adapter audit only; live eval needs --base-model and a reachable checkpoint.",
                }
            ],
            adapter_audits=audits,
            pool_meta=pool_meta,
            runtime_audit=wiring_audit,
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    rows = rows[: args.n_eval] if args.n_eval else rows
    if args.smoke:
        rows = rows[:6]
    eval_max_turns = 2 if args.smoke else int(args.max_turns)
    eval_max_new = min(int(args.max_new_tokens), 256) if args.smoke else int(args.max_new_tokens)
    eval_temperature = float(args.temperature)

    if evaluation_path == EVALUATION_PATH_UPSTREAM_API:
        from trim.eval.harness1_api_launch import run_isolated_api_eval, run_replicated_api_eval
        from trim.upstream_harness1.model_serve import identity_from_args, parse_actor_base_urls
        from trim.upstream_harness1.pin import pin_manifest
        from trim.upstream_harness1.retrieval import retrieval_from_args
        from trim.upstream_harness1.v8d_flags import describe_mask

        identity = identity_from_args(args)
        actor_urls = parse_actor_base_urls(getattr(args, "api_base_url", None))
        retrieval = retrieval_from_args(args)
        eval_replicas = int(getattr(args, "eval_replicas", 1))
        if len(actor_urls) > 1 and eval_replicas < len(actor_urls):
            eval_replicas = len(actor_urls)
            launch["eval_replicas"] = eval_replicas
        launch["served_model"] = identity.to_dict()
        if len(actor_urls) > 1:
            launch["actor_urls"] = actor_urls
        launch["retrieval_config"] = retrieval.to_dict()
        launch["eval_profile"] = retrieval.eval_profile()
        launch["component_mask"] = describe_mask(harness_mask)
        launch["upstream"] = pin_manifest()
        launch["eval_replicas"] = eval_replicas
        (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        summaries = []
        try:
            api_eval_kwargs = dict(
                rows=rows,
                out=spec.out / "upstream_api",
                harness=spec.harness,
                harness_mask=harness_mask,
                identity=identity,
                retrieval=retrieval,
                max_turns=eval_max_turns,
                max_new_tokens=eval_max_new,
                temperature=eval_temperature,
                pool_meta=pool_meta,
            )
            if eval_replicas > 1:
                ev, traces = run_replicated_api_eval(
                    **api_eval_kwargs,
                    eval_replicas=eval_replicas,
                    stagger_s=float(getattr(args, "eval_stagger_s", 0.0) or 0.0),
                    actor_urls=actor_urls if len(actor_urls) > 1 else None,
                )
            else:
                ev, traces = run_isolated_api_eval(**api_eval_kwargs)
            ev["setting"] = "upstream_api"
            ev["evaluation_path"] = EVALUATION_PATH_UPSTREAM_API
            ev["eval_mode"] = mode
            summaries.append(ev)
            cell_dir = spec.out / "upstream_api"
            cell_dir.mkdir(parents=True, exist_ok=True)
        finally:
            if held_outer:
                release_keepalive()
        payload = write_eval_outputs(
            spec.out,
            component_id=spec.coalition,
            summaries=summaries,
            adapter_audits=audits,
            pool_meta=pool_meta,
            runtime_audit=None,
        )
        payload["evaluation_path"] = EVALUATION_PATH_UPSTREAM_API
        (spec.out / "FOUR_CELL_OFFICIAL_SUMMARY.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    if not spec.base_model:
        if held_outer:
            release_keepalive()
        raise SystemExit("pass --model_name /path/to/checkpoint for live eval")

    from trim.eval.transfer_benchmarks import open_eval_retrieval
    from trim.eval.eval_parallel import parse_gpu_ids, run_replicated_eval, write_jsonl
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.four_cell_runtime import eval_closed_loop
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive
    from trim.training.vllm_hybrid import (
        HFGenerateClient,
        SchemeARuntime,
        VLLMGenerateClient,
        default_tensor_parallel_size,
        wait_gpus_quiet,
    )

    rows = rows[: args.n_eval] if args.n_eval else rows
    if args.smoke:
        rows = rows[:6]
    eval_max_turns = 2 if args.smoke else int(args.max_turns)
    eval_max_new = min(int(args.max_new_tokens), 256) if args.smoke else int(args.max_new_tokens)
    eval_temperature = float(args.temperature)
    eval_replicas = int(getattr(args, "eval_replicas", 1))
    replica_tp = (
        int(args.tensor_parallel_size)
        if args.tensor_parallel_size
        else (1 if eval_replicas > 1 else default_tensor_parallel_size(None))
    )
    launch["eval_replicas"] = eval_replicas
    launch["tensor_parallel_size"] = replica_tp
    launch["eval_gpus"] = getattr(args, "eval_gpus", None)
    launch["max_num_seqs"] = int(getattr(args, "max_num_seqs", 256) or 256)
    (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in launch.items() if k != "harness_mask"} | {"eval_mode": mode}, indent=2), flush=True)

    summaries = []
    # Parent of --tp N must not occupy GPUs; children keep themselves busy.
    keepalive = None if eval_replicas > 1 else acquire_keepalive()
    try:
        if eval_replicas > 1:
            gpu_ids = parse_gpu_ids(args.eval_gpus)
            worker_cfg = {
                "model_path": str(spec.base_model),
                "component": spec.coalition,
                "harness_mask": harness_mask,
                "rollout_backend": args.rollout_backend,
                "max_turns": eval_max_turns,
                "max_new_tokens": eval_max_new,
                "temperature": eval_temperature,
                "search_k": int(args.search_k),
                "max_model_len": int(args.max_model_len),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "max_num_seqs": int(getattr(args, "max_num_seqs", 256) or 256),
                "eval_chunk_size": getattr(args, "eval_chunk_size", None),
                "seed": int(args.seed),
                "primary_split": score_split,
                "benchmark": spec.benchmark,
                "tensor_parallel_size": replica_tp,
            }
            for cell, path in adapter_map.items():
                ev, traces = run_replicated_eval(
                    rows=rows,
                    out=spec.out,
                    cell=str(cell),
                    adapter_path=str(path) if path else None,
                    spec_out_env=worker_cfg,
                    eval_replicas=eval_replicas,
                    gpu_ids=gpu_ids,
                    tensor_parallel_size=replica_tp,
                    stagger_s=float(getattr(args, "eval_stagger_s", 0.0) or 0.0),
                )
                ev["setting"] = cell
                ev["eval_mode"] = mode
                cell_dir = spec.out / str(cell)
                cell_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(cell_dir / "PER_QUERY.jsonl", traces)
                summaries.append(ev)
        elif args.rollout_backend == "vllm":
            enc = load_model_encoding(str(spec.base_model))
            searcher = open_eval_retrieval(spec.benchmark, formal=True)
            runtime = SchemeARuntime()
            for i, (cell, path) in enumerate(adapter_map.items()):
                if keepalive is not None:
                    keepalive.pause()
                wait_gpus_quiet()
                session = spec.out / "vllm_sessions" / f"eval_{i}_{cell}"
                client = VLLMGenerateClient(
                    model_path=str(spec.base_model),
                    session_dir=session,
                    tensor_parallel_size=replica_tp,
                    max_model_len=int(args.max_model_len),
                    lora_path=str(path) if path else None,
                    gpu_memory_utilization=float(args.gpu_memory_utilization),
                    max_num_seqs=int(getattr(args, "max_num_seqs", 256) or 256),
                )
                runtime.attach_vllm(client)
                try:
                    client.start()
                    ev, traces = eval_closed_loop(
                        None,
                        rows,
                        component_id=spec.coalition,
                        max_new=eval_max_new,
                        max_turns=eval_max_turns,
                        seed=int(args.seed),
                        enc=enc,
                        searcher=searcher,
                        generate_batch=client.generate_batch,
                        harness_mask=harness_mask,
                        temperature=eval_temperature,
                        search_k=int(args.search_k),
                        primary_split=score_split,
                    )
                finally:
                    runtime.detach_vllm()
                    if keepalive is not None:
                        keepalive.resume()
                ev["setting"] = cell
                ev["eval_mode"] = mode
                cell_dir = spec.out / str(cell)
                cell_dir.mkdir(parents=True, exist_ok=True)
                with (cell_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
                    for tr in traces:
                        handle.write(json.dumps(tr, ensure_ascii=False) + "\n")
                summaries.append(ev)
        else:
            from safetensors.torch import load_file

            from trim.eval.adapter_reload_audit import remap_lora_state
            from trim.training.hf_rl_opd_client import restore_trainable, snapshot_trainable
            from trim.training.hf_tool_opd import ScapeHFToolOPD

            enc = load_model_encoding(str(spec.base_model))
            searcher = open_eval_retrieval(spec.benchmark, formal=True)
            if keepalive is not None:
                keepalive.pause()
            gpu = str(args.gpu)
            device_map = f"cuda:{gpu}" if gpu.isdigit() else "auto"
            backend = ScapeHFToolOPD(model_path=str(spec.base_model), device_map=device_map, use_lora=True)
            theta0 = snapshot_trainable(backend.model)
            gen = HFGenerateClient(backend, enc=enc)
            for cell, path in adapter_map.items():
                restore_trainable(backend.model, theta0)
                if path:
                    weights = remap_lora_state(load_file(str(Path(path) / "adapter_model.safetensors")))
                    missing, _un = backend.model.load_state_dict(weights, strict=False)
                    if [x for x in missing if "lora_" in x]:
                        raise RuntimeError(f"reload failed: {cell}")
                ev, traces = eval_closed_loop(
                    backend,
                    rows,
                    component_id=spec.coalition,
                    max_new=eval_max_new,
                    max_turns=eval_max_turns,
                    seed=int(args.seed),
                    enc=enc,
                    searcher=searcher,
                    generate_batch=gen.generate_batch,
                    harness_mask=harness_mask,
                    temperature=eval_temperature,
                    search_k=int(args.search_k),
                    primary_split=score_split,
                )
                ev["setting"] = cell
                ev["eval_mode"] = mode
                cell_dir = spec.out / str(cell)
                cell_dir.mkdir(parents=True, exist_ok=True)
                with (cell_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
                    for tr in traces:
                        handle.write(json.dumps(tr, ensure_ascii=False) + "\n")
                summaries.append(ev)
    finally:
        if keepalive is not None:
            release_keepalive()

    runtime_audit = None
    for ev in summaries:
        if isinstance(ev, dict) and ev.get("runtime_effect_audit"):
            runtime_audit = ev["runtime_effect_audit"]
            break
    runtime_audit = runtime_audit or wiring_audit
    payload = write_eval_outputs(
        spec.out,
        component_id=spec.coalition,
        summaries=summaries,
        adapter_audits=audits,
        pool_meta=pool_meta,
        runtime_audit=runtime_audit,
    )
    print(json.dumps(payload, indent=2), flush=True)
    if held_outer:
        release_keepalive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
