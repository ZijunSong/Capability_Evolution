#!/usr/bin/env python3
"""One-click Harness-1 / BC+ closed-loop eval.

Defaults match Harness-1 Table-2 eval: --max-turns 40, --max-new-tokens 2048,
--temperature 1.0, --search-k 10, --max-model-len 32768. Training stay on a
short horizon; do not copy those smoke values into eval.

Score split: ``--benchmark bcplus_full`` (or ``BC+``) uses the 830-query pool
(664 train + 166 test). ``--benchmark bcplus_test_166`` uses the 166-test subset.

Without --run-dir / --adapter, listed --component flags are turned ON (harness eval).
With a trained run directory, the student is scored under H_min (those flags OFF)
plus the saved LoRA adapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.cli.launch import (
    LaunchError,
    discover_adapter_map,
    parse_eval_args,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from scape.eval.adapter_reload_audit import audit_saved_adapter
from scape.eval.official_query_pool import (
    SCORE_SPLIT_830,
    canonical_score_split,
    is_full_score_split,
    load_bcplus_830_full,
    load_bcplus_830_split,
    score_split_for_benchmark,
)
from scape.eval.sr_opd_four_cell_eval import write_eval_outputs


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
    mapping = _adapter_map(args)
    mode = args.eval_mode
    if mode == "auto":
        mode = "adapter" if any(mapping.values()) else "harness"
    if mode == "harness":
        return "harness", mapping or {"harness": None}
    return "adapter", mapping or {"before": None}


def detect_score_split(args) -> str:
    explicit = getattr(args, "score_split", None)
    if explicit:
        return canonical_score_split(str(explicit), default=SCORE_SPLIT_830) or SCORE_SPLIT_830
    implied = score_split_for_benchmark(str(getattr(args, "benchmark", "") or ""))
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
    harness_mask = (
        teacher_mask_for_ids(spec.components)
        if mode == "harness"
        else student_mask_for_ids(spec.components)
    )

    spec.out.mkdir(parents=True, exist_ok=True)
    launch = {
        "harness": spec.harness,
        "benchmark": spec.benchmark,
        "model_name": spec.model_name,
        "component": spec.coalition,
        "component_ids": list(spec.components),
        "eval_mode": mode,
        "base_model": str(spec.base_model),
        "adapter_map": adapter_map,
        "score_split": score_split,
        "harness_mask": harness_mask,
        "max_turns": 2 if args.smoke else int(args.max_turns),
        "max_new_tokens": min(int(args.max_new_tokens), 256) if args.smoke else int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "search_k": int(args.search_k),
        "max_model_len": int(args.max_model_len),
        "out": str(spec.out),
    }
    (spec.out / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in launch.items() if k != "harness_mask"} | {"eval_mode": mode}, indent=2), flush=True)

    if is_full_score_split(score_split):
        rows, pool_meta = load_bcplus_830_full()
    else:
        _train, rows, pool_meta = load_bcplus_830_split()
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
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    if not spec.base_model:
        raise SystemExit("pass --base-model or use the default harness-1 checkpoint for live eval")

    from scape.eval.browsecomp_retrieval import open_retrieval
    from scape.eval.harmony_runtime import load_harmony_enc
    from scape.training.four_cell_runtime import eval_closed_loop
    from scape.training.vllm_hybrid import (
        HFGenerateClient,
        SchemeARuntime,
        VLLMGenerateClient,
        default_tensor_parallel_size,
        wait_gpus_quiet,
    )

    rows = rows[: args.n_eval] if args.n_eval else rows
    if args.smoke:
        rows = rows[:6]
    enc = load_harmony_enc()
    searcher = open_retrieval()
    eval_max_turns = 2 if args.smoke else int(args.max_turns)
    eval_max_new = min(int(args.max_new_tokens), 256) if args.smoke else int(args.max_new_tokens)
    eval_temperature = float(args.temperature)
    summaries = []
    runtime = SchemeARuntime()
    if args.rollout_backend == "vllm":
        tp = default_tensor_parallel_size(args.tensor_parallel_size)
        for i, (cell, path) in enumerate(adapter_map.items()):
            wait_gpus_quiet()
            session = spec.out / "vllm_sessions" / f"eval_{i}_{cell}"
            client = VLLMGenerateClient(
                model_path=str(spec.base_model),
                session_dir=session,
                tensor_parallel_size=tp,
                max_model_len=int(args.max_model_len),
                lora_path=str(path) if path else None,
                gpu_memory_utilization=float(args.gpu_memory_utilization),
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

        from scape.eval.adapter_reload_audit import remap_lora_state
        from scape.training.hf_rl_opd_client import restore_trainable, snapshot_trainable
        from scape.training.hf_tool_opd import ScapeHFToolOPD

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

    payload = write_eval_outputs(
        spec.out,
        component_id=spec.coalition,
        summaries=summaries,
        adapter_audits=audits,
        pool_meta=pool_meta,
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
