#!/usr/bin/env python3
"""Eval one query shard on the GPUs selected by CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[2]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.eval_parallel import load_json, write_json, write_jsonl


def _run_vllm(cfg: dict, rows: list[dict], harness_mask: dict) -> tuple[dict, list[dict]]:
    import threading

    from trim.eval.browsecomp_retrieval import open_retrieval
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.gpu_keepalive import GpuKeepAlive
    from trim.training.vllm_hybrid import SchemeARuntime, VLLMGenerateClient

    holder: dict = {}
    errors: list[BaseException] = []

    def cpu_prep() -> None:
        # Tokenizer/Harmony only. Pyserini/JNI must not start on this short-lived thread:
        # LuceneSearcher.search() then silently returns empty hits.
        try:
            holder["enc"] = load_model_encoding(str(cfg.get("model_path") or ""))
        except BaseException as exc:
            errors.append(exc)

    keepalive = GpuKeepAlive()
    keepalive.start()
    # Overlap Harmony load with vLLM startup. Lucene stays on the main thread
    # after the vLLM worker process is spawned.
    prep = threading.Thread(target=cpu_prep, name="trim-eval-prep", daemon=True)
    prep.start()
    out = Path(cfg["out"])
    session = out / "vllm_session"
    client = VLLMGenerateClient(
        model_path=str(cfg["model_path"]),
        session_dir=session,
        tensor_parallel_size=int(cfg.get("tensor_parallel_size") or 1),
        max_model_len=int(cfg.get("max_model_len") or 32768),
        lora_path=cfg.get("adapter_path") or None,
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization") or 0.90),
        max_num_seqs=int(cfg.get("max_num_seqs") or 256),
    )
    keepalive.pause()
    runtime = SchemeARuntime()
    runtime.attach_vllm(client)
    try:
        client.start()
        searcher = open_retrieval(formal=True)
        prep.join()
        if errors:
            raise errors[0]
        return _eval_chunks(
            cfg,
            rows,
            harness_mask=harness_mask,
            enc=holder["enc"],
            searcher=searcher,
            generate_batch=client.generate_batch,
            backend=None,
        )
    finally:
        prep.join(timeout=120.0)
        runtime.detach_vllm()
        keepalive.stop()


def _run_hf(cfg: dict, rows: list[dict], harness_mask: dict) -> tuple[dict, list[dict]]:
    from safetensors.torch import load_file

    from trim.eval.adapter_reload_audit import remap_lora_state
    from trim.eval.browsecomp_retrieval import open_retrieval
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.gpu_keepalive import GpuKeepAlive
    from trim.training.hf_rl_opd_client import restore_trainable, snapshot_trainable
    from trim.training.hf_tool_opd import ScapeHFToolOPD
    from trim.training.vllm_hybrid import HFGenerateClient

    keepalive = GpuKeepAlive()
    keepalive.start()
    try:
        enc = load_model_encoding(str(cfg.get("model_path") or ""))
        searcher = open_retrieval(formal=True)
        keepalive.pause()
        backend = ScapeHFToolOPD(model_path=str(cfg["model_path"]), device_map="cuda:0", use_lora=True)
    finally:
        keepalive.stop()
    theta0 = snapshot_trainable(backend.model)
    adapter = cfg.get("adapter_path")
    if adapter:
        weights = remap_lora_state(load_file(str(Path(adapter) / "adapter_model.safetensors")))
        missing, _un = backend.model.load_state_dict(weights, strict=False)
        if [x for x in missing if "lora_" in x]:
            raise RuntimeError(f"reload failed: {cfg.get('cell')}")
    else:
        restore_trainable(backend.model, theta0)
    gen = HFGenerateClient(backend, enc=enc)
    return _eval_chunks(
        cfg,
        rows,
        harness_mask=harness_mask,
        enc=enc,
        searcher=searcher,
        generate_batch=gen.generate_batch,
        backend=backend,
    )


def _eval_chunks(cfg, rows, *, harness_mask, enc, searcher, generate_batch, backend):
    from trim.eval.eval_parallel import merge_traces, summarize_merged_traces
    from trim.training.four_cell_runtime import eval_closed_loop

    chunk = int(cfg.get("eval_chunk_size") or 0)
    parts = [rows] if chunk <= 0 or chunk >= len(rows) else [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    all_traces: list[list[dict]] = []
    leak_count = 0
    last_ev: dict = {}
    for part in parts:
        ev, traces = eval_closed_loop(
            backend,
            part,
            component_id=cfg["component"],
            max_new=int(cfg["max_new_tokens"]),
            max_turns=int(cfg["max_turns"]),
            seed=int(cfg.get("seed") or 42),
            enc=enc,
            searcher=searcher,
            generate_batch=generate_batch,
            harness_mask=harness_mask,
            temperature=float(cfg.get("temperature") or 0.0),
            search_k=int(cfg.get("search_k") or 10),
            primary_split=str(cfg.get("primary_split") or "official_test"),
        )
        leak_count += int(ev.get("teacher_leak_count") or 0)
        all_traces.append(traces)
        last_ev = ev
    traces = merge_traces(all_traces, rows)
    extra = {
        k: last_ev.get(k)
        for k in ("max_turns", "max_new_tokens", "temperature", "search_k", "doc_store_k", "sample")
        if k in last_ev
    }
    extra["teacher_leak_count"] = leak_count
    extra["rank"] = int(cfg.get("rank") or 0)
    summary = summarize_merged_traces(
        traces,
        rows,
        leak_count=leak_count,
        primary_split=str(cfg.get("primary_split") or "official_test"),
        extra=extra,
    )
    return summary, traces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRIM eval shard worker")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_json(args.config)
    rows = load_json(Path(cfg["queries_path"]))
    harness_mask = cfg.get("harness_mask") or {}
    backend = str(cfg.get("rollout_backend") or "vllm")
    try:
        if backend == "hf":
            summary, traces = _run_hf(cfg, rows, harness_mask)
        else:
            summary, traces = _run_vllm(cfg, rows, harness_mask)
        out = Path(cfg["out"])
        write_jsonl(out / "PER_QUERY.jsonl", traces)
        write_json(out / "SUMMARY.json", summary)
        write_json(
            out / "DONE.json",
            {
                "ok": True,
                "rank": int(cfg.get("rank") or 0),
                "n_queries": len(traces),
                "leak_count": int(summary.get("teacher_leak_count") or 0),
            },
        )
        return 0
    except Exception as exc:
        out = Path(cfg["out"])
        write_json(out / "DONE.json", {"ok": False, "error": repr(exc), "rank": int(cfg.get("rank") or 0)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
