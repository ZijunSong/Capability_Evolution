#!/usr/bin/env python3
"""Isolated Harness-1 API eval worker.

Sets V8D_* before importing ultra_core. One process, one mask.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[2]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upstream Harness-1 API eval worker")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = _load_json(args.config)

    mask = dict(cfg["harness_mask"])
    from trim.upstream_harness1.v8d_flags import subprocess_env_for_mask

    # Flags must be in os.environ before ultra_core import.
    for key, value in subprocess_env_for_mask(mask, harness=cfg.get("harness") or "Harness-1").items():
        os.environ[str(key)] = str(value)

    from trim.eval.harness1_api_eval import (
        _cfg_number,
        assert_fresh_eval_dir,
        run_one_query_api,
        summarize_api_traces,
        write_infra_error_query,
        write_run_manifest,
    )
    from trim.upstream_harness1.api_adapter import ConfigError
    from trim.eval.tool_health import build_tool_health_payload, write_tool_health
    from trim.upstream_harness1.api_adapter import ChatCompletionsClient
    from trim.upstream_harness1.env_bridge import build_eval_toolset, load_scoring_dataset, load_upstream_modules
    from trim.upstream_harness1.model_serve import ServedModelIdentity
    from trim.upstream_harness1.retrieval import RETRIEVAL_LOCAL_BM25, RetrievalConfig
    from trim.upstream_harness1.token_count import resolve_token_counter

    retrieval = RetrievalConfig.from_mapping(cfg.get("retrieval") or {})
    retrieval.assert_ready()
    if retrieval.backend == RETRIEVAL_LOCAL_BM25:
        os.environ["HARNESS1_FORBID_CHROMA"] = "1"

    mods = load_upstream_modules()
    identity = ServedModelIdentity(**cfg["served_model"])
    identity.assert_tool_calling()

    ensure_harness1 = mods["root"]
    sys.path.insert(0, str(ensure_harness1))
    dataset = load_scoring_dataset(retrieval.dataset)
    token_counter, token_count_mode = resolve_token_counter(
        identity.base_model or cfg.get("model_path") or identity.api_model
    )
    pack = build_eval_toolset(
        mods, retrieval, dataset=dataset, mask=mask, token_counter=token_counter
    )
    Env = mods["SlidingWindowSearchEnv"]
    temperature = _cfg_number(cfg, "temperature", 1.0)
    max_new_tokens = int(_cfg_number(cfg, "max_new_tokens", 2048))
    max_model_len = int(_cfg_number(cfg, "max_model_len", 32768))
    prompt_token_budget = max(1024, max_model_len - max_new_tokens - 512)
    client = ChatCompletionsClient(
        base_url=identity.api_base_url,
        model=identity.api_model,
        api_key=os.environ.get("OPENAI_API_KEY") or cfg.get("api_key"),
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    rows = json.loads(Path(cfg["queries_path"]).read_text(encoding="utf-8"))
    out = Path(cfg["out"])
    assert_fresh_eval_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        out,
        mask=mask,
        identity=identity,
        retrieval=retrieval,
        pool_meta=cfg.get("pool_meta") or {},
        extra={
            "worker_rank": cfg.get("rank"),
            "eval_profile": retrieval.eval_profile(),
            "capability_log": pack.capability_log,
            "token_count_mode": token_count_mode,
            "sampling": {"temperature": temperature, "max_tokens": max_new_tokens, "model": identity.api_model},
        },
    )
    traces: list[dict] = []
    infra_errors: list[str] = []

    async def _run() -> None:
        for row in rows:
            qid = str(row["query_id"])
            query_text = str(row.get("query") or "")
            if not query_text and hasattr(pack.dataset, "get_query_text"):
                try:
                    query_text = str(pack.dataset.get_query_text(qid) or "")
                except Exception:
                    query_text = ""
            env_kwargs: dict = {
                "toolset": pack.toolset,
                "search_tool": pack.search_tool,
                "query_id": qid,
                "query_text": query_text or str(row.get("query") or ""),
                "dataset": pack.dataset,
                "max_turns": int(cfg.get("max_turns") or 40),
                "text_token_counter": token_counter,
            }
            if pack.verifier_client is not None:
                env_kwargs["openai_client"] = pack.verifier_client
            try:
                env = Env(**env_kwargs)
                result = await run_one_query_api(
                    env=env,
                    mods=mods,
                    client=client,
                    query_row={**row, "query": query_text or row.get("query")},
                    trace_dir=out,
                    max_turns=int(cfg.get("max_turns") or 40),
                    token_counter=token_counter,
                    prompt_token_budget=prompt_token_budget,
                )
                traces.append(result["metrics"])
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                infra_errors.append(qid)
                traces.append(
                    write_infra_error_query(trace_dir=out, query_row=row, exc=exc)
                )

    try:
        asyncio.run(_run())
    finally:
        tool_health = build_tool_health_payload(
            pack.capability_log,
            worker_rank=cfg.get("rank"),
            phase="partial" if infra_errors else "final",
        )
        write_tool_health(out / "TOOL_HEALTH.json", tool_health)

    summary = summarize_api_traces(traces)
    summary["n_infra_error"] = len(infra_errors)
    summary["infra_error_query_ids"] = infra_errors
    summary["eval_profile"] = retrieval.eval_profile()
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        out,
        mask=mask,
        identity=identity,
        retrieval=retrieval,
        pool_meta=cfg.get("pool_meta") or {},
        extra={
            "worker_rank": cfg.get("rank"),
            "eval_profile": retrieval.eval_profile(),
            "capability_log": pack.capability_log,
            "token_count_mode": token_count_mode,
            "sampling": {"temperature": temperature, "max_tokens": max_new_tokens, "model": identity.api_model},
            "tool_health_path": "TOOL_HEALTH.json",
        },
    )
    (out / "DONE.json").write_text(
        json.dumps(
            {
                "ok": True,
                "n_queries": len(traces),
                "n_infra_error": len(infra_errors),
                "partial": bool(infra_errors),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
