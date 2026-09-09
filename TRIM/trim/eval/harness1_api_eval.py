"""Official Harness-1 eval: original env + chat/completions adapter (E02/E07)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.upstream_harness1.api_adapter import ChatCompletionsClient, parse_chat_completion
from trim.upstream_harness1.model_serve import ServedModelIdentity
from trim.upstream_harness1.pin import pin_manifest
from trim.upstream_harness1.retrieval import RetrievalConfig
from trim.upstream_harness1.v8d_flags import (
    EVALUATION_PATH_UPSTREAM_API,
    describe_mask,
    v8d_env_from_mask,
)


def _terminal_metrics(env: Any) -> dict[str, Any]:
    metrics = dict(getattr(env, "_terminal_metrics", None) or {})
    metrics.setdefault("reward", float(getattr(env, "_terminal_reward", 0.0) or 0.0))
    metrics.setdefault("num_turns", int(getattr(env, "_current_turn", 0) or 0))
    metrics.setdefault("n_curated", len(getattr(getattr(env, "wm", None), "curated_ids", []) or []))
    return metrics


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_one_query_api(
    *,
    env: Any,
    mods: Mapping[str, Any],
    client: ChatCompletionsClient,
    query_row: Mapping[str, Any],
    trace_dir: Path,
    max_turns: int,
) -> dict[str, Any]:
    from trim.upstream_harness1.env_bridge import (
        action_from_parsed,
        openai_messages_from_env,
        openai_tools_from_env,
    )

    await env.initial_observation()
    qid = str(query_row.get("query_id") or env.query_id)
    turns: list[dict[str, Any]] = []
    started = time.time()
    done = False
    finish_reason = "running"
    while not done and env._current_turn < max_turns:
        messages = openai_messages_from_env(env, mods)
        tools = openai_tools_from_env(env, mods)
        t0 = time.time()
        response = client.complete(messages, tools)
        model_s = time.time() - t0
        parsed = parse_chat_completion(response)
        usage = response.get("usage") or {}
        turn_rec = {
            "query_id": qid,
            "turn": int(env._current_turn),
            "request_messages": messages,
            "tools": tools,
            "api_response": {k: v for k, v in response.items() if k != "choices"} | {
                "choices": response.get("choices"),
            },
            "finish_reason": parsed.finish_reason,
            "reasoning": parsed.reasoning,
            "parse_error": parsed.parse_error,
            "usage": usage,
            "model_sec": model_s,
            "request_fingerprint": response.get("_request_fingerprint"),
        }
        t1 = time.time()
        try:
            if parsed.parse_error:
                result = env._handle_format_error(parsed.parse_error)
            else:
                action = action_from_parsed(parsed, env, mods)
                result = await env.step_action(action)
            done = bool(result.episode_done)
            finish_reason = "episode_done" if done else "continue"
            turn_rec["env_metrics"] = dict(result.metrics or {})
            turn_rec["reward"] = float(result.reward)
            turn_rec["episode_done"] = done
        except Exception as exc:  # noqa: BLE001
            result = env._handle_format_error(str(exc))
            done = bool(result.episode_done)
            finish_reason = "format_or_tool_error"
            turn_rec["env_exception"] = repr(exc)
            turn_rec["episode_done"] = done
        turn_rec["harness_sec"] = time.time() - t1
        turns.append(turn_rec)
        _append_jsonl(trace_dir / "TURNS.jsonl", turn_rec)
        if done:
            break
    metrics = _terminal_metrics(env)
    metrics.update(
        {
            "query_id": qid,
            "query_text": str(query_row.get("query") or getattr(env, "query_text", "") or ""),
            "ended": bool(getattr(env, "_episode_ended", False)),
            "finish_reason": finish_reason,
            "n_turns": len(turns),
            "e2e_sec": time.time() - started,
            "evaluation_path": EVALUATION_PATH_UPSTREAM_API,
        }
    )
    _append_jsonl(trace_dir / "PER_QUERY.jsonl", {k: v for k, v in metrics.items()})
    return {"metrics": metrics, "turns": turns}


def write_run_manifest(
    out: Path,
    *,
    mask: Mapping[str, bool],
    identity: ServedModelIdentity,
    retrieval: RetrievalConfig,
    pool_meta: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "evaluation_path": EVALUATION_PATH_UPSTREAM_API,
        "upstream": pin_manifest(),
        "component_mask": describe_mask(mask),
        "v8d_env": v8d_env_from_mask(mask),
        "served_model": identity.to_dict(),
        "retrieval_config": retrieval.to_dict(),
        "eval_profile": retrieval.eval_profile(),
        "query_manifest": dict(pool_meta),
        "protocol": identity.protocol,
    }
    if extra:
        payload.update(dict(extra))
    out.mkdir(parents=True, exist_ok=True)
    (out / "EVAL_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def summarize_api_traces(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = max(1, len(traces))
    recalls = [float(t.get("recall") or 0.0) for t in traces]
    f1s = [float(t.get("f1") or 0.0) for t in traces]
    fa = [float(t.get("final_answer_recall") or 0.0) for t in traces]
    format_err = [float(t.get("format_error") or 0.0) for t in traces]
    return {
        "evaluation_path": EVALUATION_PATH_UPSTREAM_API,
        "n_queries": len(traces),
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
        "final_answer_recall": sum(fa) / n,
        "format_error_rate": sum(format_err) / n,
        "mean_turns": sum(float(t.get("num_turns") or t.get("n_turns") or 0.0) for t in traces) / n,
        "dropped_queries": 0,
    }
