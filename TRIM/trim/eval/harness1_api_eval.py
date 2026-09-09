"""Official Harness-1 eval: original env + chat/completions adapter (E02/E07)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.harness1_metrics import f1_score
from trim.upstream_harness1.api_adapter import ChatCompletionsClient, parse_chat_completion
from trim.upstream_harness1.model_serve import ServedModelIdentity
from trim.upstream_harness1.pin import pin_manifest
from trim.upstream_harness1.retrieval import RetrievalConfig
from trim.upstream_harness1.v8d_flags import (
    EVALUATION_PATH_UPSTREAM_API,
    describe_mask,
    v8d_env_from_mask,
)

TRACE_JSONL_NAMES = ("TURNS.jsonl", "PER_QUERY.jsonl")


def assert_fresh_eval_dir(out: Path) -> None:
    """Refuse implicit append into an existing eval run (R07)."""
    existing = [name for name in TRACE_JSONL_NAMES if (out / name).is_file()]
    if (out / "SUMMARY.json").is_file():
        existing.append("SUMMARY.json")
    if existing:
        raise RuntimeError(
            f"eval output dir {out} already has {existing}. "
            "Use a new --out directory. Implicit JSONL append is not resume."
        )


def _cfg_number(payload: Mapping[str, Any], key: str, default: float) -> float:
    if key not in payload or payload.get(key) is None:
        return float(default)
    return float(payload[key])


def count_tool_calls_from_turns(turns: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    n_api = 0
    n_search = 0
    n_fan_out = 0
    n_executed = 0
    for turn in turns:
        response = turn.get("api_response") or {}
        choices = response.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        api_calls = list(message.get("tool_calls") or [])
        n_api += len(api_calls)
        names = [str((c.get("function") or {}).get("name") or "") for c in api_calls]
        n_search += sum(1 for n in names if n == "search_corpus")
        n_fan_out += sum(1 for n in names if n == "fan_out_search")
        if turn.get("parse_error"):
            continue
        n_executed += len(api_calls)
    return {
        "n_tool_calls": float(n_executed),
        "n_api_tool_calls": float(n_api),
        "n_search_calls": float(n_search),
        "n_fan_out_calls": float(n_fan_out),
        "n_search_plus_fan_out": float(n_search + n_fan_out),
    }


def finish_reason_from_step(*, done: bool, step_metrics: Mapping[str, Any], env: Any, max_turns: int) -> str:
    if float(step_metrics.get("format_error") or 0.0) >= 1.0:
        return "format_error"
    if float(step_metrics.get("max_turns_reached") or 0.0) >= 1.0:
        return "max_turns"
    if done and bool(getattr(env, "_episode_ended", False)):
        return "end_search"
    if done:
        return "episode_done"
    if int(getattr(env, "_current_turn", 0) or 0) >= int(max_turns):
        return "max_turns"
    return "continue"


_COHORT_QUALITY_KEYS: tuple[str, ...] = (
    "recall",
    "precision",
    "f1",
    "f_beta",
    "trajectory_recall",
    "final_answer_recall",
)


def _resolved_n_curated(metrics: Mapping[str, Any]) -> float | None:
    for key in ("n_curated", "num_curated_docs"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _impute_cohort_quality(out: dict[str, Any]) -> dict[str, Any]:
    """Fill deterministically known zeros so cohort summaries use a fixed denominator."""
    format_error = float(out.get("format_error") or 0.0) >= 1.0
    n_curated = _resolved_n_curated(out)
    imputed: list[str] = []

    if format_error:
        for key in _COHORT_QUALITY_KEYS:
            if out.get(key) is None:
                out[key] = 0.0
                imputed.append(key)
    elif n_curated is not None and n_curated <= 0.0:
        for key in ("recall", "precision", "f1", "f_beta", "final_answer_recall"):
            if out.get(key) is None:
                out[key] = 0.0
                imputed.append(key)
        if out.get("final_answer_recall") is None:
            out["final_answer_recall"] = 0.0
            imputed.append("final_answer_recall")
    elif out.get("final_answer_recall") is None and out.get("recall") is not None and out.get("precision") is not None:
        if float(out.get("recall") or 0.0) == 0.0 and float(out.get("precision") or 0.0) == 0.0:
            out["final_answer_recall"] = 0.0
            imputed.append("final_answer_recall")

    if out.get("trajectory_recall") is None and format_error:
        out["trajectory_recall"] = 0.0
        imputed.append("trajectory_recall")

    out["quality_imputed"] = sorted(set(imputed))
    return out


def normalize_query_metrics(
    metrics: Mapping[str, Any],
    *,
    step_metrics: Mapping[str, Any] | None = None,
    turns: Sequence[Mapping[str, Any]] | None = None,
    done: bool | None = None,
) -> dict[str, Any]:
    out = dict(metrics)
    merged = dict(step_metrics or {})
    for key, value in merged.items():
        if key not in out or out.get(key) is None or key in {
            "format_error",
            "no_error",
            "format_retry",
            "max_turns_reached",
        }:
            out[key] = value
    if float(merged.get("format_error") or 0.0) >= 1.0 and merged.get("reward") is not None:
        out["reward"] = merged["reward"]
    precision = out.get("precision")
    recall = out.get("recall")
    if precision is not None and recall is not None:
        out["f1"] = f1_score(float(precision), float(recall))
        out["f1_missing"] = False
    else:
        out["f1"] = None
        out["f1_missing"] = True
    if out.get("f_beta") is None and precision is not None and recall is not None:
        from trim.eval.harness1_metrics import f_beta_score

        out["f_beta"] = f_beta_score(float(precision), float(recall))
    if done is not None:
        out["ended"] = bool(done) or bool(out.get("ended"))
    if turns is not None:
        out.update(count_tool_calls_from_turns(turns))
        out["n_turns"] = len(turns)
        out["num_turns"] = out.get("num_turns") if out.get("num_turns") is not None else len(turns)
    out = _impute_cohort_quality(out)
    missing = [key for key in _COHORT_QUALITY_KEYS if out.get(key) is None]
    out["missing_metrics"] = missing
    return out


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
    token_counter: Any | None = None,
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
    awaiting_retry = False
    last_step_metrics: dict[str, Any] = {}
    while not done and env._current_turn < max_turns:
        messages = openai_messages_from_env(
            env, mods, retry=awaiting_retry, token_counter=token_counter or getattr(env, "text_token_counter", None)
        )
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
            "sampling": {
                "temperature": client.temperature,
                "max_tokens": client.max_tokens,
                "model": client.model,
            },
            "format_retry_turn": bool(awaiting_retry),
        }
        t1 = time.time()
        result: Any = None
        try:
            if parsed.parse_error:
                result = env._handle_format_error(parsed.parse_error)
                awaiting_retry = not bool(result.episode_done)
            else:
                awaiting_retry = False
                action = action_from_parsed(parsed, env, mods)
                result = await env.step_action(action)
            last_step_metrics = dict(result.metrics or {})
            last_step_metrics.setdefault("reward", float(result.reward))
            done = bool(result.episode_done)
            finish_reason = finish_reason_from_step(
                done=done, step_metrics=last_step_metrics, env=env, max_turns=max_turns
            )
            turn_rec["env_metrics"] = dict(last_step_metrics)
            turn_rec["reward"] = float(result.reward)
            turn_rec["episode_done"] = done
        except Exception as exc:  # noqa: BLE001
            result = env._handle_format_error(str(exc))
            last_step_metrics = dict(result.metrics or {})
            last_step_metrics.setdefault("reward", float(result.reward))
            done = bool(result.episode_done)
            awaiting_retry = not done
            finish_reason = "tool_error" if not last_step_metrics.get("format_error") else "format_error"
            if done and last_step_metrics.get("format_error"):
                finish_reason = "format_error"
            turn_rec["env_exception"] = repr(exc)
            turn_rec["episode_done"] = done
            turn_rec["env_metrics"] = dict(last_step_metrics)
        turn_rec["harness_sec"] = time.time() - t1
        turns.append(turn_rec)
        _append_jsonl(trace_dir / "TURNS.jsonl", turn_rec)
        if done:
            break
    if not done and int(getattr(env, "_current_turn", 0) or 0) >= int(max_turns):
        finish_reason = "max_turns"
        done = True
        last_step_metrics.setdefault("max_turns_reached", 1.0)
    metrics = normalize_query_metrics(
        _terminal_metrics(env),
        step_metrics=last_step_metrics,
        turns=turns,
        done=done,
    )
    metrics.update(
        {
            "query_id": qid,
            "query_text": str(query_row.get("query") or getattr(env, "query_text", "") or ""),
            "finish_reason": finish_reason,
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


def _cohort_stat(values: Sequence[float | None], *, n_expected: int) -> dict[str, Any]:
    """Fixed-denominator mean: every expected query counts; unresolved values become 0."""
    n_expected = int(n_expected)
    if n_expected <= 0:
        return {"mean": 0.0, "n_expected": 0, "n_scored": 0, "n_missing": 0}
    padded = list(values[:n_expected])
    if len(padded) < n_expected:
        padded.extend([None] * (n_expected - len(padded)))
    scored = [float(v) for v in padded if v is not None]
    n_missing = sum(1 for v in padded if v is None)
    total = sum(float(v or 0.0) for v in padded)
    return {
        "mean": total / n_expected,
        "n_expected": n_expected,
        "n_scored": len(scored),
        "n_missing": n_missing,
    }


def summarize_api_traces(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    denom = max(1, n)
    recall = _cohort_stat([t.get("recall") for t in traces], n_expected=n)
    precision = _cohort_stat([t.get("precision") for t in traces], n_expected=n)
    f1 = _cohort_stat([t.get("f1") for t in traces], n_expected=n)
    traj = _cohort_stat([t.get("trajectory_recall") for t in traces], n_expected=n)
    fa = _cohort_stat([t.get("final_answer_recall") for t in traces], n_expected=n)
    format_err = [float(t.get("format_error") or 0.0) for t in traces]
    tool_calls = [float(t.get("n_tool_calls") or 0.0) for t in traces]
    search_calls = [float(t.get("n_search_plus_fan_out") or t.get("n_search_calls") or 0.0) for t in traces]
    return {
        "evaluation_path": EVALUATION_PATH_UPSTREAM_API,
        "n_queries": n,
        "recall": recall["mean"],
        "precision": precision["mean"],
        "f1": f1["mean"],
        "f1_n": f1["n_scored"],
        "f1_missing": f1["n_missing"],
        "recall_missing": recall["n_missing"],
        "precision_missing": precision["n_missing"],
        "trajectory_recall": traj["mean"],
        "trajectory_recall_missing": traj["n_missing"],
        "final_answer_recall": fa["mean"],
        "final_answer_recall_missing": fa["n_missing"],
        "cohort_denominator": n,
        "metric_denominators": {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "trajectory_recall": traj,
            "final_answer_recall": fa,
        },
        "format_error_rate": sum(format_err) / denom,
        "mean_turns": sum(float(t.get("num_turns") or t.get("n_turns") or 0.0) for t in traces) / denom,
        "mean_tool_calls_per_query": sum(tool_calls) / denom,
        "mean_search_and_fan_out_per_query": sum(search_calls) / denom,
        "dropped_queries": 0,
    }
