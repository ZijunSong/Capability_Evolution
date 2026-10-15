"""Cost ledger. Hidden tool calls and extra model calls are not free."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize_episode_costs(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy_tokens = []
    extra_llm = []
    tool_calls = []
    hidden = []
    latency = []
    component_s = []
    for episode in episodes:
        policy_tokens.append(float(episode.get("policy_tokens") or 0))
        extra_llm.append(float(episode.get("extra_llm_tokens") or 0))
        explicit = float(episode.get("explicit_tool_calls") or 0)
        hidden_calls = float(episode.get("hidden_tool_calls") or 0)
        tool_calls.append(explicit + hidden_calls)
        hidden.append(hidden_calls)
        latency.append(float(episode.get("latency_s") or 0))
        component_s.append(float(episode.get("component_wall_s") or 0))
    n = max(len(episodes), 1)
    total_latency = sum(latency)
    return {
        "n_episodes": len(episodes),
        "policy_tokens": sum(policy_tokens),
        "extra_llm_tokens": sum(extra_llm),
        "tool_calls": sum(tool_calls),
        "hidden_tool_calls": sum(hidden),
        "component_wall_s": sum(component_s),
        "latency_p95_s": percentile(latency, 0.95),
        "mean_latency_s": (sum(latency) / n) if episodes else None,
        "throughput_episodes_per_s": (len(episodes) / total_latency) if total_latency > 0 else None,
    }


def within_caps(summary: Mapping[str, Any], caps: Mapping[str, Any], *, baseline: Mapping[str, Any] | None = None) -> bool:
    if int(summary.get("hidden_tool_calls") or 0) < 0:
        return False
    max_tools = caps.get("max_total_tool_calls_per_episode")
    if max_tools is not None and summary["n_episodes"]:
        mean_tools = float(summary["tool_calls"]) / float(summary["n_episodes"])
        if mean_tools > float(max_tools):
            return False
    if baseline and caps.get("max_p95_latency_ratio") is not None:
        base = baseline.get("latency_p95_s")
        live = summary.get("latency_p95_s")
        if base and live is not None and float(live) > float(caps["max_p95_latency_ratio"]) * float(base):
            return False
    if baseline and caps.get("max_policy_token_ratio") is not None:
        base_tokens = float(baseline.get("policy_tokens") or 0)
        if base_tokens and float(summary["policy_tokens"]) > float(caps["max_policy_token_ratio"]) * base_tokens:
            return False
    return True
