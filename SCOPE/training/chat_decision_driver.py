"""Protocol-agnostic chat driver over SlidingWindowSearchEnv.

Decouples SCOPE DecisionState from Harmony tokens:
  Qwen / OpenAI chat messages  →  Action  →  env.step_from_action
  DecisionState is exported from WorkingMemory, independent of message codec.

Context policy (important): do NOT accumulate raw tool dumps like a naive
multi-turn chat. Rebuild each turn from:
  system prompt + budgeted WM text + short recent action/result summaries
    10|matching the SlidingWindow / render_context_within_budget spirit.

v2 robustness (weak Instruct policies on BrowseComp):
  - remap legacy Document tool / XML final-answer format → curate
  - block premature end_search when curated empty / too few turns
  - aggressive context clipping + overflow retry
  - inject search→curate rhythm nudges into the rebuilt prompt
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from harness.action_repair import (
    maybe_convert_user_text_to_tools,
    normalize_tool_params,
    should_block_early_end,
)
from harness.agent import InferenceContext, OpenAIAgentInferenceModel
from harness.capability.adapters import parse_action_from_tools
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState
from harness.tools import UserTextTool
from harness.trajectory import Action, Observation, Trajectory
from harness.ultra_core import CURATE_NUDGE_PROMPT, get_system_prompt
from training.train_rl import CurateTool, EndSearchTool, SlidingWindowSearchEnv

# Keep chat prompt comfortably under max_model_len - max_tokens.
_MAX_WM_CHARS = int(os.environ.get("CHAT_MAX_WM_CHARS", "18000"))
_MAX_OBS_CHARS = int(os.environ.get("CHAT_MAX_OBS_CHARS", "2000"))
_MAX_RECENT_TURNS = int(os.environ.get("CHAT_MAX_RECENT_TURNS", "4"))
_MIN_TURNS_BEFORE_END = int(os.environ.get("CHAT_MIN_TURNS_BEFORE_END", "8"))
_MIN_CURATED_BEFORE_END = int(os.environ.get("CHAT_MIN_CURATED_BEFORE_END", "1"))
_MAX_EARLY_END_BLOCKS = int(os.environ.get("CHAT_MAX_EARLY_END_BLOCKS", "4"))


@dataclass
class ChatTurnRecord:
    turn_id: int
    decision_state: DecisionState
    student_action: CapabilityAction
    action: Action
    observation_text: str
    episode_done: bool
    metrics: dict[str, Any]


def _action_to_capability(action: Action) -> CapabilityAction:
    names: list[str] = []
    params_list: list[dict[str, Any]] = []
    for tool, params in zip(action.tools, action.params):
        if isinstance(tool, UserTextTool):
            names.append("user_text")
        else:
            names.append(tool.tool_schema.name)
        params_list.append(dict(params) if isinstance(params, dict) else {})
    cap = parse_action_from_tools(names, params_list)
    if cap is not None:
        return cap
    return CapabilityAction(
        action_type=CapabilityActionType.UNKNOWN,
        arguments={"tools": names},
    )


def _short_action(action: Action) -> str:
    parts: list[str] = []
    for tool, params in zip(action.tools, action.params):
        name = (
            "user_text"
            if isinstance(tool, UserTextTool)
            else tool.tool_schema.name
        )
        if not isinstance(params, dict):
            parts.append(name)
            continue
        if name in {"search_corpus", "grep_corpus"}:
            q = str(params.get("query", params.get("pattern", "")))[:160]
            parts.append(f"{name}(query={q!r})")
        elif name == "fan_out_search":
            qs = params.get("queries") or []
            n = len(qs) if isinstance(qs, list) else 0
            parts.append(f"{name}(n_queries={n})")
        elif name in {"curate", "read_document", "verify"}:
            keys = {k: params.get(k) for k in ("add_ids", "doc_ids", "claim") if k in params}
            parts.append(f"{name}({keys})")
        elif name == "end_search":
            parts.append(f"end_search(reasoning={str(params.get('reasoning', ''))[:120]!r})")
        else:
            parts.append(name)
    return "; ".join(parts) if parts else "noop"


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 20] + "\n...[truncated]..."


class ChatDecisionDriver:
    """Run one Search-Agent episode via chat completions + DecisionState export."""

    def __init__(
        self,
        *,
        env: SlidingWindowSearchEnv,
        inference: OpenAIAgentInferenceModel,
        max_turns: int | None = None,
        min_turns_before_end: int | None = None,
        min_curated_before_end: int | None = None,
        robust: bool = True,
    ) -> None:
        self.env = env
        self.inference = inference
        self.max_turns = max_turns or env.max_turns
        self.toolset = env._build_full_toolset()
        self._recent: list[tuple[str, str]] = []  # (action_summary, obs_summary)
        self._min_turns_before_end = (
            min_turns_before_end
            if min_turns_before_end is not None
            else _MIN_TURNS_BEFORE_END
        )
        self._min_curated_before_end = (
            min_curated_before_end
            if min_curated_before_end is not None
            else _MIN_CURATED_BEFORE_END
        )
        self._robust = robust
        self._early_end_blocks = 0
        self._curate_tool = CurateTool()
        self._end_search_tool = EndSearchTool()
        # Ensure repair tools exist in toolset
        self.toolset.tools.setdefault("curate", self._curate_tool)
        self.toolset.tools.setdefault("end_search", self._end_search_tool)

    def _rhythm_nudge(self) -> str:
        turns_since = getattr(self.env, "_turns_since_curate", 0)
        pool = len(self.env.wm.pool_ids)
        curated = len(self.env.wm.curated_ids)
        bits: list[str] = []
        if pool > 0 and turns_since >= 1:
            bits.append(CURATE_NUDGE_PROMPT)
        if curated == 0 and pool > 0:
            bits.append(
                "CRITICAL: curated set is EMPTY while pool has documents. "
                "Your next action MUST be curate(add_ids=[...]) before another search or end_search."
            )
        if self.env._current_turn < self._min_turns_before_end:
            bits.append(
                f"Do NOT call end_search before turn {self._min_turns_before_end}. "
                "Keep exploring distinct query facets."
            )
        bits.append(
            "Use ONLY these tools: fan_out_search, search_corpus, grep_corpus, "
            "read_document, review_docs, curate, verify (if available), end_search. "
            "Never invent a Document tool; never emit <Document> XML — call curate instead."
        )
        return "\n".join(f"- {b}" for b in bits)

    def _build_context_observation(self) -> Observation:
        """Rebuild student-visible context from WM (not raw history dumps)."""
        try:
            wm_text = self.env.wm.to_text()
        except Exception:
            wm_text = (
                f"curated={list(self.env.wm.curated_ids)[:20]}\n"
                f"pool={list(self.env.wm.pool_ids)[:30]}"
            )
        # Progressive shrink when context was overflowing
        wm_budget = _MAX_WM_CHARS
        if self._early_end_blocks >= 2:
            wm_budget = min(wm_budget, 10000)
        wm_text = _clip(wm_text, wm_budget)

        recent_lines: list[str] = []
        for i, (act, obs) in enumerate(self._recent[-_MAX_RECENT_TURNS:], start=1):
            recent_lines.append(f"[turn-{i} action] {act}")
            if obs:
                recent_lines.append(f"[turn-{i} result] {_clip(obs, 600)}")

        summaries = list(getattr(self.env, "_result_summaries", []) or [])[-4:]
        summary_block = "\n".join(summaries) if summaries else "(none yet)"

        body = (
            f"{get_system_prompt(self.env.query_text)}\n\n"
            f"=== WorkingMemory (budgeted) ===\n{wm_text}\n\n"
            f"=== Recent result summaries ===\n{summary_block}\n\n"
            f"=== Recent actions (compact) ===\n"
            + ("\n".join(recent_lines) if recent_lines else "(start of episode)")
            + "\n\n=== Policy reminders ===\n"
            + self._rhythm_nudge()
            + "\n\nChoose the next tool call."
        )
        return Observation(
            observations=[body],
            sources=["user"],
            tool_metadata=[None],
        )

    def _trajectory(self) -> Trajectory:
        return Trajectory.model_construct(
            actions_and_observations=[self._build_context_observation()],
            id=uuid.uuid4(),
        )

    def _normalize_action(self, action: Action | None) -> Action:
        if action is None or not action.tools:
            return Action(
                tools=[self._end_search_tool],
                params=[{"reasoning": "empty model action"}],
                sources=["agent"],
            )

        if not self._robust:
            return action

        # Repair UserText Document dumps before they hard-stop the env
        allow_end = self.env._current_turn >= self._min_turns_before_end
        action = maybe_convert_user_text_to_tools(
            action,
            curate_tool=self._curate_tool,
            end_search_tool=self._end_search_tool,
            allow_end=allow_end,
        )

        # Normalize params on each tool call
        new_tools = []
        new_params = []
        new_sources = list(action.sources) if action.sources else ["agent"] * len(action.tools)
        for tool, params, source in zip(action.tools, action.params, new_sources):
            if isinstance(tool, UserTextTool):
                new_tools.append(tool)
                new_params.append(dict(params) if isinstance(params, dict) else {"text": str(params)})
                continue
            name = tool.tool_schema.name
            norm = normalize_tool_params(
                name, dict(params) if isinstance(params, dict) else {}
            )
            new_tools.append(tool)
            new_params.append(norm)
        while len(new_sources) < len(new_tools):
            new_sources.append("agent")
        return Action(tools=new_tools, params=new_params, sources=new_sources[: len(new_tools)])

    def _is_stop_action(self, action: Action) -> bool:
        for tool in action.tools:
            if isinstance(tool, UserTextTool):
                return True
            if tool.tool_schema.name == "end_search":
                return True
        return False

    def _block_to_curate_or_search(self, reason: str) -> Action:
        """Replace a blocked stop with a productive tool call."""
        pool = list(self.env.wm.pool_ids)
        curated = set(self.env.wm.curated_ids)
        uncurated = [d for d in pool if d not in curated][:12]
        if uncurated:
            return Action(
                tools=[self._curate_tool],
                params=[{"add_ids": uncurated, "remove_ids": []}],
                sources=["early_end_block"],
            )
        # Nothing to curate — force another search facet
        search = self.toolset.get_tool("fan_out_search") or self.toolset.get_tool(
            "search_corpus"
        )
        if search is not None and search.tool_schema.name == "fan_out_search":
            q = self.env.query_text
            facets = [
                " ".join(q.split()[:8]),
                " ".join(q.split()[-8:]),
            ]
            return Action(
                tools=[search],
                params=[{"queries": facets}],
                sources=["early_end_block"],
            )
        if search is not None:
            return Action(
                tools=[search],
                params=[{"query": " ".join(self.env.query_text.split()[:12])}],
                sources=["early_end_block"],
            )
        return Action(
            tools=[self._end_search_tool],
            params=[{"reasoning": reason[:200]}],
            sources=["early_end_block_fallback"],
        )

    async def _infer_action(self) -> Action:
        ctx = InferenceContext(
            trajectory=self._trajectory(),
            toolset=self.toolset,
            max_tokens=self.inference.max_output_tokens,
        )
        try:
            action = await asyncio.to_thread(self.inference, ctx)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # Resilient path: unknown Document tool / bad JSON → salvage
            if self._robust and (
                "unknown tool" in msg.lower()
                or "Invalid JSON" in msg
                or "Document" in msg
            ):
                # Best-effort: ask model again with tighter reminder, else curate nudge
                self._recent.append(
                    (
                        "invalid_tool_call",
                        "Invalid/unknown tool. Use curate/search_corpus/end_search only.",
                    )
                )
                ctx = InferenceContext(
                    trajectory=self._trajectory(),
                    toolset=self.toolset,
                    max_tokens=min(512, self.inference.max_output_tokens),
                )
                try:
                    action = await asyncio.to_thread(self.inference, ctx)
                except Exception:  # noqa: BLE001
                    action = self._block_to_curate_or_search(msg)
            elif "maximum context length" in msg or "input_tokens" in msg:
                self._recent = self._recent[-2:]
                # Shrink WM budget for subsequent turns via early_end_blocks proxy
                self._early_end_blocks = max(self._early_end_blocks, 2)
                ctx = InferenceContext(
                    trajectory=self._trajectory(),
                    toolset=self.toolset,
                    max_tokens=min(512, self.inference.max_output_tokens),
                )
                action = await asyncio.to_thread(self.inference, ctx)
            else:
                raise
        return self._normalize_action(action)

    async def run(
        self,
        *,
        on_critical_turn: Callable[[ChatTurnRecord], None] | None = None,
        pre_step_hook: Callable[[DecisionState, Action], Action] | None = None,
    ) -> dict[str, Any]:
        await self.env.initial_observation()
        self._recent = []
        self._early_end_blocks = 0

        turns: list[ChatTurnRecord] = []
        done = False
        for _ in range(self.max_turns):
            state = self.env.export_decision_state()
            action = await self._infer_action()

            # Premature-stop guard (v2)
            if self._robust and self._is_stop_action(action):
                block, reason = should_block_early_end(
                    turn=self.env._current_turn,
                    n_curated=len(self.env.wm.curated_ids),
                    n_pool=len(self.env.wm.pool_ids),
                    min_turns=self._min_turns_before_end,
                    min_curated=self._min_curated_before_end,
                )
                if block and self._early_end_blocks < _MAX_EARLY_END_BLOCKS:
                    self._early_end_blocks += 1
                    self._recent.append(("blocked_end_search", reason))
                    action = self._block_to_curate_or_search(reason)

            if pre_step_hook is not None:
                action = pre_step_hook(state, action)

            cap = _action_to_capability(action)
            rec = ChatTurnRecord(
                turn_id=state.turn_id,
                decision_state=state,
                student_action=cap,
                action=action,
                observation_text="",
                episode_done=False,
                metrics={},
            )

            step = await self.env.step_from_action(action)
            rec.episode_done = bool(step.episode_done)
            rec.metrics = dict(step.metrics or {})

            obs_text = ""
            if not step.episode_done and self.env._all_observations:
                last_obs = self.env._all_observations[-1]
                obs_text = "\n".join(last_obs.observations)
            elif step.episode_done:
                obs_text = "episode_done"

            self._recent.append((_short_action(action), _clip(obs_text, _MAX_OBS_CHARS)))
            rec.observation_text = _clip(obs_text, 4000)

            if on_critical_turn is not None:
                on_critical_turn(rec)
            turns.append(rec)

            if step.episode_done:
                done = True
                break

        if not done:
            self.env._terminal_reward, self.env._terminal_metrics = (
                self.env._compute_terminal_reward()
            )
            self.env._terminal_metrics["max_turns_reached"] = 1.0

        return {
            "query_id": self.env.query_id,
            "turns": len(turns),
            "n_curated": len(self.env.wm.curated_ids),
            "n_pool": len(self.env.wm.pool_ids),
            "recall": float(self.env._terminal_metrics.get("recall", 0.0)),
            "trajectory_recall": float(
                self.env._terminal_metrics.get("trajectory_recall", 0.0)
            ),
            "final_answer_recall": float(
                self.env._terminal_metrics.get("final_answer_recall", 0.0)
            ),
            "precision": float(self.env._terminal_metrics.get("precision", 0.0)),
            "reward": float(self.env._terminal_reward),
            "error": self.env._terminal_metrics.get("no_error", 1.0) == 0.0,
            "metrics": {
                k: v
                for k, v in self.env._terminal_metrics.items()
                if isinstance(v, (int, float, str, bool))
            },
            "turn_records": turns,
            "early_end_blocks": self._early_end_blocks,
            "driver": "ultra_chat_v2" if self._robust else "ultra_chat",
        }
