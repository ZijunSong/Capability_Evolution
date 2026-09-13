"""Load original SlidingWindowSearchEnv after V8D flags are already set.

Importing this module does not import ultra_core. Call ``load_upstream_env``
only inside an isolated worker process.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.upstream_harness1.api_adapter import chat_tools_from_upstream_schemas
from trim.upstream_harness1.pin import HARNESS1_ROOT, ensure_harness1_on_path
from trim.upstream_harness1.retrieval import (
    RETRIEVAL_LOCAL_BM25,
    RETRIEVAL_UPSTREAM,
    RetrievalConfig,
    canonical_dataset_name,
)


def load_upstream_modules() -> dict[str, Any]:
    """Import original env / tools. Caller must have set V8D_* first."""
    ensure_harness1_on_path()
    from harness.tools import (  # type: ignore[import-not-found]
        GREP_CORPUS_SCHEMA,
        READ_DOCUMENT_SCHEMA,
        SEARCH_CORPUS_SCHEMA,
        GrepCorpusTool,
        GrepCorpusToolCallMetadata,
        PruneChunksTool,
        ReadDocumentTool,
        SearchCorpusTool,
        SearchCorpusToolCallMetadata,
        Tool,
        ToolSet,
        UserTextTool,
    )
    from harness.trajectory import ActionBuilder, Observation, Trajectory  # type: ignore[import-not-found]
    from harness.utils import ProviderFormat  # type: ignore[import-not-found]
    from training.train_rl import SlidingWindowSearchEnv  # type: ignore[import-not-found]

    return {
        "SlidingWindowSearchEnv": SlidingWindowSearchEnv,
        "Tool": Tool,
        "ToolSet": ToolSet,
        "SearchCorpusTool": SearchCorpusTool,
        "GrepCorpusTool": GrepCorpusTool,
        "ReadDocumentTool": ReadDocumentTool,
        "PruneChunksTool": PruneChunksTool,
        "UserTextTool": UserTextTool,
        "SearchCorpusToolCallMetadata": SearchCorpusToolCallMetadata,
        "GrepCorpusToolCallMetadata": GrepCorpusToolCallMetadata,
        "SEARCH_CORPUS_SCHEMA": SEARCH_CORPUS_SCHEMA,
        "GREP_CORPUS_SCHEMA": GREP_CORPUS_SCHEMA,
        "READ_DOCUMENT_SCHEMA": READ_DOCUMENT_SCHEMA,
        "ActionBuilder": ActionBuilder,
        "Observation": Observation,
        "Trajectory": Trajectory,
        "ProviderFormat": ProviderFormat,
        "root": HARNESS1_ROOT,
    }


def ensure_browsecompplus_paths(*, bcp_root: Path | None = None) -> dict[str, str]:
    """Populate upstream harness Config paths for BrowseComp+ when unset."""
    if bcp_root is None:
        from trim.eval.official_query_pool import default_bcp_root

        bcp_root = default_bcp_root()
    if bcp_root is None:
        return {}
    candidates = {
        "BROWSECOMPPLUS_QRELS_GOLD_PATH": bcp_root / "topics-qrels" / "qrel_golds.txt",
        "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH": bcp_root / "topics-qrels" / "qrel_evidence.txt",
        "BROWSECOMPPLUS_QUERIES_PATH": bcp_root / "topics-qrels" / "queries.tsv",
        "BROWSECOMPPLUS_ANSWERS_PATH": bcp_root / "data" / "browsecomp_plus_decrypted.jsonl",
    }
    applied: dict[str, str] = {}
    for env_key, path in candidates.items():
        if os.environ.get(env_key):
            continue
        if path.is_file():
            os.environ[env_key] = str(path)
            applied[env_key] = str(path)
    return applied


def load_scoring_dataset(name: str) -> Any:
    ensure_harness1_on_path()
    ensure_browsecompplus_paths()
    from datagen.search_dataset import get_dataset  # type: ignore[import-not-found]

    return get_dataset(canonical_dataset_name(name))


@dataclass
class EnvToolPack:
    toolset: Any
    search_tool: Any
    dataset: Any
    verifier_client: Any | None = None
    capability_log: dict[str, Any] = field(default_factory=dict)
    backend: str = RETRIEVAL_UPSTREAM
    token_counter: Any | None = None


def build_eval_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any,
    mask: Mapping[str, bool] | None = None,
    token_counter: Any | None = None,
) -> EnvToolPack:
    from trim.upstream_harness1.token_count import TOKEN_COUNT_MODE, whitespace_token_counter

    counter = token_counter or whitespace_token_counter
    retrieval.assert_ready()
    if retrieval.backend == RETRIEVAL_LOCAL_BM25:
        from trim.local_backend.factory import build_local_toolset

        pack = build_local_toolset(
            mods, retrieval, dataset=dataset, token_counter=counter, mask=mask
        )
        if mask and mask.get("verify_tool") and pack.verifier_client is None:
            raise RuntimeError(
                "component includes verify_tool; local_bm25 requires --verify-base-url "
                "pointing at a local verifier. Do not stub verify with overlap or a fixed verdict."
            )
        return EnvToolPack(
            toolset=pack.toolset,
            search_tool=pack.search_tool,
            dataset=pack.dataset_wrapper if pack.dataset_wrapper is not None else dataset,
            verifier_client=pack.verifier_client,
            capability_log=pack.capability_log,
            backend=RETRIEVAL_LOCAL_BM25,
            token_counter=counter,
        )
    toolset, search_tool = build_upstream_toolset(
        mods, retrieval, dataset=dataset, token_counter=counter
    )
    return EnvToolPack(
        toolset=toolset,
        search_tool=search_tool,
        dataset=dataset,
        verifier_client=None,
        capability_log={
            "retrieval_backend": RETRIEVAL_UPSTREAM,
            "chroma_initialized": True,
            "token_count_mode": TOKEN_COUNT_MODE,
        },
        backend=RETRIEVAL_UPSTREAM,
        token_counter=counter,
    )


def build_upstream_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any,
    token_counter: Any | None = None,
) -> tuple[Any, Any]:
    if retrieval.backend != RETRIEVAL_UPSTREAM:
        raise RuntimeError(
            f"build_upstream_toolset requires {RETRIEVAL_UPSTREAM}; got {retrieval.backend}"
        )
    from harness.config import get_config  # type: ignore[import-not-found]

    config = get_config()
    collection_names = dataset.get_chroma_collections(split=retrieval.collection_split)
    chroma_client = config.get_chroma_client()
    openai_client = config.get_openai_client()
    reranker = None
    if retrieval.reranker != "none":
        try:
            if retrieval.reranker == "vllm":
                from harness.rerank import VLLMQwen3Reranker  # type: ignore[import-not-found]

                reranker = VLLMQwen3Reranker(max_tokens=retrieval.read_max_tokens)
            else:
                from harness.rerank import BasetenReranker  # type: ignore[import-not-found]

                reranker = BasetenReranker(max_tokens=retrieval.read_max_tokens)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"upstream reranker {retrieval.reranker!r} is not available: {exc}. "
                "Do not silently disable it for the official baseline."
            ) from exc
    SearchCorpusTool = mods["SearchCorpusTool"]
    search_tool = SearchCorpusTool(
        chroma_client=chroma_client,
        openai_client=openai_client,
        chroma_collection_name=collection_names,
        reranker=reranker,
        snippet_max_chars=retrieval.snippet_max_chars,
        knn_limit=retrieval.search_knn_limit,
        search_limit=retrieval.search_limit,
        display_limit=retrieval.search_display_limit,
    )
    toolset = mods["ToolSet"](name=f"{retrieval.dataset}_toolset")
    toolset.add_tool(search_tool)
    toolset.add_tool(
        mods["GrepCorpusTool"](
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
        )
    )
    toolset.add_tool(
        mods["ReadDocumentTool"](
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            token_counter=token_counter,
            max_tokens=retrieval.read_max_tokens,
        )
    )
    toolset.add_tool(mods["PruneChunksTool"]())
    return toolset, search_tool


def openai_tools_from_env(env: Any, mods: Mapping[str, Any]) -> list[dict[str, Any]]:
    ProviderFormat = mods["ProviderFormat"]
    full = env._build_full_toolset()
    # Chat Completions nested function schema (qwen/moonshot), not Responses flat form.
    raw = full.get_formats(ProviderFormat.QWEN_MOONSHOT)
    return chat_tools_from_upstream_schemas(raw)


# Qwen / generic Chat Completions retry (no Harmony channel requirements).
QWEN_FORMAT_RETRY_PROMPT = (
    "Your previous response could not be parsed as a valid tool call. "
    "Please output a valid function call using the chat.completions tools interface "
    "(JSON arguments on a named tool). "
    "Put any reasoning in the assistant message, then call a tool."
)

# GPT-OSS / Harness-1 Harmony-native retry: allow analysis channel, forbid API wrapper talk.
GPT_OSS_FORMAT_RETRY_PROMPT = (
    "Continue the current retrieval task. Use one of the provided functions with valid JSON "
    "arguments. Put analysis in the analysis channel and function calls in the commentary "
    "channel. When the retrieval task is complete, call end_search with a brief reason."
)

# Backward-compatible alias used by existing tests.
API_FORMAT_RETRY_PROMPT = QWEN_FORMAT_RETRY_PROMPT


def is_harmony_chat_model(model: str | None) -> bool:
    name = str(model or "").lower()
    return "gpt-oss" in name or "harness-1" in name or name.startswith("openai/gpt-oss")


def _format_failed_attempt_context(failed: Mapping[str, Any]) -> str:
    """Summarize a rejected tool batch so the model can fix it without harness replay."""
    lines = [
        "Your previous response was rejected before any tool executed.",
        f"Error category: {failed.get('error_kind') or 'unknown'}",
    ]
    detail = str(failed.get("error_detail") or "").strip()
    if detail:
        lines.append(f"Reason: {detail}")
    for call in failed.get("tool_calls") or []:
        name = str(call.get("name") or "tool")
        args = call.get("arguments") or {}
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except TypeError:
            args_text = repr(args)
        if len(args_text) > 400:
            args_text = args_text[:400] + "...(truncated)"
        lines.append(f"- Failed request: {name}({args_text})")
    lines.append(
        "Fix that exact request, or explicitly choose a different valid tool. "
        "The harness will not silently replay the failed operation."
    )
    return "\n".join(lines)


def format_retry_prompt(
    *,
    model: str | None = None,
    error_hint: str | None = None,
    failed_attempt: Mapping[str, Any] | None = None,
) -> str:
    base = GPT_OSS_FORMAT_RETRY_PROMPT if is_harmony_chat_model(model) else QWEN_FORMAT_RETRY_PROMPT
    parts = [base]
    if failed_attempt:
        parts.append(_format_failed_attempt_context(failed_attempt))
    if error_hint:
        parts.append(f"Specific issue: {error_hint}")
    return "\n\n".join(parts)
DEFAULT_CURATE_NUDGE_PROMPT = (
    "IMPORTANT: You just searched without curating. Follow the search → curate rhythm: "
    "review the results from your last search and call curate NOW to add ALL plausibly "
    "relevant documents. Do not search again until you've curated."
)
DEFAULT_MAX_OBS_CHARS = 15000
DEFAULT_PROMPT_TOKEN_BUDGET = 30720
# Reserve extra tokens: whitespace/heuristic counters often undercount vs served tokenizer.
PROMPT_BUDGET_SAFETY_MARGIN = 512
DEFAULT_CURATE_NUDGE_INTERVAL = 1


_DOC_HEADER_RE = re.compile(r"# DOCUMENT ID:\s*\S+")


def clip_observation_text(text: str, max_chars: int = DEFAULT_MAX_OBS_CHARS) -> str:
    raw = str(text or "")
    limit = int(max_chars)
    if len(raw) <= limit:
        return raw

    matches = list(_DOC_HEADER_RE.finditer(raw))
    if len(matches) >= 2:
        preamble = raw[: matches[0].start()]
        blocks: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            blocks.append(raw[start:end])
        kept: list[str] = []
        running = len(preamble)
        for block in blocks:
            if running + len(block) <= limit:
                kept.append(block)
                running += len(block)
                continue
            break
        if kept:
            clipped = (preamble + "".join(kept)).rstrip()
            hidden_docs = len(blocks) - len(kept)
            suffix = f"\n... (truncated, {len(raw)} chars total"
            if hidden_docs:
                suffix += f"; {hidden_docs} more document(s) not shown"
            suffix += ")"
            return clipped + suffix
        if blocks:
            suffix_reserve = 72
            budget = max(0, limit - len(preamble) - suffix_reserve)
            if budget > 0:
                first_prefix = blocks[0][:budget]
                clipped = (preamble + first_prefix).rstrip()
                hidden_docs = max(0, len(blocks) - 1)
                suffix = f"\n... (truncated, {len(raw)} chars total"
                if hidden_docs or len(blocks[0]) > budget:
                    suffix += "; remaining document(s) not shown"
                suffix += ")"
                return clipped + suffix

    return raw[:limit] + f"\n... (truncated, {len(raw)} chars total)"


def flatten_message_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        for key in ("reasoning", "reasoning_content"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
        elif content is not None:
            parts.append(str(content))
        for call in msg.get("tool_calls") or []:
            parts.append(json.dumps(call, ensure_ascii=False))
    return "\n".join(parts)


def trajectory_to_chat_messages(traj: Any, *, include_reasoning: bool) -> list[dict[str, Any]]:
    """OpenAI chat messages from a Trajectory, optionally preserving action reasoning."""
    UserTextTool = None
    try:
        from harness.tools import UserTextTool as _UserTextTool  # type: ignore[import-not-found]

        UserTextTool = _UserTextTool
    except Exception:
        pass

    messages: list[dict[str, Any]] = []
    for action_or_observation in traj.actions_and_observations:
        if hasattr(action_or_observation, "as_iter"):
            action = action_or_observation
            assistant_message: dict[str, Any] = {"role": "assistant", "content": ""}
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for tool, params, source in action.as_iter():
                is_user_text = UserTextTool is not None and isinstance(tool, UserTextTool)
                if is_user_text:
                    text_parts.append(str((params or {}).get("text") or ""))
                else:
                    tool_calls.append(
                        {
                            "id": str(source),
                            "type": "function",
                            "function": {
                                "name": tool.tool_schema.name,
                                "arguments": json.dumps(params or {}),
                            },
                        }
                    )
            if text_parts:
                assistant_message["content"] = "\n".join(text_parts)
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            if include_reasoning and getattr(action, "reasoning", None):
                assistant_message["reasoning"] = str(action.reasoning)
            messages.append(assistant_message)
        else:
            observation = action_or_observation
            for observation_text, source in zip(
                getattr(observation, "observations", []) or [],
                getattr(observation, "sources", []) or [],
            ):
                if source == "user":
                    messages.append({"role": "user", "content": str(observation_text or "")})
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(source),
                            "content": str(observation_text or ""),
                        }
                    )
    return messages


def _curate_nudge_policy() -> str:
    policy = str(os.environ.get("CURATE_NUDGE_POLICY") or "legacy").strip().lower()
    if policy not in {"legacy", "state_change"}:
        return "legacy"
    return policy


def _nudge_prompt_for_env(env: Any) -> str | None:
    interval = DEFAULT_CURATE_NUDGE_INTERVAL
    nudge_text = DEFAULT_CURATE_NUDGE_PROMPT
    try:
        from harness.ultra_core import CURATE_NUDGE_INTERVAL, CURATE_NUDGE_PROMPT  # type: ignore[import-not-found]

        interval = int(CURATE_NUDGE_INTERVAL)
        nudge_text = str(CURATE_NUDGE_PROMPT)
    except Exception:
        pass
    turns_since = int(getattr(env, "_turns_since_curate", 0) or 0)
    pool = 0
    wm = getattr(env, "wm", None)
    if wm is not None and hasattr(wm, "get_pool_size"):
        try:
            pool = int(wm.get_pool_size())
        except Exception:
            pool = 0
    if turns_since < interval or pool <= 0:
        return None
    policy = _curate_nudge_policy()
    if policy == "state_change":
        baseline = int(getattr(env, "_pool_size_at_last_curate", 0) or 0)
        pending_verify = bool(getattr(env, "_pending_verify_since_curate", False))
        if pool <= baseline and not pending_verify:
            return None
        return (
            "You have new retrieval or verification results since your last curate. "
            "Review them and call curate if any documents should be added, removed, or retagged."
        )
    return nudge_text


def _clip_observation(obs: Any, Observation: Any, max_chars: int) -> Any:
    texts = [clip_observation_text(t, max_chars) for t in list(getattr(obs, "observations", None) or [])]
    sources = list(getattr(obs, "sources", None) or [])
    meta = list(getattr(obs, "tool_metadata", None) or [None] * len(texts))
    try:
        return Observation(observations=texts, sources=sources, tool_metadata=meta)
    except Exception:
        return obs


def _truncate_wm_pool(wm_text: str | None, *, chars_to_cut: int) -> str | None:
    raw = str(wm_text or "")
    if not raw or chars_to_cut <= 0:
        return wm_text
    pool_start = raw.find("Document Pool:")
    hist_start = raw.find("Search History:")
    if pool_start <= 0 or hist_start <= pool_start:
        if len(raw) <= chars_to_cut:
            return raw[: max(0, len(raw) // 2)] + "\n...(WM truncated)\n"
        return raw[: max(0, len(raw) - chars_to_cut)] + "\n...(WM truncated)\n"
    pool_section = raw[pool_start:hist_start]
    cut = min(len(pool_section) - 100, chars_to_cut)
    if cut <= 0:
        return wm_text
    new_pool = pool_section[: len(pool_section) - cut] + "\n  ... (truncated for context)\n\n"
    return raw[:pool_start] + new_pool + raw[hist_start:]


def _message_token_count(messages: Sequence[Mapping[str, Any]], counter) -> int:
    return int(counter(flatten_message_text(messages)))


def _raise_over_budget(*, token_count: int, budget: int, stage: str) -> None:
    raise RuntimeError(
        f"API prompt still exceeds budget after {stage}: {token_count} > {budget}. "
        "Refuse to send a known over-budget request."
    )


def openai_messages_from_env(
    env: Any,
    mods: Mapping[str, Any],
    *,
    retry: bool = False,
    retry_error: str | None = None,
    model: str | None = None,
    token_counter: Any | None = None,
    max_obs_chars: int | None = None,
    prompt_token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Messages from the original selected window, plus original nudge/retry/clip rules."""
    from trim.upstream_harness1.token_count import whitespace_token_counter

    window = env.selected_context_window()
    Trajectory = mods["Trajectory"]
    Observation = mods["Observation"]
    import uuid

    try:
        from harness.ultra_core import MAX_OBS_CHARS, PROMPT_TOKEN_BUDGET  # type: ignore[import-not-found]

        obs_limit = int(max_obs_chars if max_obs_chars is not None else MAX_OBS_CHARS)
        budget = int(prompt_token_budget if prompt_token_budget is not None else PROMPT_TOKEN_BUDGET)
    except Exception:
        obs_limit = int(max_obs_chars if max_obs_chars is not None else DEFAULT_MAX_OBS_CHARS)
        budget = int(prompt_token_budget if prompt_token_budget is not None else DEFAULT_PROMPT_TOKEN_BUDGET)
    budget = max(1024, int(budget) - PROMPT_BUDGET_SAFETY_MARGIN)

    counter = token_counter or getattr(env, "text_token_counter", None) or whitespace_token_counter
    actions = list(window.get("recent_actions") or [])
    observations = list(window.get("recent_observations") or [])
    summaries = list(window.get("result_summaries") or [])
    extra: list[str] = []
    if not retry:
        nudge = _nudge_prompt_for_env(env)
        if nudge:
            extra.append(nudge)
    else:
        failed_attempt = getattr(env, "_pending_failed_attempt", None)
        extra.append(
            format_retry_prompt(
                model=model,
                error_hint=retry_error,
                failed_attempt=failed_attempt if isinstance(failed_attempt, Mapping) else None,
            )
        )

    wm_text = window.get("wm_text")
    obs_limit = int(obs_limit)
    tight = max(2000, obs_limit // 3)

    def _assemble_current(
        *,
        wm: str | None,
        act: list[Any],
        obs: list[Any],
        sums: list[Any],
        obs_chars: int,
    ) -> list[dict[str, Any]]:
        clipped_obs = [_clip_observation(o, Observation, obs_chars) for o in obs]
        entries: list[Any] = [
            Observation(observations=[env.system_prompt], sources=["user"], tool_metadata=[None])
        ]
        if wm:
            entries.append(Observation(observations=[wm], sources=["user"], tool_metadata=[None]))
        n = len(act)
        for i, (action, item) in enumerate(zip(act, clipped_obs)):
            entries.append(action)
            entries.append(item)
            if i < n - 1 and i < len(sums) and sums[i]:
                entries.append(
                    Observation(observations=[sums[i]], sources=["user"], tool_metadata=[None])
                )
        traj = Trajectory(actions_and_observations=entries, id=uuid.uuid4())
        include_reasoning = is_harmony_chat_model(model)
        messages = trajectory_to_chat_messages(traj, include_reasoning=include_reasoning)
        for text in extra:
            messages.append({"role": "user", "content": text})
        return messages

    messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 1: drop oldest recent turns.
    while _message_token_count(messages, counter) > budget and len(actions) > 1:
        actions = actions[1:]
        observations = observations[1:]
        if summaries:
            summaries = summaries[1:]
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 2: truncate WM pool section.
    if _message_token_count(messages, counter) > budget and wm_text:
        overshoot = _message_token_count(messages, counter) - budget
        wm_text = _truncate_wm_pool(wm_text, chars_to_cut=min(len(str(wm_text)), overshoot * 3))
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 3: aggressive WM cap + tighter observation clip.
    if _message_token_count(messages, counter) > budget:
        if wm_text and len(str(wm_text)) > 2000:
            wm_text = str(wm_text)[:2000] + "\n...(WM truncated)\n"
        tight = max(2000, obs_limit // 3)
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=tight)

    # Pass 4: shrink remaining observations until within budget or floor reached.
    floor = 512
    step = max(256, obs_limit // 8)
    if _message_token_count(messages, counter) > budget:
        current_obs_limit = tight
        while _message_token_count(messages, counter) > budget and current_obs_limit > floor:
            current_obs_limit = max(floor, current_obs_limit - step)
            messages = _assemble_current(
                wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=current_obs_limit
            )

    # Pass 5: minimal context (system prompt + tail only).
    if _message_token_count(messages, counter) > budget:
        minimal = [{"role": "user", "content": str(env.system_prompt or "")}]
        for text in extra:
            minimal.append({"role": "user", "content": text})
        messages = minimal

    if _message_token_count(messages, counter) > budget:
        _raise_over_budget(token_count=_message_token_count(messages, counter), budget=budget, stage="minimal_context")

    return messages


def _allowed_tool_names(env: Any) -> set[str]:
    toolset = env._build_full_toolset()
    tools = getattr(toolset, "tools", None) or {}
    # ToolSet.tools is a name -> Tool dict; iterating the dict yields keys only.
    if isinstance(tools, dict):
        return {str(name) for name in tools.keys()}
    names: set[str] = set()
    for tool in tools:
        schema = getattr(tool, "tool_schema", None)
        if schema is not None and getattr(schema, "name", None):
            names.add(str(schema.name))
    return names


class SchemaValidationError(ValueError):
    """Structured schema validation failure before any tool side effect."""

    category = "schema_error"


def _validate_json_value(
    val: Any,
    spec: Mapping[str, Any],
    *,
    tool_name: str,
    path: str,
) -> None:
    expected = (spec or {}).get("type")
    if expected == "array":
        if not isinstance(val, list):
            raise SchemaValidationError(f"Tool {tool_name} parameter {path} expected array")
        item_spec = (spec or {}).get("items") or {}
        for idx, item in enumerate(val):
            _validate_json_value(item, item_spec, tool_name=tool_name, path=f"{path}[{idx}]")
        return
    if expected == "object":
        if not isinstance(val, dict):
            raise SchemaValidationError(f"Tool {tool_name} parameter {path} expected object")
        props = (spec or {}).get("properties") or {}
        for key, child in props.items():
            if key in val and val[key] is not None:
                _validate_json_value(val[key], child or {}, tool_name=tool_name, path=f"{path}.{key}")
        return
    if expected == "string" and not isinstance(val, str):
        raise SchemaValidationError(f"Tool {tool_name} parameter {path} expected string")
    if expected in {"integer", "number"} and not isinstance(val, (int, float)):
        raise SchemaValidationError(f"Tool {tool_name} parameter {path} expected number")
    enum = (spec or {}).get("enum")
    if enum and val not in enum:
        raise SchemaValidationError(
            f"Tool {tool_name} parameter {path} must be one of {list(enum)}"
        )


def _validate_tool_params(tool: Any, params: Mapping[str, Any]) -> None:
    schema = getattr(tool, "tool_schema", None)
    if schema is None:
        return
    required = list(getattr(schema, "required", None) or [])
    name = str(getattr(schema, "name", "") or "tool")
    for key in required:
        if key not in params:
            raise SchemaValidationError(f"Tool {name} missing required parameter: {key}")
        if params[key] is None:
            raise SchemaValidationError(f"Tool {name} parameter {key} must not be null")
    props = getattr(schema, "parameters", None) or {}
    for key, spec in props.items():
        if key not in params:
            continue
        if params[key] is None:
            raise SchemaValidationError(f"Tool {name} parameter {key} must not be null")
        _validate_json_value(params[key], spec or {}, tool_name=name, path=key)


def action_from_parsed(parsed: Any, env: Any, mods: Mapping[str, Any]) -> Any:
    ActionBuilder = mods["ActionBuilder"]
    UserTextTool = mods["UserTextTool"]
    toolset = env._build_full_toolset()
    allowed = _allowed_tool_names(env)
    if parsed.parse_error:
        raise ValueError(parsed.parse_error)
    if getattr(parsed, "protocol_error", None):
        raise ValueError(str(parsed.protocol_error))

    pending: list[tuple[str, dict[str, Any], str]] = []
    for call in parsed.tool_calls:
        name = str(call.get("name") or "")
        params = dict(call.get("arguments") or {})
        source = str(call.get("id") or "agent")
        if name in {"user_text", "UserTextTool"}:
            pending.append(("__user_text__", params if "text" in params else {"text": json.dumps(params)}, source))
            continue
        if name not in allowed:
            raise ValueError(f"Model requested unknown tool or tool not in toolset: {name}")
        if not isinstance(params, dict):
            raise ValueError(f"Tool {name} arguments must be a JSON object")
        pending.append((name, params, source))

    builder = ActionBuilder()
    if parsed.reasoning:
        builder.add_reasoning(parsed.reasoning)
    for name, params, source in pending:
        if name == "__user_text__":
            builder.add_tool_call(UserTextTool(), params, source)
            continue
        tool = toolset.get_tool(name)
        if tool is None:
            raise ValueError(f"Model requested unknown tool or tool not in toolset: {name}")
        _validate_tool_params(tool, params)
        builder.add_tool_call(tool, params, source)
    return builder.build()
