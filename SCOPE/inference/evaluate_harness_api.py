"""Evaluate Harness search episodes via OpenAI-compatible chat APIs.

Default (v2): Ultra WorkingMemory harness via ChatDecisionDriver +
SlidingWindowSearchEnv — the same stateful search loop used by SCOPE/RL.

Legacy: TokenBudgetRetrievalSubagent + Document-XML prompt (set
USE_LEGACY_API_AGENT=1). That path was the source of the BrowseComp full-run
collapse (recall≈1%): wrong tool surface, hard token-budget rejections, and
no curate/end_search rhythm.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Dict

import structlog
import tiktoken

from datagen.search_dataset import SearchDataset
from harness.agent import OpenAIAgentInferenceModel, TokenBudgetRetrievalSubagent
from harness.llm_env import get_llm_client, get_llm_model_name
from harness.prompts import get_retrieval_subagent_prompt
from harness.tasks import SearchTaskEvaluationOutput, SearchTaskOutput
from harness.tools import ToolSet
from harness.trajectory import Observation
from training.chat_decision_driver import ChatDecisionDriver
from training.train_rl import SlidingWindowSearchEnv

logger = structlog.get_logger("evaluate_harness_api")

USE_LEGACY_API_AGENT = os.environ.get("USE_LEGACY_API_AGENT", "0") == "1"


def _default_token_counter() -> Callable[[Any], int]:
    enc = tiktoken.get_encoding("o200k_harmony")

    def counter(trajectory) -> int:
        # Rough budget estimate for API models without Harmony rendering.
        parts: list[str] = []
        for item in trajectory.actions_and_observations:
            if hasattr(item, "observations"):
                parts.extend(item.observations)
            elif hasattr(item, "tools"):
                parts.extend(str(p) for p in item.params)
        return len(enc.encode("\n".join(parts)))

    return counter


def build_api_agent(
    toolset: ToolSet,
    *,
    max_tokens: int,
    temperature: float,
    max_trajectory_length: int = 64,
) -> TokenBudgetRetrievalSubagent:
    """Legacy TokenBudget agent (Document-XML retrieval subagent)."""
    client = get_llm_client()
    model = get_llm_model_name()
    inference_model = OpenAIAgentInferenceModel(
        openai_client=client,
        model=model,
        max_output_tokens=max_tokens,
        temperature=temperature,
        api_style="chat_completions",
    )
    text_counter = lambda text: len(tiktoken.get_encoding("o200k_harmony").encode(text))
    return TokenBudgetRetrievalSubagent(
        toolset=toolset,
        inference_model=inference_model,
        token_counter=_default_token_counter(),
        text_token_counter=text_counter,
        max_trajectory_length=max_trajectory_length,
    )


def _eval_single_query_legacy_sync(
    qid: str,
    dataset: SearchDataset,
    toolset: ToolSet,
    *,
    max_tokens: int,
    temperature: float,
    max_trajectory_length: int,
) -> Dict[str, Any]:
    _, query_text = dataset.get_query_by_id(qid)
    agent = build_api_agent(
        toolset,
        max_tokens=max_tokens,
        temperature=temperature,
        max_trajectory_length=max_trajectory_length,
    )
    prompt = get_retrieval_subagent_prompt(query_text)
    initial_observation = Observation(
        observations=[prompt],
        sources=["user"],
        tool_metadata=[None],
    )
    start = time.time()
    trajectory = agent(initial_observation=initial_observation)
    elapsed = time.time() - start

    output = SearchTaskOutput(
        trajectory=trajectory,
        query_id=qid,
        dataset_name=dataset.name,
    )
    eval_output = SearchTaskEvaluationOutput.from_search_task_output(output, dataset)
    return {
        "query_id": qid,
        "query": query_text[:80],
        "recall": eval_output.recall or 0.0,
        "trajectory_recall": eval_output.trajectory_recall or 0.0,
        "final_answer_recall": eval_output.final_answer_recall or 0.0,
        "precision": eval_output.precision or 0.0,
        "reward": eval_output.recall or 0.0,
        "turns": eval_output.num_turns or 0,
        "n_curated": len(output.get_unique_output_chunk_ids()),
        "n_pool": len(output.get_unique_traversed_chunk_ids()),
        "elapsed_s": round(elapsed, 1),
        "error": eval_output.error is not None,
        "error_message": eval_output.error,
        "policy": "api",
        "model": get_llm_model_name(),
        "driver": "legacy_token_budget",
    }


async def _eval_single_query_ultra(
    qid: str,
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool,
    text_token_counter,
    *,
    max_tokens: int,
    temperature: float,
    max_trajectory_length: int,
) -> Dict[str, Any]:
    _, query_text = dataset.get_query_by_id(qid)
    client = get_llm_client()
    model = get_llm_model_name()
    inference = OpenAIAgentInferenceModel(
        openai_client=client,
        model=model,
        max_output_tokens=max_tokens,
        temperature=temperature,
        api_style="chat_completions",
    )
    env = SlidingWindowSearchEnv(
        toolset=toolset,
        search_tool=search_tool,
        query_id=qid,
        query_text=query_text,
        dataset=dataset,
        text_token_counter=text_token_counter,
        max_turns=max_trajectory_length,
    )
    driver = ChatDecisionDriver(
        env=env,
        inference=inference,
        max_turns=max_trajectory_length,
        robust=True,
    )
    start = time.time()
    result = await driver.run()
    elapsed = time.time() - start
    return {
        "query_id": qid,
        "query": query_text[:80],
        "recall": float(result.get("recall", 0.0)),
        "trajectory_recall": float(result.get("trajectory_recall", 0.0)),
        "final_answer_recall": float(result.get("final_answer_recall", 0.0)),
        "precision": float(result.get("precision", 0.0)),
        "reward": float(result.get("reward", result.get("recall", 0.0))),
        "turns": int(result.get("turns", 0)),
        "n_curated": int(result.get("n_curated", 0)),
        "n_pool": int(result.get("n_pool", 0)),
        "elapsed_s": round(elapsed, 1),
        "error": bool(result.get("error", False)),
        "error_message": None,
        "policy": "api",
        "model": model,
        "driver": result.get("driver", "ultra_chat_v2"),
        "early_end_blocks": int(result.get("early_end_blocks", 0)),
    }


async def eval_single_query(
    qid: str,
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool,
    text_token_counter,
    *,
    max_tokens: int,
    temperature: float,
    max_trajectory_length: int = 64,
) -> Dict[str, Any]:
    try:
        if USE_LEGACY_API_AGENT:
            result = await asyncio.to_thread(
                _eval_single_query_legacy_sync,
                qid,
                dataset,
                toolset,
                max_tokens=max_tokens,
                temperature=temperature,
                max_trajectory_length=max_trajectory_length,
            )
        else:
            result = await _eval_single_query_ultra(
                qid,
                dataset,
                toolset,
                search_tool,
                text_token_counter,
                max_tokens=max_tokens,
                temperature=temperature,
                max_trajectory_length=max_trajectory_length,
            )
        logger.info(
            "api_episode_result",
            qid=qid,
            recall=round(result.get("recall", 0), 3),
            turns=result.get("turns", 0),
            error=result.get("error", False),
            driver=result.get("driver"),
        )
        return result
    except Exception as exc:
        logger.error("api_episode_failed", qid=qid, error=str(exc)[:500])
        return {
            "query_id": qid,
            "error": True,
            "error_message": str(exc)[:500],
            "reward": 0,
            "recall": 0,
            "trajectory_recall": 0,
            "final_answer_recall": 0,
            "precision": 0,
            "n_curated": 0,
            "n_pool": 0,
            "turns": 0,
            "policy": "api",
            "driver": "legacy_token_budget" if USE_LEGACY_API_AGENT else "ultra_chat_v2",
        }
