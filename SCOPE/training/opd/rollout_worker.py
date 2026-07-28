"""BrowseComp-backed rollout worker for OPD data collection."""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
import tiktoken
import tinker

from datagen.search_dataset import SearchDataset, get_dataset
from harness.harness_config import HarnessConfig, config_path, load_harness_config
from harness.tools import UserTextTool
from harness.views.student_view import StudentView
from harness.views.teacher_view import TeacherView
from training.opd.env_factory import RolloutRuntime, build_rollout_runtime, build_search_env
from training.opd._policy_backend import MockPolicyBackend, OPDTransition, PolicyBackend
from training.opd.shadow_harness import ShadowHarness
from training.opd.token_alignment import is_critical_action_token
from training.train_rl import SlidingWindowSearchEnv
from tinker_cookbook.completers import TinkerTokenCompleter

logger = structlog.get_logger("training.opd.rollout_worker")


@dataclass
class QueryRecord:
    query_id: str
    query: str


@dataclass
class RolloutConfig:
    dataset: str = "browsecompplus"
    split: Literal["train", "test", "rl", "sft", "all"] = "train"
    collection_split: Literal["train", "test", "rl"] = "train"
    limit: int = 50
    seed: int = 42
    query_ids: list[str] | None = None
    query_records: list[QueryRecord] | None = None
    max_turns: int = 35
    temperature: float = 1.0
    max_tokens: int = 2048
    parallel: int = 4
    target_module: str = "verification"
    reranker: str = "baseten"
    successful_only: bool = False
    recall_threshold: float = 0.0


@dataclass
class EpisodeRollout:
    episode_id: str
    query_id: str
    query_text: str
    transitions: list[OPDTransition] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: bool = False


def load_query_records_from_json(path: str | Path) -> list[QueryRecord]:
    """Load query records from a JSON list or {query_ids: [...]} manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "query_ids" in raw:
        raise ValueError(
            "query id manifest must be paired with dataset config; "
            "use a JSON list of {query_id, query} objects for standalone rollout"
        )
    if not isinstance(raw, list):
        raise ValueError("queries JSON must be a list of {query_id, query} objects")
    records: list[QueryRecord] = []
    for item in raw:
        if not isinstance(item, dict) or "query_id" not in item or "query" not in item:
            raise ValueError("each query record must include query_id and query")
        records.append(
            QueryRecord(query_id=str(item["query_id"]), query=str(item["query"]))
        )
    return records


def resolve_query_records(config: RolloutConfig) -> list[QueryRecord]:
    if config.query_records:
        records = list(config.query_records)
        if config.query_ids:
            allowed = set(config.query_ids)
            records = [r for r in records if r.query_id in allowed]
        if config.limit > 0:
            return records[: config.limit]
        return records

    query_ids = resolve_query_ids(config)
    dataset = get_dataset(config.dataset)
    return [
        QueryRecord(query_id=qid, query=dataset.get_query_by_id(qid)[1])
        for qid in query_ids
    ]


def resolve_query_ids(config: RolloutConfig) -> list[str]:
    """Resolve paired BrowseComp query IDs for rollout."""
    dataset = get_dataset(config.dataset)
    if config.split == "all":
        pool = dataset.get_all_query_ids()
    elif config.split == "test":
        pool = dataset.get_test_query_ids()
    elif config.split == "rl":
        pool = dataset.get_rl_query_ids()
    elif config.split == "sft":
        pool = dataset.get_sft_query_ids()
    else:
        pool = dataset.get_all_query_ids(split="train")

    if config.query_ids:
        known = set(pool)
        selected = [qid for qid in config.query_ids if qid in known]
        missing = [qid for qid in config.query_ids if qid not in known]
        if missing:
            logger.warning(
                "query_ids_missing_from_split",
                n_missing=len(missing),
                examples=missing[:5],
                split=config.split,
            )
        if not selected:
            raise ValueError(
                f"No valid query IDs for dataset={config.dataset} split={config.split}"
            )
        return selected[: config.limit] if config.limit > 0 else selected

    if config.limit <= 0:
        raise ValueError("limit must be positive when query_ids are not provided")
    if len(pool) < config.limit:
        logger.warning(
            "requested_more_queries_than_available",
            requested=config.limit,
            available=len(pool),
        )
    rng = random.Random(config.seed)
    return [str(qid) for qid in rng.sample(pool, min(config.limit, len(pool)))]


def _get_token_encoder():
    return tiktoken.get_encoding("o200k_harmony")


def _model_input_to_ids(model_input: tinker.ModelInput) -> list[int]:
    return list(model_input.to_ints())


def _encode_teacher_prefix(
    student_input_ids: list[int],
    privileged_text: str,
    encoder=None,
) -> list[int]:
    encoder = encoder or _get_token_encoder()
    if not privileged_text:
        return list(student_input_ids)
    suffix = encoder.encode(
        f"\n\n=== Privileged Module Context ===\n{privileged_text}",
        disallowed_special=(),
    )
    return list(student_input_ids) + suffix


def _action_mask_for_tokens(action_ids: list[int], encoder=None) -> list[bool]:
    encoder = encoder or _get_token_encoder()
    try:
        text = encoder.decode(action_ids)
    except Exception:
        return [True] * len(action_ids)
    critical = is_critical_action_token(text)
    return [critical] * len(action_ids) if critical else [True] * len(action_ids)


def _extract_verify_calls(action) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool, params in zip(action.tools, action.params):
        name = "user_text" if isinstance(tool, UserTextTool) else tool.tool_schema.name
        if name != "verify":
            continue
        if not isinstance(params, dict):
            continue
        doc_ids = params.get("doc_ids", [])
        if not isinstance(doc_ids, list):
            doc_ids = [str(doc_ids)] if doc_ids else []
        claim = str(params.get("claim", "")).strip()
        calls.append({"doc_ids": [str(d) for d in doc_ids if d], "claim": claim})
    return calls


def _doc_texts_from_wm(env: SlidingWindowSearchEnv, doc_ids: list[str]) -> dict[str, str]:
    doc_texts: dict[str, str] = {}
    for did in doc_ids:
        norm = env.wm._normalize_id(did)
        store = env.wm.doc_store.get(norm, {})
        txt = store.get("full_text") or store.get("snippet") or ""
        if txt:
            doc_texts[norm] = txt
    return doc_texts


def extract_opd_transitions_from_episode(
    env: SlidingWindowSearchEnv,
    *,
    episode_id: str,
    turn_records: list[dict[str, Any]],
    shadow: ShadowHarness,
    target_module: str,
    reward: float,
    success: bool,
) -> list[OPDTransition]:
    """Convert per-turn rollout records into OPD transitions."""
    transitions: list[OPDTransition] = []
    encoder = _get_token_encoder()

    for record in turn_records:
        verify_calls = record.get("verify_calls", [])
        if target_module == "verification" and not verify_calls:
            continue

        student_input_ids = record["student_input_ids"]
        action_ids = record["action_ids"]
        turn_id = record["turn_id"]

        privileged_text = ""
        shadow_mode = "none"
        for call in verify_calls:
            doc_ids = call["doc_ids"]
            claim = call["claim"]
            if not doc_ids or not claim:
                continue
            shadow_result = shadow.run_verification_shadow(
                turn_id=turn_id,
                claim=claim,
                doc_ids=doc_ids,
                doc_texts=_doc_texts_from_wm(env, doc_ids),
                student_wm=env.wm,
            )
            shadow_mode = shadow_result.mode
            if shadow_result.artifacts:
                privileged_text = shadow_result.artifacts[0].compact_text

        student_view = StudentView.from_episode_state(
            env.query_text,
            env.wm,
            recent_trajectory=record.get("recent_trajectory", ""),
            include_verification=False,
        )
        teacher_view = TeacherView(
            student_view=student_view,
            privileged_artifacts=[],
            remaining_budget=max(0, env.max_turns - turn_id),
        )
        if privileged_text:
            from harness.views.privileged_artifacts import PrivilegedArtifacts

            teacher_view.privileged_artifacts.append(
                PrivilegedArtifacts(
                    module_id=target_module,
                    turn_id=turn_id,
                    compact_text=privileged_text,
                    future_leakage=False,
                )
            )

        teacher_input_ids = _encode_teacher_prefix(
            student_input_ids,
            teacher_view.render() if privileged_text else "",
            encoder=encoder,
        )

        transitions.append(
            OPDTransition(
                episode_id=episode_id,
                query_id=env.query_id,
                turn_id=turn_id,
                student_input_ids=student_input_ids,
                action_ids=action_ids,
                action_mask=_action_mask_for_tokens(action_ids, encoder=encoder),
                teacher_input_ids=teacher_input_ids,
                privileged_module_id=target_module,
                reward=reward,
                success=success,
                metadata={
                    "shadow_mode": shadow_mode,
                    "verify_calls": verify_calls,
                    "tool_names": record.get("tool_names", []),
                },
            )
        )
    return transitions


def build_mock_transitions_from_queries(
    query_records: list[QueryRecord],
    *,
    target_module: str = "verification",
    shadow: ShadowHarness | None = None,
) -> list[OPDTransition]:
    """Lightweight rollout for pipeline tests without retrieval / Tinker."""
    encoder = _get_token_encoder()
    shadow = shadow or ShadowHarness(
        load_harness_config(config_path("modules_full.yaml")),
        offline=True,
    )
    transitions: list[OPDTransition] = []
    for record in query_records:
        qid = record.query_id
        query_text = record.query
        episode_id = str(uuid.uuid4())
        student_prefix = encoder.encode(f"Query: {query_text}\n", disallowed_special=())
        action_ids = encoder.encode('{"tool":"search_corpus"}', disallowed_special=())
        shadow_result = shadow.run_verification_shadow(
            turn_id=0,
            claim=query_text[:120],
            doc_ids=["doc_stub"],
            doc_texts={"doc_stub": query_text},
            student_wm=None,
        )
        teacher_prefix = _encode_teacher_prefix(
            student_prefix,
            shadow_result.artifacts[0].compact_text if shadow_result.artifacts else "",
            encoder=encoder,
        )
        transitions.append(
            OPDTransition(
                episode_id=episode_id,
                query_id=qid,
                turn_id=0,
                student_input_ids=student_prefix,
                action_ids=action_ids,
                action_mask=[True] * len(action_ids),
                teacher_input_ids=teacher_prefix + action_ids,
                privileged_module_id=target_module,
                reward=0.0,
                success=False,
                metadata={
                    "mode": "mock_browsecomp",
                    "query_preview": query_text[:120],
                    "shadow_mode": shadow_result.mode,
                },
            )
        )
    return transitions


async def rollout_episode_with_tinker(
    *,
    runtime: RolloutRuntime,
    query_id: str,
    query_text: str,
    sampling_client: tinker.SamplingClient,
    shadow: ShadowHarness,
    config: RolloutConfig,
) -> EpisodeRollout:
    """Run one on-policy BrowseComp episode and extract OPD transitions."""
    from harness.agent import TinkerAgentInferenceModel

    episode_id = str(uuid.uuid4())
    env = build_search_env(
        runtime,
        query_id=query_id,
        query_text=query_text,
        max_turns=config.max_turns,
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    full_toolset = env._build_full_toolset()
    turn_records: list[dict[str, Any]] = []

    try:
        ob, stop_condition = await env.initial_observation()
        turn_id = 0
        while True:
            student_input_ids = _model_input_to_ids(ob)
            action_with_logprobs = await policy(ob, stop_condition)
            action_ids = list(action_with_logprobs.tokens)
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
                env.enc, action_ids, full_toolset
            )
            verify_calls = _extract_verify_calls(action)
            tool_names = [
                "user_text" if isinstance(t, UserTextTool) else t.tool_schema.name
                for t in action.tools
            ]
            turn_records.append(
                {
                    "turn_id": turn_id,
                    "student_input_ids": student_input_ids,
                    "action_ids": action_ids,
                    "verify_calls": verify_calls,
                    "tool_names": tool_names,
                    "recent_trajectory": "",
                }
            )

            step_result = await env.step(action_ids)
            turn_id += 1
            if step_result.episode_done:
                break
            ob = step_result.next_observation
            stop_condition = step_result.next_stop_condition

        reward = float(env._terminal_reward)
        recall = float(env._terminal_metrics.get("recall", 0.0))
        success = recall >= config.recall_threshold and not bool(
            env._terminal_metrics.get("no_error", 1.0) == 0.0
        )
        transitions = extract_opd_transitions_from_episode(
            env,
            episode_id=episode_id,
            turn_records=turn_records,
            shadow=shadow,
            target_module=config.target_module,
            reward=reward,
            success=success,
        )
        if config.successful_only:
            transitions = [t for t in transitions if t.success]

        return EpisodeRollout(
            episode_id=episode_id,
            query_id=query_id,
            query_text=query_text,
            transitions=transitions,
            metrics={
                "reward": reward,
                "recall": recall,
                "turns": turn_id,
                "n_transitions": len(transitions),
                "n_verify_turns": sum(
                    1 for r in turn_records if r.get("verify_calls")
                ),
                **{
                    k: v
                    for k, v in env._terminal_metrics.items()
                    if isinstance(v, (int, float, str, bool))
                },
            },
        )
    except Exception as exc:
        logger.error("rollout_episode_failed", query_id=query_id, error=str(exc)[:300])
        return EpisodeRollout(
            episode_id=episode_id,
            query_id=query_id,
            query_text=query_text,
            transitions=[],
            metrics={},
            error=True,
        )


class BrowseCompRolloutWorker:
    """Collect OPD transitions from BrowseComp(+)-backed Harness rollouts."""

    def __init__(
        self,
        config: RolloutConfig,
        *,
        student_config: HarnessConfig | None = None,
        teacher_config: HarnessConfig | None = None,
        openai_client: Any = None,
        offline_shadow: bool = False,
    ) -> None:
        self.config = config
        self.shadow = ShadowHarness(
            teacher_config
            if teacher_config is not None
            else load_harness_config(config_path("modules_full.yaml")),
            openai_client=openai_client,
            offline=offline_shadow,
        )
        self._student_config = student_config
        self._runtime: RolloutRuntime | None = None
        self._dataset: SearchDataset | None = None

    @property
    def dataset(self) -> SearchDataset:
        if self._dataset is None:
            self._dataset = get_dataset(self.config.dataset)
        return self._dataset

    def resolve_query_records(self) -> list[QueryRecord]:
        return resolve_query_records(self.config)

    def resolve_query_ids(self) -> list[str]:
        return [r.query_id for r in self.resolve_query_records()]

    def collect_mock_transitions(self) -> list[OPDTransition]:
        query_records = self.resolve_query_records()
        logger.info(
            "mock_browsecomp_rollout",
            dataset=self.config.dataset,
            split=self.config.split,
            n_queries=len(query_records),
        )
        return build_mock_transitions_from_queries(
            query_records,
            target_module=self.config.target_module,
            shadow=self.shadow,
        )

    async def collect_live_transitions(
        self,
        checkpoint_path: str,
    ) -> tuple[list[OPDTransition], list[EpisodeRollout]]:
        query_records = self.resolve_query_records()
        self._runtime = build_rollout_runtime(
            self.config.dataset,
            collection_split=self.config.collection_split,
            reranker=self.config.reranker,
        )
        sc = tinker.ServiceClient()
        sampling_client = sc.create_sampling_client(model_path=checkpoint_path)
        logger.info(
            "live_browsecomp_rollout",
            checkpoint=checkpoint_path,
            dataset=self.config.dataset,
            split=self.config.split,
            n_queries=len(query_records),
        )

        sem = asyncio.Semaphore(self.config.parallel)

        async def _one(record: QueryRecord) -> EpisodeRollout:
            async with sem:
                return await rollout_episode_with_tinker(
                    runtime=self._runtime,
                    query_id=record.query_id,
                    query_text=record.query,
                    sampling_client=sampling_client,
                    shadow=self.shadow,
                    config=self.config,
                )

        episodes = await asyncio.gather(*[_one(record) for record in query_records])
        transitions: list[OPDTransition] = []
        for ep in episodes:
            transitions.extend(ep.transitions)
        return transitions, episodes

    def collect_transitions(
        self,
        *,
        checkpoint_path: str | None = None,
        mock: bool = False,
    ) -> tuple[list[OPDTransition], list[EpisodeRollout] | None]:
        if mock or not checkpoint_path:
            return self.collect_mock_transitions(), None
        transitions, episodes = asyncio.run(
            self.collect_live_transitions(checkpoint_path)
        )
        return transitions, episodes

    def save_rollout_manifest(
        self,
        output_dir: Path,
        query_ids: list[str],
        transitions: list[OPDTransition],
        episodes: list[EpisodeRollout] | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": self.config.dataset,
            "split": self.config.split,
            "seed": self.config.seed,
            "target_module": self.config.target_module,
            "n_queries": len(query_ids),
            "n_transitions": len(transitions),
            "query_ids": query_ids,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "query_id": ep.query_id,
                    "error": ep.error,
                    "metrics": ep.metrics,
                    "n_transitions": len(ep.transitions),
                }
                for ep in (episodes or [])
            ],
        }
        path = output_dir / "rollout_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


__all__ = [
    "BrowseCompRolloutWorker",
    "EpisodeRollout",
    "MockPolicyBackend",
    "OPDTransition",
    "PolicyBackend",
    "QueryRecord",
    "RolloutConfig",
    "build_mock_transitions_from_queries",
    "extract_opd_transitions_from_episode",
    "load_query_records_from_json",
    "resolve_query_ids",
    "resolve_query_records",
    "rollout_episode_with_tinker",
]
