from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state_snapshot import SCAPEStateSnapshot, assert_same_state_before_component_fork

SUPPORTED_TOOLS = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]


@dataclass
class SCAPELiveLoopResult:
    query_id: str
    query: str
    trajectory: Any
    output_dir: str | None
    n_turns: int
    tool_calls: list[str]
    curated_ids_pre: list[str]
    curated_ids_post: list[str]
    component_name: str
    student_inference_privilege: bool
    scape_runtime_source_of_truth: bool = True

    def to_dict(self) -> dict[str, Any]:
        trajectory_payload: Any
        if hasattr(self.trajectory, "model_dump"):
            trajectory_payload = self.trajectory.model_dump(mode="json")
        else:
            trajectory_payload = str(self.trajectory)
        return {
            "query_id": self.query_id,
            "query": self.query,
            "trajectory": trajectory_payload,
            "output_dir": self.output_dir,
            "n_turns": self.n_turns,
            "tool_calls": self.tool_calls,
            "curated_ids_pre": self.curated_ids_pre,
            "curated_ids_post": self.curated_ids_post,
            "component_name": self.component_name,
            "student_inference_privilege": self.student_inference_privilege,
            "scape_runtime_source_of_truth": self.scape_runtime_source_of_truth,
        }


class ScriptedHarnessInferenceModel:
    """Deterministic inference model that drives real Harness-1 tools.

    This is intentionally not a synthetic environment: actions are emitted into
    Harness-1's `Agent.act`, so search/read observations come from the real
    configured SCAPE/Harness tool runtime.
    """

    def __init__(self) -> None:
        self._step = 0

    def __call__(self, context: Any) -> Any:
        from harness.tools import UserTextTool
        from harness.trajectory import ActionBuilder

        toolset = context.toolset
        builder = ActionBuilder()
        if self._step == 0:
            tool = toolset.get_tool("search_corpus")
            if tool is None:
                raise RuntimeError("Harness toolset missing search_corpus")
            builder.add_tool_call(tool, {"query": "SCAPE benchmark evidence"}, "scape_easyopd_search_0")
        elif self._step == 1:
            tool = toolset.get_tool("read_document")
            if tool is None:
                raise RuntimeError("Harness toolset missing read_document")
            doc_id = "doc0"
            try:
                import re

                text = str(context.trajectory)
                match = re.search(r"(?:DOCUMENT ID|Document ID|doc_id)[:# ]+([A-Za-z0-9_.:-]+)", text)
                if match:
                    doc_id = match.group(1).strip().strip("'\"`.,")
            except Exception:
                pass
            builder.add_tool_call(tool, {"doc_id": doc_id}, "scape_easyopd_read_0")
        else:
            builder.add_tool_call(UserTextTool(), {"text": "Final answer based on the retrieved evidence."}, "agent")
        self._step += 1
        return builder.build()


@dataclass
class SCAPEAgentLoop:
    component_name: str
    student_inference_privilege: bool = False
    chroma_collection_name: str = "scape_browsecompplus_local_test"
    max_steps: int = 4
    _last_live_result: SCAPELiveLoopResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.student_inference_privilege:
            raise ValueError("student_inference_privilege must be false for SCAPE component OPD")

    def available_tools(self, *, include_verify: bool = False) -> list[str]:
        if include_verify:
            return list(SUPPORTED_TOOLS)
        return [x for x in SUPPORTED_TOOLS if x != "verify"]

    def scape_runtime_available(self) -> bool:
        return importlib.util.find_spec("harness") is not None or importlib.util.find_spec("scape") is not None

    def fork_same_state(self, snapshot: SCAPEStateSnapshot) -> dict[str, str]:
        return assert_same_state_before_component_fork(snapshot)

    def build_student_view(self, state: dict[str, Any]) -> dict[str, Any]:
        view = dict(state)
        view.pop("evidence_graph", None)
        view.pop("curated_importance", None)
        view.pop("token_budget_marker", None)
        view["student_inference_privilege"] = False
        return view

    def build_teacher_view(self, state: dict[str, Any]) -> dict[str, Any]:
        view = dict(state)
        view["teacher_component"] = self.component_name
        return view

    def _ensure_local_corpus(self, output_dir: Path) -> Path:
        corpus_path = output_dir / "live_corpus.jsonl"
        if corpus_path.exists():
            return corpus_path
        rows = [
            {"id": "doc0", "source": "doc0", "text": "SCAPE benchmark evidence document about component OPD and search."},
            {"id": "doc1", "source": "doc1", "text": "Additional retrieval evidence for projected action curation."},
            {"id": "doc2", "source": "doc2", "text": "Distractor document for search harness smoke."},
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        with corpus_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return corpus_path

    def run_live_search(self, *, query_id: str, query: str, output_dir: str | os.PathLike[str] | None = None) -> SCAPELiveLoopResult:
        if not self.scape_runtime_available():
            raise RuntimeError("SCAPE/Harness runtime is not importable")
        from harness.agent import DeduplicatingPruningSearchAgent
        from harness.config import get_config
        from harness.prompts import get_retrieval_subagent_prompt
        from harness.tools import ToolSet
        from harness.trajectory import Action, Observation

        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            corpus_path = self._ensure_local_corpus(out_path)
            os.environ.setdefault("SCAPE_CHROMA_PATH", str(out_path / "empty_chroma"))
            os.environ.setdefault("SCAPE_RETRIEVAL_CORPUS", str(corpus_path))
            os.environ.setdefault("SCAPE_LOCAL_OPENAI_EMBEDDINGS", "1")
            os.environ.setdefault("SCAPE_FORCE_LOCAL_HARMONY", "1")

        config = get_config()
        toolset = ToolSet.from_config(
            config,
            chroma_collection_name=self.chroma_collection_name,
            search_display_limit=3,
            search_limit=3,
            search_knn_limit=3,
            snippet_max_chars=160,
        )
        agent = DeduplicatingPruningSearchAgent(toolset, ScriptedHarnessInferenceModel(), max_trajectory_length=self.max_steps * 2 + 2)
        initial_observation = Observation(observations=[get_retrieval_subagent_prompt(query)], sources=["user"], tool_metadata=[None])
        trajectory = agent(initial_observation=initial_observation)
        tool_calls: list[str] = []
        for item in trajectory.actions_and_observations:
            if isinstance(item, Action):
                for tool in item.tools:
                    name = getattr(getattr(tool, "tool_schema", None), "name", type(tool).__name__)
                    tool_calls.append(name)
        result = SCAPELiveLoopResult(
            query_id=query_id,
            query=query,
            trajectory=trajectory,
            output_dir=str(out_path) if out_path is not None else None,
            n_turns=len(trajectory.actions_and_observations),
            tool_calls=tool_calls,
            curated_ids_pre=[],
            curated_ids_post=[],
            component_name=self.component_name,
            student_inference_privilege=False,
        )
        self._last_live_result = result
        if out_path is not None:
            (out_path / "LIVE_SCAPE_AGENT_LOOP.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return result
