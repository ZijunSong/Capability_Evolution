from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .action_projection import project_curated_delta
from .skip_to_anchor import HARNESS_ONLY_EVENT_TYPES
from .types import ToolAction

SCAPE_ROOT = Path(os.environ.get("SCAPE_ROOT", "/mnt/songzijun/Capability_Evolution/SCAPE"))
HARNESS1_ROOT = SCAPE_ROOT / "external" / "harness-1"
CANONICAL_STUDENT_BASE = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")
QWEN3_STUDENT_BASE = CANONICAL_STUDENT_BASE
QWEN3_LOGICAL_MODEL_ID = LOGICAL_MODEL_ID
_TOKENIZER_CACHE: Any | None = None

_COMPONENT_ENV = {
    "auto_populate_first_search": "V8D_AUTO_POPULATE_FIRST_SEARCH",
    "importance_tagging": "V8D_IMPORTANCE_TAGGING",
    "subtractive_curation": "V8D_SUBTRACTIVE_CURATION",
    "evidence_graph": "V8D_EVIDENCE_GRAPH",
    "sentence_compress": "V8D_SENTENCE_COMPRESS",
    "chunk_neighbors": "V8D_CHUNK_NEIGHBORS",
    "content_dedup": "V8D_CONTENT_DEDUP",
    "verify_tool": "V8D_VERIFY_TOOL",
    "token_budget_marker": "V8D_TOKEN_BUDGET_MARKER",
    "adaptive_rerank_instruction": "V8D_ADAPTIVE_RERANK_INSTRUCTION",
}

_SEARCH_TOOLS = {"fan_out_search", "search_corpus", "grep_corpus", "read_document"}
_AUTO_SEARCH_TOOLS = {"fan_out_search", "search_corpus"}
_DOC_ID_RE = re.compile(r"#\s*DOCUMENT ID:\s*([^\s)]+)", re.IGNORECASE)


def ensure_harness1_importable() -> Path:
    if not HARNESS1_ROOT.exists():
        raise RuntimeError(f"STOP_REAL_HARNESS_RUNTIME_UNAVAILABLE: missing {HARNESS1_ROOT}")
    root = str(HARNESS1_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import harness  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"STOP_REAL_HARNESS_RUNTIME_UNAVAILABLE: {exc}") from exc
    return HARNESS1_ROOT


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _qwen3_tokenizer() -> Any:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(QWEN3_STUDENT_BASE, trust_remote_code=True, local_files_only=True)
    return _TOKENIZER_CACHE


def _compact_tool_action(action: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "tool_name": action.get("tool_name") or action.get("name") or "",
        "parameters": copy.deepcopy(action.get("parameters") or action.get("params") or {}),
        "returned_doc_ids": [str(x) for x in (action.get("returned_doc_ids") or [])],
    }
    observation = str(action.get("observation") or "")
    doc_texts = action.get("doc_texts") if isinstance(action.get("doc_texts"), dict) else {}
    compact["observation_chars"] = len(observation)
    compact["doc_text_chars"] = sum(len(str(value)) for value in doc_texts.values())
    compact["doc_text_count"] = len(doc_texts)
    return compact


def _estimate_next_context_tokens(*, query: str, wm_text: str, tool_history: list[dict[str, Any]], current_action: dict[str, Any]) -> dict[str, Any]:
    """Measure the next student-visible context with the Qwen3 tokenizer.

    Token-budget-marker events fire on the observation that follows the current
    tool action, so the relevant measurement is the next prompt context: query +
    current working memory + previous turns + the current action/observation.
    """
    current_record = copy.deepcopy(current_action)
    history = [copy.deepcopy(item) for item in tool_history] + [current_record]
    transcript_lines: list[str] = []
    for index, item in enumerate(history):
        transcript_lines.append(f"Turn {index} tool: {item.get('tool_name') or item.get('name') or ''}")
        transcript_lines.append("parameters: " + _canonical(item.get('parameters') or item.get('params') or {}))
        observation = str(item.get('observation') or '')
        doc_texts = item.get('doc_texts') if isinstance(item.get('doc_texts'), dict) else {}
        if not observation and doc_texts:
            doc_ids = item.get('returned_doc_ids') or list(doc_texts)
            observation = "\n\n".join(f"# DOCUMENT ID: {doc_id}\n{doc_texts.get(str(doc_id), doc_texts.get(doc_id, ''))}" for doc_id in doc_ids)
        transcript_lines.append("observation: " + observation)
    messages = [
        {"role": "system", "content": "You are a SCAPE research agent. Use available tools to find and curate evidence."},
        {"role": "user", "content": str(query)},
        {"role": "user", "content": str(wm_text)},
        {"role": "user", "content": "\n".join(transcript_lines)},
    ]
    tokenizer = _qwen3_tokenizer()
    rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(str(rendered_text), add_special_tokens=False)
    input_ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
    return {
        "used_tokens": len(input_ids),
        "measurement": "qwen3_native_chat_template_next_context_with_current_observation",
        "n_history_turns_included": len(history),
        "current_observation_chars": len(str(current_record.get('observation') or '')),
        "current_doc_text_chars": sum(len(str(value)) for value in (current_record.get('doc_texts') or {}).values()) if isinstance(current_record.get('doc_texts'), dict) else 0,
    }


@contextmanager
def harness_component_env(component: str, *, enabled: bool) -> Iterator[None]:
    if component not in _COMPONENT_ENV:
        raise KeyError(f"unknown Harness-1 component: {component}")
    ensure_harness1_importable()
    old = {name: os.environ.get(name) for name in _COMPONENT_ENV.values()}
    old_force = os.environ.get("SCAPE_FORCE_LOCAL_HARMONY")
    try:
        for name in _COMPONENT_ENV.values():
            os.environ[name] = "0"
        os.environ[_COMPONENT_ENV[component]] = "1" if enabled else "0"
        if enabled and component in {"importance_tagging", "subtractive_curation"}:
            os.environ["V8D_IMPORTANCE_TAGGING"] = "1"
        os.environ.setdefault("SCAPE_FORCE_LOCAL_HARMONY", "1")
        import harness.ultra_core as ultra_core

        importlib.reload(ultra_core)
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if old_force is None:
            os.environ.pop("SCAPE_FORCE_LOCAL_HARMONY", None)
        else:
            os.environ["SCAPE_FORCE_LOCAL_HARMONY"] = old_force
        try:
            import harness.ultra_core as ultra_core

            importlib.reload(ultra_core)
        except Exception:
            pass


def parse_doc_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in _DOC_ID_RE.finditer(text or ""):
        doc_id = match.group(1).strip().strip("'\"`.,")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ids


class Qwen3NativeChatAdapter:
    """Contract wrapper for the local Qwen3 tokenizer chat template."""

    logical_model_id = QWEN3_LOGICAL_MODEL_ID

    def __init__(self, model_path: str | Path = QWEN3_STUDENT_BASE) -> None:
        self.model_path = str(model_path)
        self.contract_mode = "qwen3_native_chat_template"
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"STOP_QWEN3_CHAT_CONTRACT_FAILED: transformers import failed: {exc}") from exc
        path = Path(self.model_path)
        if not path.is_dir() or not (path / "config.json").is_file():
            raise RuntimeError(f"STOP_QWEN3_CHAT_CONTRACT_FAILED: missing local model directory/config: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("STOP_QWEN3_CHAT_CONTRACT_FAILED: tokenizer has no native chat_template")

    def build_context(self, query: str, *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a SCAPE research agent. Use available tools to find and curate evidence."},
            {"role": "user", "content": str(query)},
        ]

    def render(self, query: str, *, tools: list[dict[str, Any]] | None = None) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            self.build_context(query, tools=tools),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        elif isinstance(rendered, dict):
            rendered = rendered.get("input_ids") or []
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token_id) for token_id in rendered]

    def tokenizer_consistency_check(self) -> dict[str, Any]:
        digests = []
        for idx in range(10):
            tokens = self.render(f"deterministic serialization case {idx}: find cited evidence {idx}")
            if not tokens:
                raise RuntimeError("STOP_QWEN3_CHAT_CONTRACT_FAILED: empty Qwen3 rendering")
            digests.append(hashlib.sha256(json.dumps(tokens[:4096], separators=(",", ":")).encode()).hexdigest())
        chat_template = str(getattr(self.tokenizer, "chat_template", ""))
        return {
            "logical_model_id": self.logical_model_id,
            "resolved_model_path": self.model_path,
            "contract_mode": self.contract_mode,
            "n_cases": 10,
            "unique_digests": len(set(digests)),
            "digests": digests,
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        }


GptOssHarmonyAdapter = Qwen3NativeChatAdapter


@dataclass
class Harness1Event:
    component: str
    event_type: str
    event_active: bool
    payload: dict[str, Any] = field(default_factory=dict)
    projectable_target: dict[str, Any] | None = None
    projection_valid: bool = False
    valid_args: bool = False
    harness_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        harness_only = self.harness_only or (
            self.event_type in HARNESS_ONLY_EVENT_TYPES and self.projectable_target is None
        )
        return {
            "component": self.component,
            "event_type": self.event_type,
            "event_active": self.event_active,
            "payload": self.payload,
            "projectable_target": self.projectable_target,
            "projection_valid": self.projection_valid,
            "valid_args": self.valid_args,
            "harness_only": harness_only,
        }


@dataclass
class Harness1Bridge:
    component: str
    enabled: bool = False
    student_base: str = QWEN3_STUDENT_BASE
    student_inference_privilege: bool = False
    _wm: Any = field(default=None, init=False, repr=False)
    _query_record: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _rollout_seed: int = field(default=0, init=False, repr=False)
    _step_id: int = field(default=0, init=False, repr=False)
    _tool_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _last_event: Harness1Event | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.student_base != QWEN3_STUDENT_BASE:
            raise ValueError(f"canonical student base must be {QWEN3_STUDENT_BASE}")
        if self.student_inference_privilege:
            raise ValueError("student_inference_privilege must be false")
        if self.component not in _COMPONENT_ENV:
            raise KeyError(f"unknown Harness-1 component: {self.component}")
        ensure_harness1_importable()

    def reset(self, query_record: dict[str, Any], rollout_seed: int) -> dict[str, Any]:
        self._query_record = dict(query_record)
        self._rollout_seed = int(rollout_seed)
        self._step_id = 0
        self._tool_history = []
        self._last_event = None
        query = str(query_record.get("query") or query_record.get("question") or query_record.get("query_text") or query_record.get("id") or query_record.get("query_id"))
        with harness_component_env(self.component, enabled=False):
            from harness.ultra_core import WorkingMemory

            self._wm = WorkingMemory(query, normalize_ids=True)
        return self.snapshot_student_visible_state()

    def _require_reset(self) -> None:
        if self._wm is None:
            raise RuntimeError("Harness1Bridge.reset must be called before step")

    def snapshot_student_visible_state(self) -> dict[str, Any]:
        self._require_reset()
        wm = self._wm
        state = {
            "query_id": str(self._query_record.get("query_id", self._query_record.get("id", ""))),
            "query": str(self._query_record.get("query") or self._query_record.get("question") or self._query_record.get("query_text") or ""),
            "rollout_seed": self._rollout_seed,
            "step_id": self._step_id,
            "student_visible_prefix": wm.to_text(),
            "tool_history": [_compact_tool_action(item) for item in self._tool_history],
            "student_observable_env_state": {
                "curated_ids": list(wm.curated_ids),
                "visible_doc_ids": list(wm.pool_ids),
                "search_history": list(wm.search_history),
                "pool_size": len(wm.pool_ids),
                "curated_count": len(wm.curated_ids),
            },
            "student_inference_privilege": False,
            "component_masks": {self.component: False},
        }
        state["state_hash"] = _sha({"prefix": state["student_visible_prefix"], "history": state["tool_history"], "env": state["student_observable_env_state"]})
        return state

    def _apply_action_to_wm(self, wm: Any, action: dict[str, Any], *, component_enabled: bool) -> Harness1Event | None:
        tool_name = str(action.get("tool_name") or action.get("name") or "")
        params = action.get("parameters") or action.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        observation = str(action.get("observation") or "")
        doc_ids = [str(x) for x in (action.get("returned_doc_ids") or parse_doc_ids(observation))]
        doc_texts = action.get("doc_texts") if isinstance(action.get("doc_texts"), dict) else {}
        pre_curated = list(wm.curated_ids)
        pre_visible = list(wm.pool_ids)
        event: Harness1Event | None = None

        if tool_name in _SEARCH_TOOLS:
            before_pool = wm.get_pool_size()
            before_dup_skipped = int(getattr(wm, "dup_skipped", 0))
            wm.add_to_pool(doc_ids, doc_texts if doc_texts else None)
            num_new = wm.get_pool_size() - before_pool
            dup_delta = int(getattr(wm, "dup_skipped", 0)) - before_dup_skipped
            if tool_name == "fan_out_search":
                summary = "; ".join(str(q)[:30] for q in (params.get("queries") or [])[:3])
                wm.add_search_record("fan_out", summary, len(doc_ids), num_new=num_new)
            elif tool_name == "search_corpus":
                wm.add_search_record("search", str(params.get("query", ""))[:50], len(doc_ids), num_new=num_new)
            elif tool_name == "grep_corpus":
                wm.add_search_record("grep", str(params.get("pattern", ""))[:50], len(doc_ids), num_new=num_new)
            else:
                wm.add_search_record("read", str(params.get("doc_id", "")), len(doc_ids), num_new=num_new)

            if self.component == "content_dedup" and component_enabled and dup_delta > 0:
                event = Harness1Event(
                    component=self.component,
                    event_type="near_duplicate_pool_suppressed",
                    event_active=True,
                    payload={
                        "search_result_doc_ids": doc_ids,
                        "pool_ids_pre": pre_visible,
                        "pool_ids_teacher_post": list(wm.pool_ids),
                        "duplicate_suppressed_count": dup_delta,
                        "student_native_targets": ["READ_CANONICAL", "CURATE_CANONICAL", "SKIP_REDUNDANT"],
                    },
                    projectable_target={"name": "curate", "arguments": {"add_ids": list(wm.pool_ids[-max(num_new, 0):]), "remove_ids": []}} if num_new > 0 else None,
                    projection_valid=num_new > 0,
                    valid_args=num_new > 0,
                    harness_only=num_new <= 0,
                )

            if self.component == "adaptive_rerank_instruction" and component_enabled and tool_name in _AUTO_SEARCH_TOOLS:
                from harness.ultra_core import build_rerank_instruction

                query_text = str(params.get("query") or self._query_record.get("query") or self._query_record.get("question") or "")
                instruction = build_rerank_instruction(query_text, dataset_name="browsecompplus", use_llm=False)
                wm.rerank_instruction = instruction
                event = Harness1Event(
                    component=self.component,
                    event_type="adaptive_rerank_instruction_available",
                    event_active=bool(instruction),
                    payload={
                        "instruction_effect": instruction,
                        "retrieval_ranking_effect": "not_measured_by_bridge_without_reranker_metadata",
                        "retrieved_doc_ids_off": doc_ids,
                        "retrieved_doc_ids_on": doc_ids,
                        "topK_overlap": 1.0,
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=False,
                    harness_only=True,
                )

            if self.component == "chunk_neighbors" and component_enabled and tool_name in _SEARCH_TOOLS:
                event = Harness1Event(
                    component=self.component,
                    event_type="chunk_neighbors_no_runtime_hook_detected",
                    event_active=False,
                    payload={
                        "reason": "V8D_CHUNK_NEIGHBORS flag is defined in Harness-1 but no student-visible neighbor injection hook is present in ultra_core/tool runtime.",
                        "student_native_access": "read_document/search_corpus can access documents already returned by normal retrieval only",
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=False,
                    harness_only=True,
                )

            if self.component == "verify_tool" and component_enabled and doc_ids:
                event = Harness1Event(
                    component=self.component,
                    event_type="verify_tool_action_available",
                    event_active=True,
                    payload={
                        "available_teacher_action": "verify(doc_ids, claim)",
                        "candidate_doc_ids": doc_ids,
                        "claim_source": str(params.get("query") or self._query_record.get("query") or self._query_record.get("question") or "")[:240],
                        "student_action_space_has_verify": False,
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=True,
                    harness_only=True,
                )
            if self.component == "token_budget_marker" and component_enabled:
                budget_proxy = int(os.environ.get("SCAPE_TOKEN_BUDGET_PROXY", "30720"))
                measured = _estimate_next_context_tokens(
                    query=str(self._query_record.get("query") or self._query_record.get("question") or self._query_record.get("query_text") or ""),
                    wm_text=str(wm.to_text()),
                    tool_history=self._tool_history,
                    current_action=action,
                )
                used_tokens = int(measured["used_tokens"])
                marker = None
                try:
                    from harness.ultra_core import format_token_budget_marker

                    marker = format_token_budget_marker(used_tokens, budget=budget_proxy)
                except Exception:
                    marker = f"[Context: {used_tokens}/{budget_proxy}]"
                event = Harness1Event(
                    component=self.component,
                    event_type="token_budget_marker_visible",
                    event_active=True,
                    payload={
                        "used_tokens_proxy": used_tokens,
                        "used_tokens_measurement": measured,
                        "budget_proxy": budget_proxy,
                        "token_budget_marker": marker,
                        "current_tool_name": tool_name,
                        "current_returned_doc_ids": doc_ids,
                        "termination_timing": "budget-aware runtime bookkeeping",
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=True,
                    harness_only=True,
                )
            if self.component == "auto_populate_first_search" and component_enabled and tool_name in _AUTO_SEARCH_TOOLS and doc_ids:
                from harness.ultra_core import auto_populate_from_first_search

                auto_populate_from_first_search(wm, doc_ids)
                post_curated = list(wm.curated_ids)
                action_obj, audit = project_curated_delta(curated_ids_pre=pre_curated, curated_ids_post=post_curated, visible_doc_ids=list(wm.pool_ids))
                event = Harness1Event(
                    component=self.component,
                    event_type="first_successful_search_auto_populate",
                    event_active=bool(action_obj),
                    payload={"search_result_doc_ids": doc_ids, "curated_ids_pre": pre_curated, "curated_ids_teacher_post": post_curated, "delta_add": audit.get("add_ids", []), "visible_doc_ids": list(wm.pool_ids)},
                    projectable_target=action_obj.to_dict() if action_obj else None,
                    projection_valid=bool(audit.get("projection_valid")),
                    valid_args=bool(audit.get("projection_valid")),
                )
            elif self.component == "evidence_graph" and component_enabled and doc_ids:
                from harness.ultra_core import EvidenceGraph

                graph = getattr(wm, "evidence_graph", None)
                if graph is None:
                    graph = EvidenceGraph()
                    wm.evidence_graph = graph
                for did in doc_ids:
                    text = ""
                    if isinstance(doc_texts, dict):
                        text = str(doc_texts.get(did, doc_texts.get(str(did), "")) or "")
                    if text:
                        graph.update_from_doc(str(did), text)
                graph_text = graph.render_summary()
                event = Harness1Event(
                    component=self.component,
                    event_type="evidence_graph_privileged_context",
                    event_active=bool(graph_text),
                    payload={
                        "search_result_doc_ids": doc_ids,
                        "visible_doc_ids": list(wm.pool_ids),
                        "evidence_graph_summary": graph_text,
                        "same_underlying_docs": True,
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=True,
                    harness_only=True,
                )
            elif self.component == "sentence_compress" and component_enabled and observation:
                from harness.ultra_core import compress_search_observation

                compressed = compress_search_observation(str(wm.query), observation)
                event = Harness1Event(
                    component=self.component,
                    event_type="sentence_compress_privileged_context",
                    event_active=compressed != observation,
                    payload={
                        "search_result_doc_ids": doc_ids,
                        "compression_input_doc_ids": doc_ids,
                        "original_observation_chars": len(observation),
                        "compressed_observation_chars": len(compressed),
                        "compressed_teacher_view": compressed,
                        "same_underlying_docs": True,
                    },
                    projectable_target=None,
                    projection_valid=False,
                    valid_args=True,
                    harness_only=True,
                )
        elif tool_name == "review_docs":
            ids = params.get("doc_ids") or []
            wm.add_search_record("review", ", ".join([str(x) for x in ids[:3]]), len(ids))
        elif tool_name == "verify":
            ids = params.get("doc_ids") or []
            wm.add_search_record("verify", str(params.get("claim", ""))[:50], len(ids), num_new=0)
        elif tool_name == "curate":
            add_ids = params.get("add_ids") or []
            remove_ids = params.get("remove_ids") or []
            if not isinstance(add_ids, list):
                add_ids = [add_ids]
            if not isinstance(remove_ids, list):
                remove_ids = [remove_ids]
            importance = params.get("importance") if component_enabled else None
            wm.curate([str(x) for x in add_ids], [str(x) for x in remove_ids], importance=importance)
            if component_enabled and self.component in {"importance_tagging", "subtractive_curation"}:
                post_curated = list(wm.curated_ids)
                action_obj, audit = project_curated_delta(curated_ids_pre=pre_curated, curated_ids_post=post_curated, visible_doc_ids=pre_visible)
                event = Harness1Event(
                    component=self.component,
                    event_type="importance_informed_curation_delta" if self.component == "importance_tagging" else "subtractive_curation_delta",
                    event_active=bool(action_obj),
                    payload={"curated_ids_pre": pre_curated, "curated_ids_teacher_post": post_curated, "added": audit.get("add_ids", []), "removed": audit.get("remove_ids", []), "visible_doc_ids": pre_visible},
                    projectable_target=action_obj.to_dict() if action_obj else None,
                    projection_valid=bool(audit.get("projection_valid")),
                    valid_args=bool(audit.get("projection_valid")),
                )
        wm.advance_turn()
        return event

    def build_teacher_view_from_same_state(self, *, student_action: dict[str, Any] | None = None, pre_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_reset()
        wm_copy = copy.deepcopy(self._wm)
        action = student_action or (self._tool_history[-1] if self._tool_history else {})
        with harness_component_env(self.component, enabled=True):
            if self.component == "content_dedup" and getattr(wm_copy, "content_dedup", None) is None:
                from harness.ultra_core import ContentDedupTracker

                wm_copy.content_dedup = ContentDedupTracker()
            event = self._apply_action_to_wm(wm_copy, action, component_enabled=True)
        student_state = pre_state or self.snapshot_student_visible_state()
        return {
            "component": self.component,
            "query_id": student_state.get("query_id"),
            "rollout_seed": self._rollout_seed,
            "step_id": student_state.get("step_id"),
            "teacher_component_enabled": True,
            "student_state_hash": student_state.get("state_hash"),
            "teacher_observable_env_state": {"curated_ids": list(wm_copy.curated_ids), "visible_doc_ids": list(wm_copy.pool_ids), "search_history": list(wm_copy.search_history)},
            "event": event.to_dict() if event else None,
        }

    def step(self, student_action: dict[str, Any]) -> dict[str, Any]:
        self._require_reset()
        pre = self.snapshot_student_visible_state()
        teacher_view = self.build_teacher_view_from_same_state(student_action=student_action, pre_state=pre)
        with harness_component_env(self.component, enabled=False):
            self._apply_action_to_wm(self._wm, student_action, component_enabled=False)
        self._tool_history.append(copy.deepcopy(student_action))
        self._step_id += 1
        event_payload = teacher_view.get("event")
        if event_payload:
            self._last_event = Harness1Event(
                **{k: v for k, v in event_payload.items() if k in Harness1Event.__dataclass_fields__}
            )
        else:
            self._last_event = None
        post = self.snapshot_student_visible_state()
        return {
            "pre_state": pre,
            "post_state": post,
            "teacher_view": teacher_view,
            "event": self.get_component_event(),
            "student_action": copy.deepcopy(student_action),
        }

    def get_component_event(self) -> dict[str, Any] | None:
        return self._last_event.to_dict() if self._last_event else None

    def event_row_from_step(self, step: dict[str, Any], *, rollout_id: str) -> dict[str, Any] | None:
        event = step.get("event") or {}
        if not event.get("event_active"):
            return None
        pre = step["pre_state"]
        payload = event.get("payload") or {}
        target = event.get("projectable_target")
        row = {
            "component": self.component,
            "query_id": pre["query_id"],
            "rollout_id": rollout_id,
            "rollout_seed": self._rollout_seed,
            "step_id": pre["step_id"],
            "event_type": event.get("event_type"),
            "student_visible_prefix": pre["student_visible_prefix"],
            "tool_history": pre["tool_history"],
            "student_observable_env_state": pre["student_observable_env_state"],
            "event_payload_student_visible": payload,
            "teacher_privileged_view_ref": _sha(step.get("teacher_view", {})),
            "projectable_target": target,
            "terminal_reward": None,
            "state_uid": "",
            "collector_mode": "real_harness1",
            "projection_valid": bool(event.get("projection_valid")),
            "valid_args": bool(event.get("valid_args")),
            "visible_doc_ids": pre["student_observable_env_state"].get("visible_doc_ids", []),
            "curated_ids_pre": payload.get("curated_ids_pre", pre["student_observable_env_state"].get("curated_ids", [])),
            "curated_ids_post": payload.get("curated_ids_teacher_post", []),
        }
        return row


def tool_action_to_record(name: str, arguments: dict[str, Any], *, observation: str = "", returned_doc_ids: list[str] | None = None, doc_texts: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "tool_name": name,
        "parameters": arguments,
        "observation": observation,
        "returned_doc_ids": returned_doc_ids or parse_doc_ids(observation),
        "doc_texts": doc_texts or {},
    }


def projected_tool_action(add_ids: list[str], remove_ids: list[str] | None = None) -> dict[str, Any]:
    return ToolAction("curate", {"add_ids": add_ids, "remove_ids": remove_ids or []}).to_dict()
