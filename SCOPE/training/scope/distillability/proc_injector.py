"""PROC-mode runtime injector: information-safe procedural capability probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.artifacts.gates import capture_env_fingerprint, run_information_safe_gates
from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.capability_id import CapabilityId
from harness.capability.state import DecisionState
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.registry import build_default_registry
from harness.trajectory import Action
from harness.tools import UserTextTool
from training.scope.distillability.registry import get_probe_spec, shadow_capability_for
from training.train_rl import CurateTool, EndSearchTool, SlidingWindowSearchEnv


@dataclass
class ProcInjector:
    capability_id: CapabilityId
    env: SlidingWindowSearchEnv
    audit: Any  # ProcAuditStats
    realizer: ActionRealizer = field(default_factory=ActionRealizer)
    _registry: Any = field(default=None, repr=False)
    _curate: CurateTool = field(default_factory=CurateTool)
    _end_search: EndSearchTool = field(default_factory=EndSearchTool)

    def __post_init__(self) -> None:
        spec = get_probe_spec(self.capability_id)
        self._registry = build_default_registry(
            evidence_state=spec.shadow_module == "evidence_state"
            or self.capability_id
            in {
                CapabilityId.DUPLICATE_EVIDENCE,
                CapabilityId.EVIDENCE_CURATION,
            },
            verification=spec.shadow_module == "verification"
            or self.capability_id
            in {
                CapabilityId.STOP_DECISION,
                CapabilityId.VERIFICATION_DECISION,
                CapabilityId.EXTERNAL_VERIFICATION,
            },
            budget_control=False,
        )

    def maybe_inject(self, state: DecisionState, action: Action) -> Action:
        spec = get_probe_spec(self.capability_id)
        if not spec.proc_supported:
            return action

        shadow_cap = shadow_capability_for(self.capability_id)
        module_id = spec.shadow_module
        if not module_id or not self._registry.has(module_id):
            return action

        from harness.capability.adapters import parse_action_from_tools

        tool_names: list[str] = []
        params_list: list[dict] = []
        for tool, params in zip(action.tools, action.params):
            if isinstance(tool, UserTextTool):
                tool_names.append("user_text")
            else:
                tool_names.append(tool.tool_schema.name)
            params_list.append(dict(params) if isinstance(params, dict) else {})

        student_cap = parse_action_from_tools(tool_names, params_list)
        if student_cap is None:
            return action

        # External verification PROC: never inject verify calls (no new external info).
        if self.capability_id == CapabilityId.EXTERNAL_VERIFICATION:
            if student_cap.action_type == CapabilityActionType.VERIFY_CLAIM:
                self.audit.external_call_from_proc += 1
                return self._strip_verify(action)
            return action

        # Verification decision PROC: allow verify *decision* routing but block execution.
        if self.capability_id == CapabilityId.VERIFICATION_DECISION:
            if student_cap.action_type == CapabilityActionType.VERIFY_CLAIM:
                self.audit.external_call_from_proc += 1
                return self._strip_verify(action)

        module = self._registry.get(module_id)
        fp_before = capture_env_fingerprint(self.env)
        try:
            artifact = module.analyze(state, student_cap)
        except Exception:
            return action
        fp_after = capture_env_fingerprint(self.env)
        if fp_before != fp_after:
            self.audit.state_mutation_rate += 1.0

        cap = artifact.resolved_capability()
        if cap.value != shadow_cap.value and shadow_cap != CapabilityId.UNKNOWN:
            # Only intervene for target shadow capability family.
            if self.capability_id == CapabilityId.EVIDENCE_CURATION:
                if cap not in {
                    CapabilityId.IRRELEVANT_EVIDENCE,
                    CapabilityId.EVIDENCE_PRIORITIZATION,
                    CapabilityId.SUBTRACTIVE_CURATION,
                }:
                    return action
            else:
                return action

        self.audit.n_shadow_calls += 1
        gate_report = run_information_safe_gates(
            state,
            artifact,
            candidate_action=student_cap,
            fingerprint_before=fp_before,
            fingerprint_after=fp_after,
        )
        if not gate_report.visible:
            self.audit.visibility_violation_rate += 1.0
        if not gate_report.purity_ok:
            self.audit.state_mutation_rate += 1.0

        if artifact.mode == GuidanceMode.IGNORE:
            return action

        candidate = self.realizer.realize(state, artifact)
        if candidate is None:
            return action

        proc_cap = candidate.action
        if proc_cap.action_type == CapabilityActionType.VERIFY_CLAIM:
            if self.capability_id in {
                CapabilityId.EXTERNAL_VERIFICATION,
                CapabilityId.VERIFICATION_DECISION,
            }:
                self.audit.external_call_from_proc += 1
                return action

        if proc_cap.action_type in {
            CapabilityActionType.SEARCH,
            CapabilityActionType.OPEN_DOCUMENT,
            CapabilityActionType.GREP,
        }:
            self.audit.new_observation_from_proc += 1
            return action

        new_action = capability_to_action(
            proc_cap,
            toolset=self.env._build_full_toolset(),
            curate_tool=self._curate,
            end_search_tool=self._end_search,
        )
        if new_action is not None:
            self.audit.n_proc_interventions += 1
            return new_action
        return action

    def _strip_verify(self, action: Action) -> Action:
        new_tools = []
        new_params = []
        for tool, params in zip(action.tools, action.params):
            if isinstance(tool, UserTextTool):
                new_tools.append(tool)
                new_params.append(params)
                continue
            if tool.tool_schema.name == "verify":
                continue
            new_tools.append(tool)
            new_params.append(params)
        if not new_tools:
            return self._block_to_search()
        return Action(tools=new_tools, params=new_params, sources=["proc_strip_verify"])

    def _block_to_search(self) -> Action:
        toolset = self.env._build_full_toolset()
        search = toolset.get_tool("fan_out_search") or toolset.get_tool("search_corpus")
        if search is not None and search.tool_schema.name == "fan_out_search":
            q = self.env.query_text
            return Action(
                tools=[search],
                params=[{"queries": [" ".join(q.split()[:8])]}],
                sources=["proc_fallback_search"],
            )
        if search is not None:
            return Action(
                tools=[search],
                params=[{"query": " ".join(self.env.query_text.split()[:12])}],
                sources=["proc_fallback_search"],
            )
        return Action(
            tools=[self._end_search],
            params=[{"reasoning": "proc_fallback"}],
            sources=["proc_fallback"],
        )


def capability_to_action(
    cap_action: CapabilityAction,
    *,
    toolset,
    curate_tool: CurateTool,
    end_search_tool: EndSearchTool,
) -> Action | None:
    at = cap_action.action_type
    args = dict(cap_action.arguments)

    if at in {CapabilityActionType.CURATE_DOCUMENT, CapabilityActionType.UPDATE_EVIDENCE}:
        return Action(
            tools=[curate_tool],
            params=[
                {
                    "add_ids": args.get("add_ids", []) or [],
                    "remove_ids": args.get("remove_ids", []) or [],
                }
            ],
            sources=["proc_inject"],
        )
    if at == CapabilityActionType.STOP_AND_ANSWER:
        return Action(
            tools=[end_search_tool],
            params=[{"reasoning": args.get("reasoning", "proc_stop")}],
            sources=["proc_inject"],
        )
    if at == CapabilityActionType.ANSWER:
        return Action(
            tools=[UserTextTool()],
            params=[{"text": args.get("text") or args.get("answer", "")}],
            sources=["proc_inject"],
        )
    if at in {CapabilityActionType.SEARCH, CapabilityActionType.CONTINUE_SEARCH, CapabilityActionType.REWRITE_QUERY}:
        if args.get("fan_out") or args.get("queries"):
            tool = toolset.get_tool("fan_out_search") or toolset.get_tool("search_corpus")
            if tool is None:
                return None
            return Action(
                tools=[tool],
                params=[{"queries": args.get("queries", [args.get("query", "")])}],
                sources=["proc_inject"],
            )
        tool = toolset.get_tool("search_corpus")
        if tool is None:
            return None
        return Action(
            tools=[tool],
            params=[{"query": args.get("query", "")}],
            sources=["proc_inject"],
        )
    if at == CapabilityActionType.OPEN_DOCUMENT:
        tool = toolset.get_tool("read_document")
        if tool is None:
            return None
        doc_id = args.get("doc_id") or (args.get("doc_ids") or [""])[0]
        return Action(
            tools=[tool],
            params=[{"doc_id": doc_id}],
            sources=["proc_inject"],
        )
    if at == CapabilityActionType.GREP:
        tool = toolset.get_tool("grep_corpus")
        if tool is None:
            return None
        return Action(
            tools=[tool],
            params=[{"pattern": args.get("pattern", args.get("query", ""))}],
            sources=["proc_inject"],
        )
    if at == CapabilityActionType.REVIEW_DOCS:
        tool = toolset.get_tool("review_docs")
        if tool is None:
            return None
        return Action(
            tools=[tool],
            params=[{"doc_ids": args.get("doc_ids", [])}],
            sources=["proc_inject"],
        )
    return None
