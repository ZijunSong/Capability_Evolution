"""H0: the fixed minimal agent loop.

Per step: render user goal / tool schemas / legal history -> model proposes one
action -> parse/validate -> run the basic tool -> append the real observation ->
continue or finish.  No auto-decomposition, reflection, verification, retrieval
planning, summarizer, cross-task memory, or extra LLM calls exist in H0 itself.
Format errors are visible feedback and consume action budget.  Network-level
resend is separated from model decision errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from rsi.contracts.action import Action, ActionParseError, Finish, ToolCall, parse_action
from rsi.contracts.common import digest_of
from rsi.contracts.errors import (
    BudgetExceeded,
    InfraError,
    ModelInvalidAction,
)
from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
from rsi.contracts.observation import Observation
from rsi.contracts.policy import PolicyArtifact, RuntimeContract, SamplingParams
from rsi.contracts.task import PublicTask
from rsi.contracts.trace import DecisionRecord, EpisodeTrace
from rsi.harness.composition import HarnessComposition
from rsi.harness.component import EpisodeContext, PublicView
from rsi.harness.loader import LoadedBank
from rsi.runtime.api import MeteredAPI
from rsi.runtime.budgets import BudgetLedger, BudgetSpec
from rsi.runtime.context import CONTEXT_POLICY_ID, ContextPolicy, prompt_hash, view_digest
from rsi.contracts.observation import tool_catalog_digest


@dataclass
class EpisodeResult:
    trace: EpisodeTrace
    final_artifact: Any = None
    env_state_digest: str = ""
    budget_snapshot: dict[str, int] = field(default_factory=dict)
    component_triggers: dict[str, int] = field(default_factory=dict)


class MinimalLoop:
    """Runs one episode under H0 (+ optional bank components for Teacher/Full)."""

    def __init__(
        self,
        *,
        env: Any,
        policy: Any,
        bank: LoadedBank,
        budgets: BudgetSpec | None = None,
        sampling: SamplingParams | None = None,
        context_policy: ContextPolicy | None = None,
        campaign_id: str = "dev",
        round_index: int = 0,
        batch_index: int = 0,
        visibility: Visibility = Visibility.STUDENT,
        max_chars: int = 8000,
    ):
        self.env = env
        self.policy = policy
        self.bank = bank
        self.budgets = budgets or BudgetSpec()
        self.sampling = sampling or SamplingParams()
        self.context_policy = context_policy or ContextPolicy()
        self.campaign_id = campaign_id
        self.round_index = round_index
        self.batch_index = batch_index
        self.visibility = visibility
        self.max_chars = max_chars

    # ------------------------------------------------------------------
    def run_episode(self, task: PublicTask, seed: int) -> EpisodeResult:
        ledger = BudgetLedger(spec=self.budgets)
        catalog = self.env.tool_catalog()
        init_obs = self.env.reset(task, seed)

        api = MeteredAPI(
            tool_runner=self._component_tool_runner,
            model_caller=self._component_model_caller,
            ledger=ledger,
            visibility=self.visibility.value,
        )
        composition = HarnessComposition(self.bank, api)
        ctx = EpisodeContext(
            episode_id=digest_of({"task": task.task_id, "seed": seed, "bank": self.bank.bank_digest})[:16],
            task=task,
            tool_catalog=catalog,
            rng_seed=seed,
            extra_llm_call_budget=self.budgets.max_extra_llm_calls,
        )
        composition.initialize_all(ctx)

        events: list[Event] = []
        decisions: list[DecisionRecord] = []
        order = 0

        def push(ev: Event) -> None:
            nonlocal order
            events.append(ev)
            order += 1

        push(
            Event(
                event_id=Event.make_id(ctx.episode_id, 0),
                order=0,
                component_id="h0",
                event_kind=EventKind.EPISODE_BOUNDARY,
                pre_state_digest=self.env.state_digest(),
                post_state_digest=self.env.state_digest(),
                observation=init_obs,
                visibility=self.visibility,
                effect_type="state_change",
                information_provenance=("environment:reset",),
            )
        )

        done = False
        final_answer: str | None = None
        terminated_reason = "max_actions"
        t_start = time.perf_counter()

        while not done:
            try:
                ledger.check_wall()
            except BudgetExceeded:
                terminated_reason = "budget_wall"
                break
            if ledger.actions_used >= self.budgets.max_actions:
                terminated_reason = "max_actions"
                break

            pre_state = self.env.state_digest()
            view = PublicView(
                task=task,
                tool_catalog=catalog,
                history=tuple(events),
                remaining_action_budget=ledger.remaining_actions(),
            )
            view = composition.apply_before_model(view)
            v_digest = view_digest(view)
            snap = self.env.snapshot()

            sample, prompt_ids = self._sample_with_network_retry(view, ledger)
            action_token_ids = sample.token_ids
            old_logprobs = sample.logprobs

            try:
                action = parse_action(sample.text)
            except ActionParseError as e:
                ledger.charge_action(1)
                err = Observation(
                    content=f"invalid action format: {e}; respond with one JSON object",
                    ok=False,
                    error_kind="model_invalid_action",
                )
                push(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="h0",
                        event_kind=EventKind.ERROR,
                        pre_state_digest=pre_state,
                        post_state_digest=self.env.state_digest(),
                        observation=err,
                        visibility=self.visibility,
                        effect_type="none",
                        information_provenance=("runtime:parser",),
                        cost=CostRecord(actor_output_tokens=len(action_token_ids)),
                    )
                )
                self._record_decision(
                    decisions,
                    task=task,
                    episode_id=ctx.episode_id,
                    decision_index=len(decisions),
                    snapshot_ref=snap.ref_id,
                    v_digest=v_digest,
                    prompt_ids=prompt_ids,
                    action_token_ids=action_token_ids,
                    old_logprobs=old_logprobs,
                )
                continue

            # Component veto / rewrite of the proposal.
            eff = composition.apply_after_proposal(action, view)
            if eff.veto:
                ledger.charge_action(1)
                veto_obs = Observation(
                    content=f"proposal vetoed: {eff.veto_reason}",
                    ok=False,
                    error_kind="component_veto",
                )
                push(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="h0",
                        event_kind=EventKind.ERROR,
                        pre_state_digest=pre_state,
                        post_state_digest=self.env.state_digest(),
                        observation=veto_obs,
                        visibility=self.visibility,
                    )
                )
                continue
            if eff.action is not None:
                action = eff.action

            ledger.charge_action(1)

            if isinstance(action, Finish):
                fin = composition.apply_before_finish(action, view)
                if fin.block:
                    blocked = Observation(
                        content=f"finish blocked: {fin.block_reason}",
                        ok=False,
                        error_kind="finish_blocked",
                    )
                    push(
                        Event(
                            event_id=Event.make_id(ctx.episode_id, order),
                            order=order,
                            component_id="h0",
                            event_kind=EventKind.ERROR,
                            pre_state_digest=pre_state,
                            post_state_digest=self.env.state_digest(),
                            observation=blocked,
                            visibility=self.visibility,
                        )
                    )
                    self._record_decision(
                        decisions,
                        task=task,
                        episode_id=ctx.episode_id,
                        decision_index=len(decisions),
                        snapshot_ref=snap.ref_id,
                        v_digest=v_digest,
                        prompt_ids=prompt_ids,
                        action_token_ids=action_token_ids,
                        old_logprobs=old_logprobs,
                    )
                    continue
                if fin.action is not None:
                    action = fin.action
                assert isinstance(action, Finish)
                final_answer = action.answer
                result = self.env.step(action)
                push(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="h0",
                        event_kind=EventKind.FINISH,
                        pre_state_digest=pre_state,
                        post_state_digest=self.env.state_digest(),
                        action=action,
                        observation=result.observation,
                        visibility=self.visibility,
                        effect_type="state_change",
                        information_provenance=("model:finish",),
                        cost=CostRecord(actor_output_tokens=len(action_token_ids)),
                    )
                )
                self._record_decision(
                    decisions,
                    task=task,
                    episode_id=ctx.episode_id,
                    decision_index=len(decisions),
                    snapshot_ref=snap.ref_id,
                    v_digest=v_digest,
                    prompt_ids=prompt_ids,
                    action_token_ids=action_token_ids,
                    old_logprobs=old_logprobs,
                )
                done = True
                terminated_reason = "finish"
                break

            assert isinstance(action, ToolCall)
            try:
                ledger.charge_tool_call(1)
            except BudgetExceeded:
                terminated_reason = "budget_tools"
                break
            result = self.env.step(action)
            tool_event = Event(
                event_id=Event.make_id(ctx.episode_id, order),
                order=order,
                component_id="h0",
                event_kind=EventKind.TOOL_CALL,
                pre_state_digest=pre_state,
                post_state_digest=self.env.state_digest(),
                action=action,
                observation=result.observation,
                visibility=self.visibility,
                effect_type="state_change",
                information_provenance=(f"environment:{action.name}",),
                cost=CostRecord(tool_calls=1, actor_output_tokens=len(action_token_ids)),
            )
            push(tool_event)
            composition.apply_after_observation(tool_event)
            self._record_decision(
                decisions,
                task=task,
                episode_id=ctx.episode_id,
                decision_index=len(decisions),
                snapshot_ref=snap.ref_id,
                v_digest=v_digest,
                prompt_ids=prompt_ids,
                action_token_ids=action_token_ids,
                old_logprobs=old_logprobs,
            )
            if result.done:
                done = True
                terminated_reason = "env_done"

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        trace = EpisodeTrace(
            episode_id=ctx.episode_id,
            task_id=task.task_id,
            domain=task.domain,
            policy_digest=self.policy.policy_artifact.policy_id,
            bank_digest=self.bank.bank_digest,
            events=tuple(events),
            decisions=tuple(decisions),
            final_answer=final_answer,
            outcome_reward=None,
            terminated_reason=terminated_reason,
            synthetic=bool(getattr(self.env, "environment_identity", lambda: {})().get("synthetic", False)),
        )
        artifact = None
        if hasattr(self.env, "final_artifact"):
            try:
                artifact = self.env.final_artifact()
            except Exception:  # noqa: BLE001 - artifact is best-effort at runtime level
                artifact = None
        return EpisodeResult(
            trace=trace,
            final_artifact=artifact,
            env_state_digest=self.env.state_digest(),
            budget_snapshot=ledger.snapshot(),
            component_triggers=dict(composition.trigger_report()),
        )

    # ------------------------------------------------------------------
    def _sample_with_network_retry(self, view: PublicView, ledger: BudgetLedger) -> tuple[Any, list[int]]:
        prompt = self.context_policy.render(view)
        prompt_ids = self.policy.encode(prompt)
        last_exc: BaseException | None = None
        for _attempt in range(self.budgets.max_network_retries + 1):
            try:
                return self.policy.sample(prompt_ids, self.sampling), prompt_ids
            except InfraError as e:
                last_exc = e
                try:
                    ledger.charge_network_retry()
                except BudgetExceeded:
                    raise
        raise InfraError(f"network retries exhausted: {last_exc}")

    def _record_decision(
        self,
        decisions: list[DecisionRecord],
        *,
        task: PublicTask,
        episode_id: str,
        decision_index: int,
        snapshot_ref: str,
        v_digest: str,
        prompt_ids: Sequence[int],
        action_token_ids: Sequence[int],
        old_logprobs: Sequence[float],
    ) -> None:
        decisions.append(
            DecisionRecord(
                campaign_id=self.campaign_id,
                round_index=self.round_index,
                batch_index=self.batch_index,
                episode_id=episode_id,
                decision_index=decision_index,
                environment_snapshot_ref=snapshot_ref,
                public_view_digest=v_digest,
                runtime_contract_digest="",  # filled by the caller with the frozen contract
                policy_digest=self.policy.policy_artifact.policy_id,
                tokenizer_digest=self.policy.tokenizer_digest,
                chat_template_digest=self.policy.chat_template_digest,
                bank_digest=self.bank.bank_digest,
                student_prompt_token_ids=tuple(int(i) for i in prompt_ids),
                sampled_action_token_ids=tuple(int(i) for i in action_token_ids),
                old_logprobs=tuple(float(x) for x in old_logprobs),
                action_mask=tuple(1 for _ in action_token_ids),
                behavior_policy_version=self.policy.policy_artifact.policy_id,
                sampling_parameters={
                    "temperature": self.sampling.temperature,
                    "top_p": self.sampling.top_p,
                    "max_new_tokens": self.sampling.max_new_tokens,
                },
            )
        )

    # -- component-callable side channels --------------------------------
    def _component_tool_runner(self, call: ToolCall) -> Observation:
        result = self.env.step(call)
        return result.observation

    def _component_model_caller(self, prompt: str, max_new_tokens: int) -> str:
        ids = self.policy.encode(prompt)
        from rsi.contracts.policy import SamplingParams

        sample = self.policy.sample(ids, SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0))
        return sample.text


def build_runtime_contract(*, env: Any, bank: LoadedBank, budgets: BudgetSpec, policy: Any) -> RuntimeContract:
    identity = env.environment_identity()
    catalog = env.tool_catalog()
    return RuntimeContract(
        h0_source_hash=digest_of({"loop": "MinimalLoop", "version": "h0_v1"}),
        base_prompt_hash=prompt_hash(),
        parser_id="rsi_action_v1",
        context_policy_id=CONTEXT_POLICY_ID,
        base_tool_schema_hash=tool_catalog_digest(catalog),
        visibility_protocol="student_visible_history_v1",
        budgets=budgets.to_json(),
        scoring_protocol="official_per_domain_v1",
        env_identity=dict(identity),
        tokenizer_hash=policy.tokenizer_digest,
        chat_template_hash=policy.chat_template_digest,
    )
