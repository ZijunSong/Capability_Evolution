"""Teacher fork: run ``H0 + Bk`` on a forked environment at Student decision
points (task book §8.1).

The Teacher runs on the *same* frozen policy snapshot as the Student batch
(batch-synchronized self-distillation; whole-round frozen teacher is a separate
declared ablation).  Teacher branches never replace Student actions and never
write shadow tool results back into the Student environment.  Everything a
Teacher branch produces is ``TEACHER_PRIVATE`` until alignment promotes a
*Student-native* action to supervision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall, parse_action, ActionParseError
from rsi.contracts.common import digest_of
from rsi.contracts.errors import BudgetExceeded, InfraError
from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
from rsi.contracts.policy import SamplingParams
from rsi.contracts.task import PublicTask
from rsi.harness.composition import HarnessComposition
from rsi.harness.component import EpisodeContext, PublicView
from rsi.harness.loader import LoadedBank
from rsi.runtime.api import MeteredAPI
from rsi.runtime.budgets import BudgetLedger, BudgetSpec
from rsi.runtime.context import ContextPolicy


@dataclass
class TeacherBranch:
    events: tuple[Event, ...]
    context_ref: str
    snapshot_ref: str
    triggered_components: Mapping[str, int]
    cost: CostRecord
    terminated_reason: str

    @property
    def native_actions(self) -> tuple[Action, ...]:
        return tuple(e.action for e in self.events if e.action is not None)


@dataclass
class TeacherForkRunner:
    """Executes isolated Teacher branches.  ``env_factory`` must return an
    adapter whose ``fork(snapshot)`` produces an independent world; the runner
    asserts the base env digest is unchanged after every branch."""

    env_factory: Any
    policy: Any
    bank: LoadedBank
    budgets: BudgetSpec = field(default_factory=BudgetSpec)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    max_shadow_events: int = 8

    def __post_init__(self) -> None:
        if self.sampling is None:
            self.sampling = SamplingParams()
        if self.budgets is None:
            self.budgets = BudgetSpec()

    def run_branch(
        self,
        *,
        base_env: Any,
        snapshot_ref: Any,
        task: PublicTask,
        student_events: Sequence[Event],
        decision_index: int,
    ) -> TeacherBranch:
        pre_digest = base_env.state_digest()
        fork_env = base_env.fork(snapshot_ref)
        ledger = BudgetLedger(spec=self.budgets)
        catalog = fork_env.tool_catalog()

        api = MeteredAPI(
            tool_runner=lambda call: fork_env.step(call).observation,
            model_caller=self._aux_model,
            ledger=ledger,
            visibility=Visibility.TEACHER_PRIVATE.value,
        )
        composition = HarnessComposition(self.bank, api)
        ctx = EpisodeContext(
            episode_id=f"teacher-{decision_index}",
            task=task,
            tool_catalog=catalog,
            extra_llm_call_budget=self.budgets.max_extra_llm_calls,
        )
        composition.initialize_all(ctx)

        events: list[Event] = []
        order = 0
        t0 = time.perf_counter()
        terminated = "max_shadow_events"
        view = PublicView(
            task=task,
            tool_catalog=catalog,
            history=tuple(student_events),  # legal Student prefix as starting history
            remaining_action_budget=self.max_shadow_events,
        )

        for _step in range(self.max_shadow_events):
            try:
                ledger.check_wall()
            except BudgetExceeded:
                terminated = "budget_wall"
                break
            view = composition.apply_before_model(view)
            prompt = self.context_policy.render(view)
            prompt_ids = self.policy.encode(prompt)
            try:
                sample = self.policy.sample(prompt_ids, self.sampling)
            except InfraError:
                terminated = "infra"
                break
            try:
                action = parse_action(sample.text)
            except ActionParseError as e:
                events.append(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="teacher",
                        event_kind=EventKind.ERROR,
                        pre_state_digest=fork_env.state_digest(),
                        post_state_digest=fork_env.state_digest(),
                        visibility=Visibility.TEACHER_PRIVATE,
                        data={"error": str(e)},
                        cost=CostRecord(actor_output_tokens=len(sample.token_ids)),
                    )
                )
                order += 1
                continue
            eff = composition.apply_after_proposal(action, view)
            if eff.veto:
                continue
            if eff.action is not None:
                action = eff.action
            if isinstance(action, Finish):
                events.append(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="teacher",
                        event_kind=EventKind.FINISH,
                        pre_state_digest=fork_env.state_digest(),
                        post_state_digest=fork_env.state_digest(),
                        action=action,
                        visibility=Visibility.TEACHER_PRIVATE,
                        cost=CostRecord(actor_output_tokens=len(sample.token_ids)),
                    )
                )
                order += 1
                terminated = "finish"
                break
            assert isinstance(action, ToolCall)
            result = fork_env.step(action)
            ev = Event(
                event_id=Event.make_id(ctx.episode_id, order),
                order=order,
                component_id="teacher",
                event_kind=EventKind.TOOL_CALL,
                pre_state_digest=fork_env.state_digest(),
                post_state_digest=fork_env.state_digest(),
                action=action,
                observation=result.observation,
                visibility=Visibility.TEACHER_PRIVATE,
                information_provenance=(f"environment:{action.name}",),
                cost=CostRecord(tool_calls=1, actor_output_tokens=len(sample.token_ids)),
            )
            events.append(ev)
            composition.apply_after_observation(ev)
            order += 1
            # The Teacher's view advances within the branch; Student history is
            # never modified.
            view = PublicView(
                task=task,
                tool_catalog=catalog,
                history=tuple(student_events) + tuple(events),
                remaining_action_budget=self.max_shadow_events - order,
            )

        # Isolation assertion: the base env must be untouched by the branch.
        post_digest = base_env.state_digest()
        if post_digest != pre_digest:
            raise RuntimeError(
                f"teacher branch mutated the base environment: {pre_digest} -> {post_digest}"
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cost = CostRecord(
            actor_output_tokens=sum(e.cost.actor_output_tokens for e in events),
            tool_calls=sum(e.cost.tool_calls for e in events),
            wall_ms=wall_ms,
        )
        context_ref = digest_of(
            {"snapshot": str(snapshot_ref), "decision": decision_index, "n_events": len(events)}
        )[:16]
        return TeacherBranch(
            events=tuple(events),
            context_ref=context_ref,
            snapshot_ref=str(snapshot_ref),
            triggered_components=dict(composition.trigger_report()),
            cost=cost,
            terminated_reason=terminated,
        )

    def _aux_model(self, prompt: str, max_new_tokens: int) -> str:
        from rsi.contracts.policy import SamplingParams as SP

        ids = self.policy.encode(prompt)
        sample = self.policy.sample(ids, SP(max_new_tokens=max_new_tokens, temperature=0.0))
        return sample.text
