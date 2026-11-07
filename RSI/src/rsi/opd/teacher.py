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
from rsi.runtime.clock import monotonic_ms


@dataclass
class TeacherBranch:
    events: tuple[Event, ...]
    context_ref: str
    snapshot_ref: str
    triggered_components: Mapping[str, int]
    cost: CostRecord
    terminated_reason: str
    # PR-09 (guide §8.1): number of events in ``events`` that existed BEFORE
    # the first proposed target action was sampled.  The scoring prefix is
    # events[:pre_target_event_count] — never the full branch (which contains
    # the target's own execution feedback, letting the teacher "see the
    # answer" and inflating the gate).
    pre_target_event_count: int = 0

    @property
    def native_actions(self) -> tuple[Action, ...]:
        return tuple(e.action for e in self.events if e.action is not None)

    @property
    def pre_target_events(self) -> tuple[Event, ...]:
        """Events strictly before the first proposed target action."""
        return self.events[: self.pre_target_event_count]


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
        t0 = monotonic_ms()
        terminated = "max_shadow_events"
        # PR-09 (guide §8.1): the target-generation prompt is the state at the
        # decision point itself — i.e. the student prefix + restored component
        # state, BEFORE any teacher event.  The first teacher sample of the
        # branch IS the candidate target, so pre_target_event_count starts at
        # 0: every event in this branch happened at-or-after the target
        # proposal.  (Branches that need internal retries record those events
        # here too — the scoring view still must not include them unless they
        # causally precede the accepted target.)
        pre_target_event_count = 0
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
                # PR-07: a parse error consumed a model call — record it AND
                # charge it, so malformed-output loops hit the shadow budget
                # instead of spinning silently.
                order += 0  # ordering handled by the append below
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
                # charge the wasted action so parse-error loops are bounded
                ledger.charge_action(1)
                continue
            eff = composition.apply_after_proposal(action, view)
            if eff.veto:
                # PR-07 (guide §6.4): a veto must generate private FEEDBACK and
                # consume budget — the pre-fix silent ``continue`` allowed an
                # infinite free resample loop and hid the veto from the
                # teacher's own next-prompt history.
                events.append(
                    Event(
                        event_id=Event.make_id(ctx.episode_id, order),
                        order=order,
                        component_id="teacher",
                        event_kind=EventKind.ERROR,
                        pre_state_digest=fork_env.state_digest(),
                        post_state_digest=fork_env.state_digest(),
                        observation=None,
                        data={"veto_reason": eff.veto_reason, "vetoed_action": str(action)[:200]},
                        visibility=Visibility.TEACHER_PRIVATE,
                        cost=CostRecord(actor_output_tokens=len(sample.token_ids)),
                    )
                )
                order += 1
                view = PublicView(
                    task=task,
                    tool_catalog=catalog,
                    history=tuple(student_events) + tuple(events),
                    remaining_action_budget=self.max_shadow_events - order,
                )
                continue
            if eff.action is not None:
                action = eff.action
            if isinstance(action, Finish):
                # PR-07 (guide §6.4): the teacher MUST run the same
                # before_finish gating as the student runtime — a Finish that
                # the bank's finish-guard would block is not a branch
                # termination (and must never become supervision).
                fin = composition.apply_before_finish(action, view)
                if fin.block:
                    events.append(
                        Event(
                            event_id=Event.make_id(ctx.episode_id, order),
                            order=order,
                            component_id="teacher",
                            event_kind=EventKind.ERROR,
                            pre_state_digest=fork_env.state_digest(),
                            post_state_digest=fork_env.state_digest(),
                            data={"finish_blocked": fin.block_reason},
                            visibility=Visibility.TEACHER_PRIVATE,
                            cost=CostRecord(actor_output_tokens=len(sample.token_ids)),
                        )
                    )
                    order += 1
                    continue
                if fin.action is not None:
                    action = fin.action
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
        wall_ms = monotonic_ms() - t0
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
            pre_target_event_count=pre_target_event_count,
        )

    def _aux_model(self, prompt: str, max_new_tokens: int) -> str:
        from rsi.contracts.policy import SamplingParams as SP

        ids = self.policy.encode(prompt)
        sample = self.policy.sample(ids, SP(max_new_tokens=max_new_tokens, temperature=0.0))
        return sample.text
