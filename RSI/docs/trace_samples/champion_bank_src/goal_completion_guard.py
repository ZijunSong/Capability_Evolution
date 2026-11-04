from rsi.harness.component import (
    EpisodeContext, FinishEffect, ProposalEffect, PublicView, RuntimeAPI,
    StatePatch, ViewPatch,
)
from rsi.contracts.action import ToolCall, Finish
from rsi.contracts.events import Event


class GoalCompletionGuard:
    """
    Prevents premature finishing and enforces valid action proposals.
    1. after_proposal: Veto if action is None (prevents 'unknown action type None').
    2. before_finish: Block if last observation is not OK and task not explicitly done.
    3. after_observation: Track success state.
    """

    def initialize(self, ctx: EpisodeContext) -> None:
        self.last_obs_ok = False
        self.last_obs_content = ""

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        # Guidance to steer model towards tool use if stuck
        if not self.last_obs_ok:
            return ViewPatch(
                append_instructions="Focus on extracting information or performing an action. Do not finalize unless the task is explicitly completed."
            )
        return ViewPatch(changed=False)

    def after_proposal(self, action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        # CRITICAL FIX: Prevent 'unknown action type None' crashes
        if action is None:
            return ProposalEffect(
                veto=True, 
                veto_reason="Action proposal was None, which is invalid. Please propose a ToolCall or Finish with an answer."
            )
        return ProposalEffect(action=None)

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch:
        obs = event.observation
        if obs is not None:
            self.last_obs_ok = obs.ok
            self.last_obs_content = obs.content
            if obs.ok:
                # Log success for potential finish gating
                api.log("Observation succeeded", obs_content_len=len(obs.content))
            else:
                # Log error to trigger guidance
                api.log("Observation failed", error_kind=obs.error_kind)
        return StatePatch()

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        # If the last observation was an error, block finishing unless the user explicitly answered
        # This prevents 'terminated=finish value=0.0' on failed attempts
        if self.last_obs_ok is False and action.answer is None:
            return FinishEffect(
                block=True, 
                block_reason="Last observation was an error. Please try a different approach or provide an answer."
            )
        return FinishEffect(block=False)
