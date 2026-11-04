from rsi.harness.component import (
    EpisodeContext, FinishEffect, ProposalEffect, PublicView, RuntimeAPI,
    StatePatch, ViewPatch,
)
from rsi.contracts.action import ToolCall, Finish
from rsi.contracts.events import Event


class ActionRecoveryGuard:
    """
    Prevents the model from proposing invalid/None actions when stuck or after errors.
    1. before_model: If history is empty or ends with an error, inject specific remediation instructions
       to force a valid ToolCall generation.
    2. after_proposal: Veto any action that is None or not a ToolCall (double safety).
    3. after_observation: Track recent error state to trigger recovery.
    """

    def initialize(self, ctx: EpisodeContext) -> None:
        self.recent_error = False
        self.history_length = 0

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        # Check if history is empty or the last event was an error
        # view.history is a list of Event objects
        if len(view.history) == 0:
            # First step: often fails if the model doesn't know to call a tool
            return ViewPatch(
                append_instructions="You are starting fresh. Analyze the task instruction and immediately propose a ToolCall to gather information or proceed. Do not propose 'Finish' yet. Ensure the action is a valid JSON object with 'name', 'type', and 'arguments'."
            )
        
        # Check last event for errors
        last_event = view.history[-1]
        obs = last_event.observation
        
        if obs is not None and not obs.ok:
            self.recent_error = True
            return ViewPatch(
                append_instructions="Previous step failed (error detected). Do not finalize the task. Analyze the error message and propose a corrected ToolCall to resolve the issue or retry the action."
            )
        
        # Reset flag if observation was okay
        if obs is not None and obs.ok:
            self.recent_error = False
            
        return ViewPatch(changed=False)

    def after_proposal(self, action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        # Double check: ensure action is a ToolCall and not None
        if action is None:
            return ProposalEffect(
                veto=True, 
                veto_reason="Action proposal was None. This is invalid. Please propose a ToolCall with 'name', 'type', and 'arguments'."
            )
        
        if not isinstance(action, ToolCall):
            return ProposalEffect(
                veto=True, 
                veto_reason=f"Action proposal type {type(action)} is not a ToolCall. Please propose a ToolCall or Finish action."
            )
            
        # Verify the tool name exists in the catalog (basic validation)
        tool_names = {t.name for t in view.tool_catalog}
        if action.name not in tool_names:
            return ProposalEffect(
                veto=True, 
                veto_reason=f"Tool '{action.name}' is not available in the tool catalog. Available tools: {', '.join(tool_names)}."
            )
            
        return ProposalEffect(action=None)

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch:
        obs = event.observation
        if obs is not None:
            if obs.ok:
                self.recent_error = False
                api.log("Recovery: Observation succeeded", obs_content_len=len(obs.content))
            else:
                self.recent_error = True
                api.log("Recovery: Observation failed", error_kind=obs.error_kind)
        return StatePatch()

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        # If there was a recent error, block finishing unless the user explicitly answered
        # This complements the GoalCompletionGuard
        if self.recent_error and action.answer is None:
            return FinishEffect(
                block=True, 
                block_reason="Recent observation failed. Please attempt a recovery action before finishing."
            )
        return FinishEffect(block=False)
