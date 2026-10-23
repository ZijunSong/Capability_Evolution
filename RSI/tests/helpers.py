"""Shared test helpers.

Everything here is scaffolding for code-path validation (marked synthetic);
none of it is a research result.  ``HybridPolicy`` samples from a deterministic
plan so rollouts are meaningful on toy tasks, while teacher-forced logprobs go
through a real tiny HF model so training updates are real gradients.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from rsi.contracts.action import Action, Finish, ToolCall, encode_action
from rsi.contracts.artifact import CandidateFile, CandidatePatch
from rsi.contracts.policy import PolicyArtifact, SamplingParams
from rsi.providers.base import SampleResult
from rsi.providers.tiny_hf import HFPolicy, build_tiny_model

_OBS_RE = re.compile(r"\[\d+\] observation: (.*)")


def toy_file_solver_plan(view_text: str, step: int) -> Action:
    """Deterministic generic solver for toy_file arithmetic tasks."""
    obs = _OBS_RE.findall(view_text)[1:]  # skip the reset observation
    n = len(obs)
    if n == 0:
        return ToolCall(name="read_file", arguments={"path": "a.txt"})
    if n == 1:
        return ToolCall(name="read_file", arguments={"path": "b.txt"})
    if n == 2:
        a, b = obs[-2], obs[-1]
        try:
            v = str(int(a) + int(b))
        except ValueError:
            v = a + b
        return ToolCall(name="write_file", arguments={"path": "out.txt", "content": v})
    return Finish(answer=None)


def toy_file_poor_plan(view_text: str, step: int) -> Action:
    """A policy that always finishes immediately (baseline for harness gain)."""
    return Finish(answer=None)


class HybridPolicy:
    """Scripted sampling + real HF teacher-forcing (synthetic test scaffolding)."""

    def __init__(self, plan=toy_file_solver_plan, *, model: Any | None = None, base_model_id: str = "hybrid-tiny"):
        self._plan = plan
        self._hf = HFPolicy(model if model is not None else build_tiny_model(), base_model_id=base_model_id)
        self.policy_artifact: PolicyArtifact = self._hf.policy_artifact
        self.tokenizer_digest = self._hf.tokenizer_digest
        self.chat_template_digest = self._hf.chat_template_digest

    # PolicyProvider ----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._hf.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._hf.decode(ids)

    def sample(self, prompt_ids: Sequence[int], params: SamplingParams) -> SampleResult:
        text_view = self.decode(prompt_ids)
        action = self._plan(text_view, 0)
        text = encode_action(action)
        ids = tuple(self.encode(text))
        # Old logprobs from the real model (teacher-forced on the emitted action)
        lp = tuple(self._hf.teacher_force_logprobs(list(prompt_ids), list(ids)))
        return SampleResult(text=text, token_ids=ids, logprobs=lp, finish_reason="stop")

    def teacher_force_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        return self._hf.teacher_force_logprobs(prompt_ids, target_ids)

    def freeze_for_batch(self) -> PolicyArtifact:
        return self.policy_artifact

    @property
    def model(self) -> Any:
        return self._hf.model


# ---------------------------------------------------------------------------
# fixture candidate: a real component with real code
# ---------------------------------------------------------------------------

TOY_FINALIZER_SOURCE = '''
"""Evolved-style toy component: artifact finalizer + premature-finish gate.

Parses observations from the visible history; when the task looks like an
arithmetic-write task and out.txt is missing, it synthesizes out.txt through
the controlled tool API before allowing finish.  Marked synthetic in tests.
"""

from __future__ import annotations

import re
from typing import Any

from rsi.contracts.action import Finish, ToolCall
from rsi.contracts.events import Event
from rsi.harness.component import EpisodeContext, FinishEffect, ProposalEffect, PublicView, RuntimeAPI, StatePatch, ViewPatch

_OBS = re.compile(r"\\[\\d+\\] observation: (.*)")


class ToyFinalizer:
    def initialize(self, ctx: EpisodeContext) -> None:
        self._state = {"reads": []}

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        return ViewPatch(changed=False)

    def after_proposal(self, action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        return ProposalEffect(action=None)

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch:
        return StatePatch()

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        history_text = "\\n".join(
            f"[{e.order}] observation: {e.observation.content}"
            for e in view.history
            if e.observation is not None
        )
        obs = _OBS.findall(history_text)
        wrote = any("wrote" in o for o in obs)
        if wrote or len(obs) < 2:
            return FinishEffect(block=False)
        a, b = obs[-2], obs[-1]
        try:
            value = str(int(a) + int(b))
        except ValueError:
            value = a + b
        api.call_tool(ToolCall(name="write_file", arguments={"path": "out.txt", "content": value}))
        return FinishEffect(block=False)
'''


def toy_finalizer_patch(parent_bank_digest: str) -> CandidatePatch:
    """A fixture CandidatePatch that adds the ToyFinalizer component (real code)."""
    manifest_patch = {
        "id": "toy_finalizer",
        "version": "1",
        "files": ["components/toy_finalizer.py"],
        "entrypoint": "components.toy_finalizer:ToyFinalizer",
        "depends_on": [],
        "conflicts_with": [],
        "hook_order": 50,
        "required_capabilities": [],
        "state_schema": {},
        "resource_caps": {"max_source_bytes": 200000},
        "provenance": {"kind": "fixture", "synthetic": True},
        "alignment_hints": {},
        "kind": "decision_hook",
    }
    return CandidatePatch(
        parent_bank_digest=parent_bank_digest,
        operation="add",
        hypothesis="finalizer writes missing out.txt from visible observations before finish",
        files=(CandidateFile(path="components/toy_finalizer.py", content=TOY_FINALIZER_SOURCE),),
        manifest_patch=manifest_patch,
    )
