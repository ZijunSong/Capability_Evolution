"""Deterministic scripted policy.

Used for protocol smoke tests and toy closed-loop dry runs.  Every artifact
produced with a ScriptedPolicy carries ``synthetic=True`` markers downstream;
report merge refuses to fold them into formal results.  It is never a research
result (task book §15).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall, encode_action
from rsi.contracts.policy import PolicyArtifact, SamplingParams
from rsi.providers.base import SampleResult


class ScriptedPolicy:
    """A policy that emits actions from a callable plan (or a default plan).

    ``plan(view_text, step) -> Action`` is fully deterministic.  Token ids are
    byte-level encodings of the emitted action text (identity is stable).
    """

    def __init__(
        self,
        plan: Callable[[str, int], Action] | None = None,
        *,
        base_model_id: str = "scripted",
    ):
        self._plan = plan or self._default_plan
        self._step = 0
        self.policy_artifact = PolicyArtifact.make(base_model_id, kind="base", extra={"synthetic": True})
        self.tokenizer_digest = "scripted_byte_tokenizer_v1"
        self.chat_template_digest = "scripted_render_v1"

    @staticmethod
    def _default_plan(view_text: str, step: int) -> Action:
        # Extremely small generic heuristic for toy_file: read a.txt / b.txt
        # then write sum.  Marked synthetic; not a scientific baseline.
        if "a.txt" not in view_text:
            return Finish(answer=None)
        if "observation:" not in view_text:
            return ToolCall(name="read_file", arguments={"path": "a.txt"})
        return Finish(answer=None)

    def encode(self, text: str) -> list[int]:
        return [b + 1 for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        return bytes((i - 1) & 0xFF for i in ids).decode("utf-8", errors="replace")

    def sample(self, prompt_ids: Sequence[int], params: SamplingParams) -> SampleResult:
        text_view = self.decode(prompt_ids)
        action = self._plan(text_view, self._step)
        self._step += 1
        text = encode_action(action)
        ids = tuple(self.encode(text))
        return SampleResult(text=text, token_ids=ids, logprobs=(0.0,) * len(ids), finish_reason="stop")

    def teacher_force_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        return [0.0] * len(target_ids)

    def freeze_for_batch(self) -> PolicyArtifact:
        return self.policy_artifact


class FixedActionPolicy(ScriptedPolicy):
    """Emits a fixed sequence of actions (tests: fixed-artifact official-scorer
    parity).  After the sequence is exhausted it finishes."""

    def __init__(self, actions: Sequence[Action], *, base_model_id: str = "fixed"):
        self._actions = list(actions)
        self._i = 0
        self.policy_artifact = PolicyArtifact.make(base_model_id, kind="base", extra={"synthetic": True})
        self.tokenizer_digest = "scripted_byte_tokenizer_v1"
        self.chat_template_digest = "scripted_render_v1"

        def plan(view_text: str, step: int) -> Action:
            if self._i < len(self._actions):
                act = self._actions[self._i]
                self._i += 1
                return act
            return Finish(answer=None)

        self._plan = plan
