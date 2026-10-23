"""Policy provider protocol.

The provider owns inference (generation + teacher forcing) and exposes the
token identity of what actually entered the model.  Training backends consume
the *recorded* token ids from DecisionRecords — never a re-assembled prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from rsi.contracts.policy import PolicyArtifact, SamplingParams


@dataclass(frozen=True)
class SampleResult:
    text: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    finish_reason: str = "stop"


@runtime_checkable
class PolicyProvider(Protocol):
    policy_artifact: PolicyArtifact
    tokenizer_digest: str
    chat_template_digest: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def sample(self, prompt_ids: Sequence[int], params: SamplingParams) -> SampleResult: ...

    def teacher_force_logprobs(
        self, prompt_ids: Sequence[int], target_ids: Sequence[int]
    ) -> list[float]: ...

    def freeze_for_batch(self) -> PolicyArtifact: ...


def action_mask_for(token_ids: Sequence[int]) -> tuple[int, ...]:
    """Default mask: every generated action token is supervised.

    EOS/pad semantics are defined by the provider's tokenizer contract and
    tested in ``test_exact_prompt_ids``.
    """
    return tuple(1 for _ in token_ids)
