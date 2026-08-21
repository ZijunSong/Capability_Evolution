from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 4096
    top_p: float = 0.9
    stop: list[int] | None = None


@dataclass
class ModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        return cls(tokens=list(tokens))

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(tokens=[])

    def to_ints(self) -> list[int]:
        return list(self.tokens)


@dataclass
class _Sequence:
    tokens: list[int]


@dataclass
class SampleResponse:
    sequences: list[_Sequence]


class _SampleFuture:
    def __init__(self, response: SampleResponse):
        self._response = response

    def result(self, timeout: int | None = None) -> SampleResponse:
        return self._response


class SamplingClient:
    def sample(self, *, prompt: ModelInput, sampling_params: SamplingParams, num_samples: int = 1):
        if num_samples != 1:
            raise NotImplementedError("tinker shim only supports num_samples=1")
        return _SampleFuture(SampleResponse(sequences=[_Sequence(tokens=list(prompt.tokens))]))


class ServiceClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def create_sampling_client(self, *args: Any, **kwargs: Any) -> SamplingClient:
        return SamplingClient()


types = SimpleNamespace(
    SamplingParams=SamplingParams,
    ModelInput=ModelInput,
    SampleResponse=SampleResponse,
)
