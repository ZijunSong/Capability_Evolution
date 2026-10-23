"""Proposer clients: turn an EvolutionPacket into CandidatePatch objects.

Real model requests only.  Parse failures are recorded and retried a limited
number of times; a fabricated candidate is never returned.  ``self`` mode uses
the current actor policy for code generation; ``assisted`` mode may use an
external stronger coding model and must be labeled as such.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rsi.contracts.artifact import CandidateFile, CandidatePatch
from rsi.contracts.common import JSONValue
from rsi.contracts.errors import CandidateInvalid, InfraError
from rsi.evolution.packet import EvolutionPacket

PROPOSER_VERSION = "rsi_proposer_v1"

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def load_prompt_template(name: str = "proposer_v1.md") -> str:
    root = Path(__file__).resolve().parents[3] / "prompts"
    path = root / name
    if path.is_file():
        return path.read_text("utf-8")
    return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """You evolve an agent harness component library.
Given the parent bank sources and real execution feedback, propose ONE candidate
patch as JSON with keys: parent_bank_digest, operation (add|edit|compose|prune|replace),
hypothesis, files ([{path, content}]), delete_paths, manifest_patch, alignment_hints,
evidence_refs, expected_cost_change, self_checks.
Write real Python code in files[].content. Do not reference dataset task ids in code.
"""


def render_proposer_prompt(packet: EvolutionPacket, template: str | None = None) -> str:
    tpl = template or load_prompt_template()
    payload = packet.to_json()
    # Keep parent sources inline: the proposer must read real parent code.
    return (
        tpl
        + "\n\n=== EVOLUTION PACKET (JSON) ===\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n\nRespond with a single JSON object for the CandidatePatch now."
    )


def parse_candidate_patch(text: str, *, parent_bank_digest: str) -> CandidatePatch:
    """Strictly parse one CandidatePatch from model output.  Never fabricates."""
    m = _JSON_BLOCK.search(text)
    body = m.group(1) if m else None
    if body is None:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            body = stripped
    if body is None:
        raise CandidateInvalid("proposer output contains no JSON object")
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise CandidateInvalid(f"proposer JSON invalid: {e}") from e
    if not isinstance(obj, dict):
        raise CandidateInvalid("proposer JSON must be an object")
    obj.setdefault("parent_bank_digest", parent_bank_digest)
    if obj.get("parent_bank_digest") != parent_bank_digest:
        raise CandidateInvalid(
            f"candidate parent digest mismatch: {obj.get('parent_bank_digest')} != {parent_bank_digest}"
        )
    patch = CandidatePatch.from_json(obj)
    if patch.operation != "prune" and not patch.files and not patch.manifest_patch:
        raise CandidateInvalid("candidate must contain real files or a manifest_patch")
    return patch


class Proposer(Protocol):
    mode: str  # self | assisted | fixture

    def propose(self, packet: EvolutionPacket) -> CandidatePatch: ...


@dataclass
class APIProposer:
    """Assisted mode: external coding model over an OpenAI-compatible API."""

    client: Any
    max_retries: int = 2
    mode: str = "assisted"
    temperature: float = 0.7

    def propose(self, packet: EvolutionPacket) -> CandidatePatch:
        from rsi.providers.openai_compat import APIMessage

        prompt = render_proposer_prompt(packet)
        last_err: BaseException | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                out = self.client.chat(
                    [APIMessage(role="user", content=prompt)],
                    temperature=self.temperature,
                )
            except InfraError as e:
                last_err = e
                continue
            try:
                return parse_candidate_patch(out, parent_bank_digest=packet.parent_bank_digest)
            except CandidateInvalid as e:
                last_err = e
                continue
        raise CandidateInvalid(f"proposer failed after retries: {last_err!r}")


@dataclass
class SelfProposer:
    """Self mode: the current actor policy generates the candidate code.

    For HF policies generation goes through the same provider interface used
    for task rollouts (batch-synchronized proposer reload is the caller's job).
    ``max_new_tokens`` bounds the generation budget per attempt (a candidate
    patch is a JSON object, not a novel).
    """

    policy: Any
    max_retries: int = 2
    mode: str = "self"
    max_new_tokens: int = 1024

    def propose(self, packet: EvolutionPacket) -> CandidatePatch:
        from rsi.contracts.policy import SamplingParams

        prompt = render_proposer_prompt(packet)
        prompt_ids = self.policy.encode(prompt)
        last_err: BaseException | None = None
        for _attempt in range(self.max_retries + 1):
            sample = self.policy.sample(
                prompt_ids, SamplingParams(temperature=0.7, max_new_tokens=self.max_new_tokens)
            )
            try:
                return parse_candidate_patch(sample.text, parent_bank_digest=packet.parent_bank_digest)
            except CandidateInvalid as e:
                last_err = e
                continue
        raise CandidateInvalid(f"self proposer failed after retries: {last_err!r}")


@dataclass
class FixtureProposer:
    """Test-only proposer emitting prebuilt patches.

    Fixed fixtures validate code paths (``test_real_code_materialization``)
    and are never research results.
    """

    patches: Sequence[CandidatePatch]
    mode: str = "fixture"
    _i: int = 0

    def propose(self, packet: EvolutionPacket) -> CandidatePatch:
        if self._i >= len(self.patches):
            raise CandidateInvalid("fixture proposer exhausted")
        patch = self.patches[self._i]
        self._i += 1
        return patch
