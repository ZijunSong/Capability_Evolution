"""Proposer boundary. A missing endpoint does not silently become a stronger model."""

from __future__ import annotations

from typing import Any, Protocol

from trim.evolution.operators import Proposal


class ProposerError(RuntimeError):
    pass


class Proposer(Protocol):
    def propose(self, packet: dict[str, Any]) -> list[Proposal]: ...


class ProposerClient:
    """Reads an explicit endpoint. ``assisted`` must be configured, not inferred."""

    def __init__(self, *, mode: str, endpoint: str | None, model: str) -> None:
        if mode not in {"self", "assisted"}:
            raise ProposerError(f"proposer mode {mode!r} is not self or assisted")
        self.mode = mode
        self.endpoint = endpoint or ""
        self.model = model

    def propose(self, packet: dict[str, Any]) -> list[Proposal]:
        del packet
        if not self.endpoint:
            raise ProposerError(
                "no proposer endpoint is configured; refusing to invent components or "
                "silently switch to an external model"
            )
        raise ProposerError(
            "live proposer HTTP calls are not part of this process; "
            f"mode={self.mode} model={self.model} endpoint is set but the client is plan-only"
        )


class ScriptedProposer:
    def __init__(self, proposals: list[Proposal]) -> None:
        self._proposals = list(proposals)

    def propose(self, packet: dict[str, Any]) -> list[Proposal]:
        del packet
        if not self._proposals:
            return []
        item = self._proposals.pop(0)
        return [item]
