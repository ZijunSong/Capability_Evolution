"""Bank / candidate artifacts: manifests, patches, provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of, digest_of_text

MANIFEST_SCHEMA = "rsi_bank_manifest_v1"
CANDIDATE_PATCH_SCHEMA = "rsi_candidate_patch_v1"

# Hook names of the HarnessComponent protocol, in pipeline order.
HOOK_NAMES = ("initialize", "before_model", "after_proposal", "after_observation", "before_finish")
COMPONENT_KINDS = (
    "tool_wrapper",
    "tool_composition",
    "context_policy",
    "decision_hook",
    "workflow",
    "task_memory",
    "error_recovery",
    "candidate_validation",
    "aux_model_call",
    "other",
)


@dataclass(frozen=True)
class ComponentManifest:
    """Manifest entry for one evolvable component.

    ``kind`` is an extension-interface category, not a semantic capability
    whitelist.  ``alignment_hints`` may suggest mapping/recipes, but only an
    independent projector decides ``distillable``.
    """

    id: str
    version: str
    files: tuple[str, ...]
    entrypoint: str  # "module:ClassName" relative to the bank root
    depends_on: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    hook_order: int = 100
    required_capabilities: tuple[str, ...] = ()
    state_schema: Mapping[str, JSONValue] = field(default_factory=dict)
    resource_caps: Mapping[str, JSONValue] = field(default_factory=dict)
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)
    alignment_hints: Mapping[str, JSONValue] = field(default_factory=dict)
    kind: str = "other"

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "version": self.version,
            "files": list(self.files),
            "entrypoint": self.entrypoint,
            "depends_on": list(self.depends_on),
            "conflicts_with": list(self.conflicts_with),
            "hook_order": self.hook_order,
            "required_capabilities": list(self.required_capabilities),
            "state_schema": dict(self.state_schema),
            "resource_caps": dict(self.resource_caps),
            "provenance": dict(self.provenance),
            "alignment_hints": dict(self.alignment_hints),
            "kind": self.kind,
        }

    @staticmethod
    def from_json(obj: Mapping[str, JSONValue]) -> "ComponentManifest":
        return ComponentManifest(
            id=str(obj["id"]),
            version=str(obj.get("version", "0")),
            files=tuple(str(f) for f in obj.get("files", [])),  # type: ignore[arg-type]
            entrypoint=str(obj["entrypoint"]),
            depends_on=tuple(str(d) for d in obj.get("depends_on", [])),  # type: ignore[arg-type]
            conflicts_with=tuple(str(d) for d in obj.get("conflicts_with", [])),  # type: ignore[arg-type]
            hook_order=int(obj.get("hook_order", 100)),
            required_capabilities=tuple(str(c) for c in obj.get("required_capabilities", [])),  # type: ignore[arg-type]
            state_schema=dict(obj.get("state_schema") or {}),  # type: ignore[arg-type]
            resource_caps=dict(obj.get("resource_caps") or {}),  # type: ignore[arg-type]
            provenance=dict(obj.get("provenance") or {}),  # type: ignore[arg-type]
            alignment_hints=dict(obj.get("alignment_hints") or {}),  # type: ignore[arg-type]
            kind=str(obj.get("kind", "other")),
        )


@dataclass(frozen=True)
class BankManifest:
    """A bank directory's manifest.  ``banks/empty`` has all lists empty."""

    bank_id: str
    schema_version: str = MANIFEST_SCHEMA
    components: tuple[ComponentManifest, ...] = ()
    hooks: tuple[JSONValue, ...] = ()
    tools: tuple[JSONValue, ...] = ()
    skills: tuple[JSONValue, ...] = ()
    parent_digest: str | None = None
    extra: Mapping[str, JSONValue] = field(default_factory=dict)

    def digest(self) -> str:
        return digest_of(
            {
                "schema_version": self.schema_version,
                "bank_id": self.bank_id,
                "components": [c.to_json() for c in self.components],
                "hooks": [dict(h) for h in self.hooks],  # type: ignore[arg-type]
                "tools": [dict(t) for t in self.tools],  # type: ignore[arg-type]
                "skills": [dict(s) for s in self.skills],  # type: ignore[arg-type]
                "extra": dict(self.extra),
            }
        )

    def is_empty(self) -> bool:
        return not (self.components or self.hooks or self.tools or self.skills)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "components": [c.to_json() for c in self.components],
            "hooks": [dict(h) for h in self.hooks],  # type: ignore[arg-type]
            "tools": [dict(t) for t in self.tools],  # type: ignore[arg-type]
            "skills": [dict(s) for s in self.skills],  # type: ignore[arg-type]
            "parent_digest": self.parent_digest,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_json(obj: Mapping[str, JSONValue]) -> "BankManifest":
        return BankManifest(
            bank_id=str(obj["bank_id"]),
            schema_version=str(obj.get("schema_version", MANIFEST_SCHEMA)),
            components=tuple(ComponentManifest.from_json(c) for c in obj.get("components", [])),  # type: ignore[arg-type]
            hooks=tuple(dict(h) for h in obj.get("hooks", [])),  # type: ignore[arg-type]
            tools=tuple(dict(t) for t in obj.get("tools", [])),  # type: ignore[arg-type]
            skills=tuple(dict(s) for s in obj.get("skills", [])),  # type: ignore[arg-type]
            parent_digest=obj.get("parent_digest"),  # type: ignore[arg-type]
            extra=dict(obj.get("extra") or {}),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CandidateFile:
    path: str
    content: str

    def digest(self) -> str:
        return digest_of_text(self.content)


@dataclass(frozen=True)
class CandidatePatch:
    """Proposer output.  ``files`` must really participate in execution; the
    materializer may never replace them with a fixed template."""

    parent_bank_digest: str
    operation: str  # add | edit | compose | prune | replace
    hypothesis: str
    files: tuple[CandidateFile, ...] = ()
    delete_paths: tuple[str, ...] = ()
    manifest_patch: Mapping[str, JSONValue] = field(default_factory=dict)
    alignment_hints: Mapping[str, JSONValue] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    expected_cost_change: Mapping[str, JSONValue] = field(default_factory=dict)
    self_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in ("add", "edit", "compose", "prune", "replace"):
            raise ValueError(f"unknown operation {self.operation!r}")

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "schema": CANDIDATE_PATCH_SCHEMA,
            "parent_bank_digest": self.parent_bank_digest,
            "operation": self.operation,
            "hypothesis": self.hypothesis,
            "files": [{"path": f.path, "content": f.content} for f in self.files],
            "delete_paths": list(self.delete_paths),
            "manifest_patch": dict(self.manifest_patch),
            "alignment_hints": dict(self.alignment_hints),
            "evidence_refs": list(self.evidence_refs),
            "expected_cost_change": dict(self.expected_cost_change),
            "self_checks": list(self.self_checks),
        }

    @staticmethod
    def from_json(obj: Mapping[str, JSONValue]) -> "CandidatePatch":
        return CandidatePatch(
            parent_bank_digest=str(obj["parent_bank_digest"]),
            operation=str(obj["operation"]),
            hypothesis=str(obj.get("hypothesis", "")),
            files=tuple(CandidateFile(path=str(f["path"]), content=str(f["content"])) for f in obj.get("files", [])),  # type: ignore[arg-type]
            delete_paths=tuple(str(p) for p in obj.get("delete_paths", [])),  # type: ignore[arg-type]
            manifest_patch=dict(obj.get("manifest_patch") or {}),  # type: ignore[arg-type]
            alignment_hints=dict(obj.get("alignment_hints") or {}),  # type: ignore[arg-type]
            evidence_refs=tuple(str(r) for r in obj.get("evidence_refs", [])),  # type: ignore[arg-type]
            expected_cost_change=dict(obj.get("expected_cost_change") or {}),  # type: ignore[arg-type]
            self_checks=tuple(str(s) for s in obj.get("self_checks", [])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CandidateRecord:
    """Materialized candidate: source of truth for archive and audit."""

    candidate_id: str  # digest of resolved source tree + manifest
    parent_bank_digest: str
    bank_digest: str
    operation: str
    hypothesis: str
    validation: Mapping[str, JSONValue] = field(default_factory=dict)
    model_output_ref: str | None = None
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "parent_bank_digest": self.parent_bank_digest,
            "bank_digest": self.bank_digest,
            "operation": self.operation,
            "hypothesis": self.hypothesis,
            "validation": dict(self.validation),
            "model_output_ref": self.model_output_ref,
            "provenance": dict(self.provenance),
        }


def empty_bank_manifest(bank_id: str = "empty") -> BankManifest:
    return BankManifest(bank_id=bank_id)


__all__ = [
    "MANIFEST_SCHEMA",
    "CANDIDATE_PATCH_SCHEMA",
    "HOOK_NAMES",
    "COMPONENT_KINDS",
    "ComponentManifest",
    "BankManifest",
    "CandidateFile",
    "CandidatePatch",
    "CandidateRecord",
    "empty_bank_manifest",
]
