"""Bank manifest types. Dynamic component ids stay out of the static taxonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


BANK_SCHEMA = "trim_bank_v1"
COMPONENT_KINDS = (
    "view",
    "auto_state",
    "native_recipe",
    "privileged_tool",
    "persistent_system",
)
DISTILL_ROUTES = (
    "native_anchor",
    "explicit_map",
    "skip_to_anchor",
    "native_recipe",
    "persistent_uninternalized",
)
HOOK_PHASES = ("before_model", "after_tool", "before_finish")
PROJECTOR_VERSION = "trim_bank_projector_v1"


class BankComponent(Protocol):
    component_id: str

    def on_reset(self, view: Any, private_state: dict[str, Any]) -> dict[str, Any]: ...

    def before_model(self, view: Any, private_state: dict[str, Any]) -> dict[str, Any]: ...

    def after_tool(
        self, view: Any, tool_result: dict[str, Any], private_state: dict[str, Any]
    ) -> dict[str, Any]: ...

    def before_finish(
        self, view: Any, proposed_action: dict[str, Any], private_state: dict[str, Any]
    ) -> dict[str, Any]: ...

    def export_private_state(self) -> dict[str, Any]: ...

    def import_private_state(self, state: dict[str, Any]) -> None: ...


class BankRuntime(Protocol):
    def render(self, visible_state: dict[str, Any], *, role: str) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any], *, role: str) -> dict[str, Any]: ...

    def export_state(self) -> dict[str, Any]: ...

    def fork_for_teacher(self, student_snapshot: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    kind: str
    factory: str
    depends_on: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    phase: str = "before_model"
    priority: int = 100
    state_scope: str = "episode"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    distill_route: str = "native_anchor"
    target_tools: tuple[str, ...] = ()
    guidance: str = ""
    recipe: str = ""
    max_extra_llm_calls_per_episode: int = 0
    max_extra_tool_calls_per_episode: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "depends_on",
            "conflicts_with",
            "reads",
            "writes",
            "target_tools",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class BankSpec:
    schema_version: str
    bank_id: str
    legacy_flags: dict[str, bool]
    active_components: tuple[str, ...]
    components: tuple[ComponentSpec, ...]
    hook_order: tuple[str, ...]
    caps: dict[str, Any]
    base_contract_sha256: str = ""
    policy_digest_at_search: str = ""
    root: str = ""
    bank_sha256: str = ""

    def component(self, component_id: str) -> ComponentSpec:
        for spec in self.components:
            if spec.id == component_id:
                return spec
        raise KeyError(component_id)


@dataclass(frozen=True)
class ResolvedHarness:
    """Explicit zero/all resolution. Do not infer this from an empty static id list."""

    mode: str
    preset: str
    bank_sha256: str
    active_components: tuple[str, ...]
    student_mask: dict[str, bool]
    teacher_mask: dict[str, bool]
    eval_mask: dict[str, bool]
    student_role: str
    teacher_role: str
    hooks_on_student: bool
    hooks_on_teacher: bool

    @property
    def empty_bank(self) -> bool:
        return not self.active_components


@dataclass(frozen=True)
class BankBranchTrace:
    branch_id: str
    bank_sha256: str
    source_policy_digest: str
    student_pre_state_hash: str
    events: tuple[Any, ...]
    teacher_prompt_token_ids: tuple[int, ...]
    teacher_state_hash: str
    component_effect_ids: tuple[str, ...]
    continuous: bool
    information_provenance: tuple[Any, ...]
    cost: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = list(self.events)
        payload["teacher_prompt_token_ids"] = list(self.teacher_prompt_token_ids)
        payload["component_effect_ids"] = list(self.component_effect_ids)
        payload["information_provenance"] = list(self.information_provenance)
        return payload
