from __future__ import annotations

from .action_projection import project_curated_delta, validate_projected_action
from .component_spec import ComponentSpec


def _curated_delta_projection(row):
    return project_curated_delta(
        curated_ids_pre=list(row.get("curated_ids_pre") or []),
        curated_ids_post=list(row.get("curated_ids_post") or []),
        visible_doc_ids=list(row.get("visible_doc_ids") or []),
    )


def _projection_visibility(row) -> bool:
    action, audit = _curated_delta_projection(row)
    return action is not None and bool(audit.get("projection_valid"))


COMPONENT_SPECS: dict[str, ComponentSpec] = {
    "verify_tool": ComponentSpec(
        name="verify_tool",
        effect_type="ACTION_SPACE_CHANGE",
        realizability="NON_REALIZABLE",
        default_loss_mode="none",
        train_refusal_code="NON_REALIZABLE_ACTION_SPACE_MISMATCH",
        mechanism_metrics=[],
    ),
    "importance_tagging": ComponentSpec(
        name="importance_tagging",
        effect_type="ARGUMENT_PRIVILEGE",
        realizability="PARTIAL",
        projection_builder=_curated_delta_projection,
        visibility_validator=_projection_visibility,
        action_schema_validator=lambda row: bool((row.get("projected_action") or {}).get("arguments")),
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["support_rate", "valid_add_rate", "valid_remove_rate"],
    ),
    "subtractive_curation": ComponentSpec(
        name="subtractive_curation",
        effect_type="AUTOMATIC_SIDE_EFFECT",
        realizability="PROJECTABLE",
        projection_builder=_curated_delta_projection,
        visibility_validator=_projection_visibility,
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["valid_remove_rate", "irrelevant_removed_rate", "curated_churn"],
    ),
    "importance_tagging_plus_subtractive_curation": ComponentSpec(
        name="importance_tagging_plus_subtractive_curation",
        effect_type="ARGUMENT_PRIVILEGE",
        realizability="PROJECTABLE",
        projection_builder=_curated_delta_projection,
        visibility_validator=_projection_visibility,
        default_loss_mode="projected_action_ce",
        mechanism_metrics=[
            "valid_add_rate",
            "valid_remove_rate",
            "irrelevant_removed_rate",
            "curated_churn",
            "support_rate",
        ],
    ),
    "auto_populate_first_search": ComponentSpec(
        name="auto_populate_first_search",
        effect_type="AUTOMATIC_SIDE_EFFECT",
        realizability="PROJECTABLE",
        projection_builder=_curated_delta_projection,
        visibility_validator=_projection_visibility,
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["search_to_curate_delay", "immediate_curate_rate", "relevant_added_rate"],
    ),
    "content_dedup": ComponentSpec(
        name="content_dedup",
        effect_type="AUTOMATIC_POOL_FILTER",
        realizability="PARTIAL",
        default_loss_mode="none",
        train_refusal_code="STOP_NO_ACTIVE_EVENT_SUPPORT",
        mechanism_metrics=["duplicate_trigger_rate", "duplicate_read_rate", "duplicate_curate_rate"],
    ),
    "chunk_neighbors": ComponentSpec(
        name="chunk_neighbors",
        effect_type="EXTERNAL_INFORMATION_AUGMENTATION",
        realizability="PARTIAL",
        default_loss_mode="none",
        train_refusal_code="KEEP_RUNTIME",
        mechanism_metrics=["projectable_action_sequence_rate"],
    ),
    "evidence_graph": ComponentSpec(
        name="evidence_graph",
        effect_type="PRIVILEGED_CONTEXT",
        realizability="DIRECT",
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["bridge_entity_search_rate", "new_entity_discovery"],
    ),
    "sentence_compress": ComponentSpec(
        name="sentence_compress",
        effect_type="PRIVILEGED_CONTEXT",
        realizability="DIRECT",
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["compression_ratio", "same_state_rate"],
    ),
    "token_budget_marker": ComponentSpec(
        name="token_budget_marker",
        effect_type="PRIVILEGED_CONTEXT",
        realizability="PARTIAL",
        default_loss_mode="projected_action_ce",
        train_refusal_code="RUNTIME_ANCHOR_PREFERRED_IF_EXACT_ACCOUNTING_REQUIRED",
        mechanism_metrics=["termination_timing", "tool_calls", "late_step_waste"],
    ),
    "adaptive_rerank_instruction": ComponentSpec(
        name="adaptive_rerank_instruction",
        effect_type="PRIVILEGED_RETRIEVAL_CONTEXT",
        realizability="PARTIAL",
        default_loss_mode="projected_action_ce",
        mechanism_metrics=["retrieval_delta", "topK_overlap", "qrel_recall_delta"],
    ),
}


def get_component_spec(name: str) -> ComponentSpec:
    try:
        return COMPONENT_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown SCAPE component: {name}") from exc


def list_component_specs() -> list[ComponentSpec]:
    return [COMPONENT_SPECS[name] for name in sorted(COMPONENT_SPECS)]


def audit_component(name: str, *, event_support: int | None = None, student_has_tool: bool = False) -> dict[str, object]:
    spec = get_component_spec(name)
    can_train, reason = spec.can_train(student_has_tool=student_has_tool)
    if name == "content_dedup" and event_support == 0:
        can_train, reason = False, "STOP_NO_ACTIVE_EVENT_SUPPORT"
    return {
        "name": spec.name,
        "effect_type": spec.effect_type,
        "realizability": spec.realizability,
        "default_loss_mode": spec.default_loss_mode,
        "mechanism_metrics": list(spec.mechanism_metrics),
        "can_train": can_train,
        "decision_code": reason,
    }


def validate_action_visibility(action, *, visible_doc_ids: list[str], curated_ids_pre: list[str]) -> dict[str, object]:
    return validate_projected_action(action, visible_doc_ids=visible_doc_ids, curated_ids_pre=curated_ids_pre)
