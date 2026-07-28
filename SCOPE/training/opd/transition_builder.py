"""Build OPD transitions: vLLM rollout + shadow privileged context."""

from __future__ import annotations

from training.opd._policy_backend import OPDTransition, RolloutBackend
from training.opd.rollout_worker import QueryRecord
from training.opd.shadow_harness import ShadowHarness
from training.opd.token_alignment import is_critical_action_token


def _search_agent_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are a search agent. Given the question below, "
                "propose the next tool call as compact JSON.\n\n"
                f"Question: {query}"
            ),
        }
    ]


def build_transitions_from_rollout(
    rollout: RolloutBackend,
    records: list[QueryRecord],
    shadow: ShadowHarness,
    *,
    target_module: str = "verification",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    tokenizer=None,
) -> list[OPDTransition]:
    transitions: list[OPDTransition] = []
    for turn_id, record in enumerate(records):
        result = rollout.rollout_chat(
            _search_agent_messages(record.query),
            {"max_new_tokens": max_new_tokens, "temperature": temperature},
        )
        if not result.action_token_ids:
            continue

        shadow_result = shadow.run_verification_shadow(
            turn_id=turn_id,
            claim=record.query[:160],
            doc_ids=["doc_stub"],
            doc_texts={"doc_stub": record.query},
            student_wm=None,
        )
        privileged_text = (
            shadow_result.artifacts[0].compact_text if shadow_result.artifacts else ""
        )
        teacher_suffix = ""
        if privileged_text:
            teacher_suffix = (
                "\n\n=== Privileged Module Context ===\n" + privileged_text
            )
        teacher_prefix = result.prompt_token_ids
        if tokenizer is not None and teacher_suffix:
            teacher_prefix = result.prompt_token_ids + tokenizer.encode(
                teacher_suffix, add_special_tokens=False
            )
        elif teacher_suffix and hasattr(rollout, "tokenizer"):
            teacher_prefix = result.prompt_token_ids + rollout.tokenizer.encode(
                teacher_suffix, add_special_tokens=False
            )

        action_mask = [True] * len(result.action_token_ids)
        if tokenizer is not None and result.text:
            action_mask = [
                is_critical_action_token(result.text)
            ] * len(result.action_token_ids)

        transitions.append(
            OPDTransition(
                episode_id=f"rollout_{record.query_id}",
                query_id=record.query_id,
                turn_id=turn_id,
                student_input_ids=result.prompt_token_ids,
                action_ids=result.action_token_ids,
                action_mask=action_mask,
                teacher_input_ids=teacher_prefix,
                privileged_module_id=target_module,
                reward=0.0,
                success=True,
                metadata={
                    "rollout_backend": result.metadata.get("backend", "unknown"),
                    "shadow_mode": shadow_result.mode,
                    "action_text": result.text[:200],
                },
            )
        )
    return transitions
