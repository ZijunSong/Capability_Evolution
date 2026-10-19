"""Prompt/action encoding and the no-pack CE collator."""

from __future__ import annotations

from typing import Any, Sequence

from trim.training.sft_ablation.contracts import SFTBatchRow


def load_bound_encoding(model_path: str, *, max_model_len: int, max_new_tokens: int) -> Any:
    """Bind the real tokenizer. Never falls back to byte-level ``default_encode``."""
    from trim.eval.model_tokenizer import load_model_encoding

    if not str(model_path or "").strip():
        raise ValueError("SFT encoding requires a model path")
    return load_model_encoding(
        model_path,
        max_model_len=int(max_model_len),
        max_new_tokens=int(max_new_tokens),
    )


def assert_same_tokenizer(student_digest: str, teacher_digest: str) -> None:
    if not student_digest or not teacher_digest:
        raise ValueError("missing tokenizer digest")
    if student_digest != teacher_digest:
        raise ValueError(
            "heterogeneous teacher/student tokenizers are not supported in v1; "
            "re-render structured messages with the student tokenizer as an explicit extension"
        )


def encode_action(action: dict[str, Any], *, enc: Any, target_text: str = "") -> Any:
    from trim.training.action_codec import render_action
    from trim.training.action_encoding import SupervisedActionEncoding, encode_supervised_action

    try:
        return encode_supervised_action(
            target_action=action,
            target_text=target_text,
            encode=enc.encode,
            model_enc=enc,
        )
    except ValueError as exc:
        # The existing action regex stops at the first closing brace, so a
        # nested importance object cannot round-trip through parse_action.
        # Keep the canonical text, including importance, as the supervised tokens.
        if not any(_nested(item) for item in (action.get("arguments") or {}).values()):
            raise
        text = target_text or render_action(action)
        ids = [int(token) for token in enc.encode(text)]
        if not ids:
            raise ValueError("empty nested action encoding") from exc
        return SupervisedActionEncoding(
            target_ids=ids,
            target_text=text,
            canonical_action=dict(action),
            supervision_mask=[True] * len(ids),
            termination_kind="nested_json",
        )


def _nested(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if isinstance(value, list):
        return any(_nested(item) for item in value)
    return False


def require_visible_action(
    action: dict[str, Any],
    *,
    prompt_ids: Sequence[int],
    enc: Any,
    metadata: dict[str, Any] | None,
    snapshot: Any = None,
) -> list[str]:
    """B/C visibility. Unknown visibility is a skip, never a pass."""
    from trim.training.action_encoding import assert_action_visible, visible_doc_ids_for_decision

    visible = visible_doc_ids_for_decision(
        snapshot=snapshot,
        prompt_ids=prompt_ids,
        enc=enc,
        metadata=metadata,
    )
    if visible is None:
        raise ValueError("visibility_unknown")
    assert_action_visible(action, visible)
    return list(visible)


def fits_budget(prompt_ids: Sequence[int], response_ids: Sequence[int], *, max_model_len: int, max_new_tokens: int) -> None:
    if len(response_ids) > int(max_new_tokens):
        raise ValueError("action exceeds reserved max_new_tokens")
    if len(prompt_ids) + len(response_ids) > int(max_model_len):
        raise ValueError("sequence exceeds sampled context budget")


def collate_ce_rows(rows: Sequence[SFTBatchRow], *, pad_id: int, max_length: int) -> dict[str, list[list[int | float]]]:
    if not rows:
        raise ValueError("empty CE microbatch")
    for row in rows:
        row.validate(max_length)
    width = max(len(row.prompt_ids) + len(row.response_ids) for row in rows)
    batch: dict[str, list] = {key: [] for key in ("input_ids", "attention_mask", "labels", "token_weights")}
    for row in rows:
        prompt, response = row.prompt_ids, row.response_ids
        pad = width - len(prompt) - len(response)
        labels = [-100] * len(prompt) + [
            token if keep else -100 for token, keep in zip(response, row.response_mask)
        ]
        weights = [0.0] * len(prompt) + [
            float(row.weight) if keep else 0.0 for keep in row.response_mask
        ]
        batch["input_ids"].append(prompt + response + [int(pad_id)] * pad)
        batch["attention_mask"].append([1] * (len(prompt) + len(response)) + [0] * pad)
        batch["labels"].append(labels + [-100] * pad)
        batch["token_weights"].append(weights + [0.0] * pad)
    return batch
