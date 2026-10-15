"""Static checks before a candidate is allowed to spend evaluation budget."""

from __future__ import annotations

import json
from pathlib import Path

from trim.evolution.operators import Proposal, check_edit_budget
from trim.harness_bank.loader import BankLoadError, load_bank
from trim.harness_bank.sandbox import scan_component_ast

_BANNED_TEXT = ("gold_doc_ids", "gold_ids", "answer_key", "gold_answer", "task_answer_table")


def validate_proposal(
    proposal: Proposal,
    *,
    max_edits: int = 1,
    max_active_components: int = 6,
    forbidden_ids: set[str] | None = None,
) -> str | None:
    edit_error = check_edit_budget(proposal, max_edits=max_edits)
    if edit_error:
        return edit_error
    root = Path(proposal.bank_dir)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "manifest.json" or rel.startswith(("components/", "guidance/", "recipes/")):
            continue
        if rel in {"adaptation.json", "provenance.json", "hypothesis.json"}:
            continue
        return f"proposal writes outside the editable surface: {rel}"
    try:
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*") if path.is_file())
    except OSError as exc:
        return str(exc)
    lowered = text.lower()
    for token in _BANNED_TEXT:
        if token in lowered:
            return f"proposal contains task-specific answer material {token}"
    for qid in forbidden_ids or set():
        if qid and qid in text:
            return f"proposal contains query id {qid}"
    for path in (root / "components").glob("*.py") if (root / "components").is_dir() else []:
        try:
            scan_component_ast(path.read_text(encoding="utf-8"), label=path.name)
        except BankLoadError as exc:
            return str(exc)
    try:
        bank = load_bank(root)
    except BankLoadError as exc:
        return str(exc)
    if len(bank.active_components) > max_active_components:
        return f"active components {len(bank.active_components)} exceed {max_active_components}"
    hypothesis = proposal.hypothesis
    for key in ("failure_type", "trajectory_id", "mechanism", "expected_fix", "rollback"):
        if key not in hypothesis:
            return f"hypothesis.json missing {key}"
    if "source_component" not in proposal.adaptation:
        return "adaptation.json missing source_component"
    examples = proposal.examples
    for key in ("positive", "negative", "missing_evidence", "invalid_args"):
        if key not in examples:
            return f"structural examples missing {key}"
    return None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
