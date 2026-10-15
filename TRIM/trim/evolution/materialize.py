"""Write one candidate bank from a structured self-evolution proposal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from trim.evolution.operators import Proposal
from trim.harness_bank.loader import load_bank

_STDLIB = Path(__file__).resolve().parents[1] / "harness_bank" / "stdlib" / "query_repair.py"
_REQUIRED = ("failure_type", "trajectory_id", "mechanism", "expected_fix", "rollback", "guidance")


def parse_proposal_text(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("proposer output has no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("proposer JSON must be an object")
    missing = [key for key in _REQUIRED if not str(payload.get(key) or "").strip()]
    if missing:
        raise ValueError(f"proposer JSON missing {missing}")
    failure = str(payload["failure_type"])
    if failure not in {
        "bad_query",
        "unread_evidence",
        "redundant_search",
        "bad_selection",
        "early_stop",
        "context_overflow",
        "format_or_tool_error",
    }:
        raise ValueError(f"unsupported failure_type {failure}")
    return payload


def write_candidate_bank(root: Path, payload: Mapping[str, object], *, proposal_type: str = "add") -> Proposal:
    if not _STDLIB.is_file():
        raise FileNotFoundError(_STDLIB)
    root.mkdir(parents=True, exist_ok=True)
    component_id = "evo.query_repair"
    (root / "components").mkdir(exist_ok=True)
    (root / "guidance").mkdir(exist_ok=True)
    (root / "recipes").mkdir(exist_ok=True)
    shutil.copyfile(_STDLIB, root / "components" / "query_repair.py")
    guidance = str(payload["guidance"]).strip()
    (root / "guidance" / "query_repair.md").write_text(guidance + "\n", encoding="utf-8")
    recipe = {
        "component_id": component_id,
        "route": "native_anchor",
        "target_tools": ["search_corpus"],
        "trigger": "after_search",
        "guidance": guidance,
    }
    (root / "recipes" / "query_repair.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    hypothesis = {key: payload[key] for key in ("failure_type", "trajectory_id", "mechanism", "expected_fix", "rollback")}
    (root / "hypothesis.json").write_text(json.dumps(hypothesis, indent=2) + "\n", encoding="utf-8")
    adaptation = {
        "source_component": component_id,
        "route": "native_anchor",
        "target_action_schema": "search_corpus",
        "information_source": "visible_query",
        "skip_if_unmapped": True,
    }
    (root / "adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "trim_bank_v1",
        "bank_id": root.name,
        "legacy_flags": {},
        "active_components": [component_id],
        "components": [
            {
                "id": component_id,
                "kind": "view",
                "factory": "components/query_repair.py:create",
                "depends_on": [],
                "conflicts_with": [],
                "phase": "before_model",
                "priority": 20,
                "reads": ["query", "visible_search_history"],
                "writes": ["teacher_guidance"],
                "distill_route": "native_anchor",
                "target_tools": ["search_corpus"],
                "guidance": "guidance/query_repair.md",
                "recipe": "recipes/query_repair.json",
                "max_extra_llm_calls_per_episode": 0,
                "max_extra_tool_calls_per_episode": 0,
            }
        ],
        "hook_order": [component_id],
        "caps": {"active_components": 6, "added_prompt_tokens": 1536},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    loaded = load_bank(root)
    return Proposal(
        proposal_id=loaded.bank_sha256[:12],
        proposal_type=proposal_type if proposal_type in {"add", "edit", "prune"} else "edit",
        bank_dir=str(root),
        content_hash=loaded.bank_sha256,
        hypothesis=dict(hypothesis),
        adaptation=adaptation,
        examples={
            "positive": {"history": 1, "expects_guidance": True},
            "negative": {"history": 0, "expects_guidance": False},
            "missing_evidence": {"history": 1, "query": ""},
            "invalid_args": {"tool": "not_a_tool"},
        },
        expected_cost={"extra_llm_calls": 0},
        touched_components=(component_id,),
        edits=1,
    )
