#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAPE_ROOT = Path(os.environ.get("SCAPE_ROOT", "/mnt/songzijun/Capability_Evolution/SCAPE"))
OUT = ROOT / "outputs" / "scape_easyopd" / "framework" / "HARNESS1_RUNTIME_INVENTORY.md"
HARNESS = SCAPE_ROOT / "external" / "harness-1"
EASY = ROOT


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN ({exc})"


def rel(path: Path, line: int) -> str:
    return f"{path}:{line}"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Harness-1 checkout", rel(HARNESS / "pyproject.toml", 1), "Pinned upstream runtime checkout, source of truth for harness package."),
        ("Main runtime entry", rel(HARNESS / "harness/agent.py", 79), "InferenceContext and AgentInferenceModel drive closed-loop agent actions."),
        ("Search agent loop", rel(HARNESS / "harness/agent.py", 156), "DeduplicatingPruningSearchAgent is used by EasyOPD live loop for Harness tool execution."),
        ("Environment reset state", rel(HARNESS / "harness/ultra_core.py", 679), "WorkingMemory(query, normalize_ids=True) is the reset object used by Harness1Bridge."),
        ("Environment step state mutation", rel(HARNESS / "harness/ultra_core.py", 725), "WorkingMemory.add_to_pool mutates visible document pool and v8d graph/dedup state."),
        ("Curate state mutation", rel(HARNESS / "harness/ultra_core.py", 780), "WorkingMemory.curate mutates curated_ids and importance/subtractive semantics."),
        ("Student-visible prefix", rel(HARNESS / "harness/ultra_core.py", 924), "WorkingMemory.to_text renders curated set, document pool, search history and visible markers."),
        ("Harmony context build", rel(HARNESS / "harness/ultra_core.py", 1159), "build_context is the GPT-OSS Harmony conversation construction path."),
        ("System prompt", rel(HARNESS / "harness/ultra_core.py", 403), "get_system_prompt injects enabled tool/component instructions."),
        ("Tool schema definitions", rel(HARNESS / "harness/tools.py", 126), "ToolSchema converts parser-visible schemas to provider formats."),
        ("Search schema", rel(HARNESS / "harness/tools.py", 199), "SEARCH_CORPUS_SCHEMA defines search_corpus."),
        ("Read schema", rel(HARNESS / "harness/tools.py", 215), "READ_DOCUMENT_SCHEMA defines read_document."),
        ("Ultra curate schema", rel(HARNESS / "harness/ultra_core.py", 266), "CURATE_SCHEMA changes with importance tagging flag."),
        ("Ultra verify schema", rel(HARNESS / "harness/ultra_core.py", 275), "VERIFY_SCHEMA is appended only when V8D_VERIFY_TOOL is enabled."),
        ("ToolSet dispatch", rel(HARNESS / "harness/tools.py", 872), "ToolSet.from_config constructs concrete search/grep/read/prune tools."),
        ("Component flags", rel(HARNESS / "harness/ultra_core.py", 170), "All ten V8D_* flags are read from environment at module import."),
        ("Auto-populate hook", rel(HARNESS / "harness/ultra_core.py", 1935), "auto_populate_from_first_search applies the first-search side effect."),
        ("Content dedup hook", rel(HARNESS / "harness/ultra_core.py", 526), "ContentDedupTracker is enabled by V8D_CONTENT_DEDUP."),
        ("Evidence graph hook", rel(HARNESS / "harness/ultra_core.py", 703), "WorkingMemory.evidence_graph is created when V8D_EVIDENCE_GRAPH is enabled."),
        ("Token budget marker", rel(HARNESS / "harness/ultra_core.py", 1717), "format_token_budget_marker/append_token_marker implement marker semantics."),
        ("Verify execution", rel(HARNESS / "harness/ultra_core.py", 1870), "exec_verify_claim performs claim checking against document text."),
        ("Reward source", rel(HARNESS / "harness/ultra_core.py", 1502), "compute_reward derives terminal reward and diagnostics."),
        ("EasyOPD bridge", rel(EASY / "easyopd/methods/scape_component_opd/harness1_bridge.py", 159), "Harness1Bridge reset/step/snapshot/event API for EasyOPD."),
        ("Bridge component environment", rel(EASY / "easyopd/methods/scape_component_opd/harness1_bridge.py", 60), "harness_component_env toggles exactly one V8D flag and reloads ultra_core."),
        ("Bridge teacher same-state fork", rel(EASY / "easyopd/methods/scape_component_opd/harness1_bridge.py", 291), "build_teacher_view_from_same_state replays one action on copied WorkingMemory with target ON."),
        ("Bridge event row", rel(EASY / "easyopd/methods/scape_component_opd/harness1_bridge.py", 324), "event_row_from_step emits formal EVENT_ACTIVE_STATES rows."),
        ("Collector selection", rel(EASY / "easyopd/methods/scape_component_opd/event_collection.py", 110), "collect_event_states validates real_harness1 rows, dedups state_uid, freezes 5K."),
        ("Collector CLI", rel(EASY / "scripts/scape_component_opd.py", 98), "cmd_collect routes formal collection through collect_event_states with require_real_harness=True."),
    ]
    component_flags = {
        "subtractive_curation": "V8D_SUBTRACTIVE_CURATION",
        "importance_tagging": "V8D_IMPORTANCE_TAGGING",
        "auto_populate_first_search": "V8D_AUTO_POPULATE_FIRST_SEARCH",
        "evidence_graph": "V8D_EVIDENCE_GRAPH",
        "sentence_compress": "V8D_SENTENCE_COMPRESS",
        "chunk_neighbors": "V8D_CHUNK_NEIGHBORS",
        "content_dedup": "V8D_CONTENT_DEDUP",
        "verify_tool": "V8D_VERIFY_TOOL",
        "token_budget_marker": "V8D_TOKEN_BUDGET_MARKER",
        "adaptive_rerank_instruction": "V8D_ADAPTIVE_RERANK_INSTRUCTION",
    }
    lines = [
        "# HARNESS1_RUNTIME_INVENTORY",
        "",
        f"- SCAPE commit: `{git_head(SCAPE_ROOT)}`",
        f"- Harness-1 commit: `{git_head(HARNESS)}`",
        f"- SCAPE root: `{SCAPE_ROOT}`",
        f"- EasyOPD root: `{ROOT}`",
        "",
        "## Runtime Mapping",
        "",
        "| Item | Path | Mapping |",
        "|---|---|---|",
    ]
    for item, path, mapping in rows:
        lines.append(f"| {item} | `{path}` | {mapping} |")
    lines.extend(["", "## Ten Component Enable/Disable Mapping", "", "| Component | Harness flag | Bridge hook |", "|---|---|---|"])
    for component, flag in component_flags.items():
        lines.append(f"| `{component}` | `{flag}` | `harness_component_env(component, enabled=...)` in `{EASY / 'easyopd/methods/scape_component_opd/harness1_bridge.py'}:60` |")
    lines.extend([
        "",
        "## Required Phase-U Questions",
        "",
        "1. Runtime entry: `harness.agent.DeduplicatingPruningSearchAgent` and `harness.ultra_core.WorkingMemory`.",
        "2. Reset/step/tool dispatch: `Harness1Bridge.reset`, `Harness1Bridge.step`, `ToolSet.from_config`, and WorkingMemory mutation methods listed above.",
        "3. Student-visible state/prefix: `WorkingMemory.to_text` plus `Harness1Bridge.snapshot_student_visible_state`.",
        "4. Tool schema/parser: `harness.tools.ToolSchema`, base schemas, and ultra-core curate/verify/end_search schemas.",
        "5. Component hooks: all V8D flags mapped above, toggled by bridge environment reload.",
        "6. Event pre/post runtime state: `Harness1Bridge.step` captures `pre_state`, same-state `teacher_view`, `post_state`, and `event`.",
        "7. Side effects/privileged context: generated in Harness-1 WorkingMemory/helpers, not copied into EasyOPD semantics.",
        "8. Trajectory/reward/terminal state: Harness-1 trajectory in `harness.agent`; reward in `compute_reward`; EasyOPD formal collector records terminal reward when present and otherwise keeps `N/A`/null.",
    ])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
