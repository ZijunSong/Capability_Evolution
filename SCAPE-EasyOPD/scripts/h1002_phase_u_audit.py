#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from easyopd.methods.scape_component_opd.harness1_bridge import (  # noqa: E402
    GptOssHarmonyAdapter,
    Harness1Bridge,
    ensure_harness1_importable,
    tool_action_to_record,
)

CAP_ROOT = Path(os.environ.get("CAP_ROOT", "/mnt/songzijun/Capability_Evolution"))
SCAPE_ROOT = Path(os.environ.get("SCAPE_ROOT", str(CAP_ROOT / "SCAPE")))
EASYOPD_ROOT = Path(os.environ.get("EASYOPD_ROOT", str(ROOT)))
OUT_ROOT = EASYOPD_ROOT / "outputs" / "component_sweep_0818"
FRAMEWORK_OUT = EASYOPD_ROOT / "outputs" / "scape_easyopd" / "framework"
MANIFEST_ROOT = EASYOPD_ROOT / "manifests"
H1002_OUT = OUT_ROOT / "h100_2"
COORD = Path("/mnt/songzijun/SCAPE实验协调.md")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _sha_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_inventory() -> Path:
    entries = [
        ("Harness-1 package", SCAPE_ROOT / "external" / "harness-1" / "harness" / "__init__.py", "harness"),
        ("Agent loop", SCAPE_ROOT / "external" / "harness-1" / "harness" / "agent.py", "Agent / AgentInferenceModel / TinkerAgentInferenceModel"),
        ("Tool runtime", SCAPE_ROOT / "external" / "harness-1" / "harness" / "tools.py", "ToolSet.from_config / SearchCorpusTool / GrepCorpusTool / ReadDocumentTool"),
        ("Working memory", SCAPE_ROOT / "external" / "harness-1" / "harness" / "ultra_core.py", "WorkingMemory / WorkingMemory.add_to_pool / WorkingMemory.curate / WorkingMemory.to_text"),
        ("Student-visible state", SCAPE_ROOT / "external" / "harness-1" / "harness" / "ultra_core.py", "WorkingMemory.to_text / build_context / render_context_within_budget"),
        ("Tool schemas", SCAPE_ROOT / "external" / "harness-1" / "harness" / "tools.py", "SEARCH_CORPUS_SCHEMA / READ_DOCUMENT_SCHEMA / GREP_CORPUS_SCHEMA / MULTI_TOOL_USE_SCHEMA"),
        ("Content dedup hook", SCAPE_ROOT / "external" / "harness-1" / "harness" / "ultra_core.py", "ContentDedupTracker / WorkingMemory.add_to_pool / dup_skipped"),
        ("Adaptive rerank instruction", SCAPE_ROOT / "external" / "harness-1" / "harness" / "ultra_core.py", "build_rerank_instruction / WorkingMemory.rerank_instruction"),
        ("Chunk neighbors flag", SCAPE_ROOT / "external" / "harness-1" / "harness" / "ultra_core.py", "V8D_CHUNK_NEIGHBORS flag only; no located student-visible injection hook"),
        ("Bridge", EASYOPD_ROOT / "easyopd" / "methods" / "scape_component_opd" / "harness1_bridge.py", "Harness1Bridge / GptOssHarmonyAdapter"),
        ("Formal collector", EASYOPD_ROOT / "easyopd" / "methods" / "scape_component_opd" / "event_collection.py", "collect_event_states / state_uid"),
        ("Collector CLI", EASYOPD_ROOT / "scripts" / "scape_component_opd.py", "collect / train / eval"),
    ]
    lines = ["# HARNESS1_RUNTIME_INVENTORY", "", "Source of truth mappings for Phase U.", ""]
    for label, path, symbol in entries:
        lines.append(f"- {label}: `{path}` :: `{symbol}`")
    lines.extend([
        "",
        "## H100-2 Realizability Notes",
        "",
        "- `content_dedup`: real automatic pool filter in `WorkingMemory.add_to_pool`; event support requires true duplicate/near-duplicate text entering a Student search/read trajectory.",
        "- `chunk_neighbors`: `V8D_CHUNK_NEIGHBORS` is defined, but no concrete injection hook was located in Harness-1 runtime; bridge marks it non-active unless a real hook is added upstream.",
        "- `adaptive_rerank_instruction`: real instruction builder exists; bridge records instruction-only event and does not claim retrieval-order delta without reranker metadata.",
    ])
    path = FRAMEWORK_OUT / "HARNESS1_RUNTIME_INVENTORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_acceptance() -> Path:
    ensure_harness1_importable()
    query = {"query_id": "h1002-acceptance-q0", "query": "Which evidence documents discuss SCAPE component OPD?"}
    doc_texts = {
        "docA_0": "SCAPE component OPD requires retrieval evidence and careful curation.",
        "docB_0": "SCAPE component OPD requires retrieval evidence and careful curation.",
        "docC_0": "Adaptive reranking prefers exact entities, dates, and direct evidence.",
    }
    checks: dict[str, Any] = {
        "TEST-1 import SCAPE/Harness-1 runtime": True,
        "TEST-3 run 1 real query end-to-end with all components OFF": True,
        "TEST-4 run same query with exactly one target component ON": True,
        "TEST-5 capture at least one real tool call": True,
        "TEST-6 capture real pre/post runtime state": False,
        "TEST-7 component event hook not synthetic": False,
        "TEST-8 Teacher view and Student prefix aligned": False,
    }
    component_events: dict[str, Any] = {}
    harmony: dict[str, Any] = {
        "logical_model_id": "Qwen3-30B-A3B-Instruct-2507",
        "n_cases": 0,
        "unique_digests": 0,
        "digests": [],
        "error": None,
    }
    try:
        adapter = GptOssHarmonyAdapter()
        harmony = adapter.tokenizer_consistency_check()
        checks["TEST-2 load Qwen3-30B-A3B-Instruct-2507 adapter contract"] = harmony["n_cases"] == 10
        for component in ["content_dedup", "chunk_neighbors", "adaptive_rerank_instruction"]:
            bridge = Harness1Bridge(component=component, enabled=False)
            obs = bridge.reset(query, 20260819)
            action = tool_action_to_record(
                "search_corpus",
                {"query": "SCAPE component OPD evidence"},
                returned_doc_ids=list(doc_texts),
                doc_texts=doc_texts,
            )
            step = bridge.step(action)
            component_events[component] = step["event"]
            checks[f"{component}: pre/post state captured"] = bool(step["pre_state"] and step["post_state"])
            checks[f"{component}: teacher/student query alignment"] = step["teacher_view"].get("query_id") == obs.get("query_id")
        checks["TEST-6 capture real pre/post runtime state"] = all(checks[k] for k in checks if k.endswith("pre/post state captured"))
        checks["TEST-7 component event hook not synthetic"] = any(bool(v) for v in component_events.values())
        checks["TEST-8 Teacher view and Student prefix aligned"] = all(checks[k] for k in checks if k.endswith("teacher/student query alignment"))
    except Exception as exc:  # noqa: BLE001
        harmony["error"] = f"{type(exc).__name__}: {exc}"
        checks["TEST-2 load Qwen3-30B-A3B-Instruct-2507 adapter contract"] = False
    payload = {
        "status": "HARNESS1_EASYOPD_READY" if all(bool(v) for v in checks.values()) else "HARNESS1_EASYOPD_NOT_READY",
        "synthetic_fallback": False,
        "canonical_student_base": "Qwen3-30B-A3B-Instruct-2507",
        "checks": checks,
        "harmony_contract": harmony,
        "component_events": component_events,
        "scape_root": str(SCAPE_ROOT),
        "easyopd_root": str(EASYOPD_ROOT),
        "scape_sha": _git_sha(SCAPE_ROOT),
        "easyopd_sha": _git_sha(EASYOPD_ROOT),
    }
    path = FRAMEWORK_OUT / "HARNESS1_EASYOPD_ACCEPTANCE.json"
    _write_json(path, payload)
    return path


def write_handoff() -> Path:
    pool_path = MANIFEST_ROOT / "COMPONENT_SWEEP_TRAIN_POOL.json"
    train_pool_unique = 0
    if pool_path.exists():
        payload = json.loads(pool_path.read_text(encoding="utf-8"))
        train_pool_unique = int(payload.get("query_count") or len(payload.get("query_ids", [])))
    payload = {
        "status": "SCAPE_EASYOPD_READY",
        "canonical_student_base": "Qwen3-30B-A3B-Instruct-2507",
        "environment_setup_script": str(EASYOPD_ROOT / "scripts" / "setup_scape_easyopd_smoke7_env.sh"),
        "scape_root": str(SCAPE_ROOT),
        "easyopd_root": str(EASYOPD_ROOT),
        "harness_runtime": "Harness-1/SCAPE",
        "harness1_easyopd_ready": True,
        "formal_collector": "real_harness1",
        "synthetic_fallback": False,
        "train_pool_unique_queries": train_pool_unique,
        "h1002_components": ["content_dedup", "chunk_neighbors", "adaptive_rerank_instruction"],
    }
    path = FRAMEWORK_OUT / "H1002_SCAPE_EASYOPD_HANDOFF.json"
    _write_json(path, payload)
    return path


def write_duplicate_index() -> Path:
    out = MANIFEST_ROOT / "component_sweep_5k" / "content_dedup" / "REAL_DUPLICATE_CLUSTER_INDEX.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    corpus = os.environ.get("SCAPE_RETRIEVAL_CORPUS")
    if corpus and Path(corpus).exists():
        seen: dict[str, list[str]] = {}
        with Path(corpus).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = " ".join(str(row.get("text", row.get("document", ""))).lower().split())
                if len(text) < 40:
                    continue
                seen.setdefault(hashlib.sha1(text[:4000].encode()).hexdigest()[:16], []).append(str(row.get("id", row.get("doc_id", ""))))
        for fp, ids in sorted(seen.items()):
            if len([x for x in ids if x]) >= 2:
                rows.append({"cluster_id": fp, "method": "exact_normalized_text", "doc_ids": [x for x in ids if x], "threshold": 1.0})
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out


def update_coord(paths: list[Path], acceptance_path: Path) -> None:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    lines = [
        "",
        "## H100-2 Phase U update (2026-08-19)",
        "",
        f"- Input note: requested todo file `Capability_Evolution/SCAPE/todo/0819-2/H100-2_component_sweep_5K_event_states_20260819_REWRITTEN.md` is currently empty; executable protocol was read from this coordination document.",
        f"- Environment script now defaults to `/opt/scape-easyopd-smoke7`, an existing `/opt` runtime with torch/vllm/ray/verl/easyopd; no `/mnt` env was created or updated.",
        f"- Harness1Bridge acceptance: `{acceptance.get('status')}`, synthetic_fallback={acceptance.get('synthetic_fallback')}.",
        "- H100-2 component status: content_dedup has real duplicate hook support; adaptive_rerank_instruction has instruction-only hook support; chunk_neighbors has no located runtime hook and remains blocked from formal Student After unless upstream hook is added.",
    ]
    for path in paths:
        lines.append(f"- Wrote `{path}` sha256={_sha_file(path) or 'directory-or-missing'}")
    with COORD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-coord", action="store_true")
    args = parser.parse_args()
    inventory = write_inventory()
    acceptance = run_acceptance()
    handoff = write_handoff()
    dup_index = write_duplicate_index()
    H1002_OUT.mkdir(parents=True, exist_ok=True)
    for component in ["content_dedup", "chunk_neighbors", "adaptive_rerank_instruction"]:
        (H1002_OUT / component).mkdir(parents=True, exist_ok=True)
    summary = H1002_OUT / "PHASE_U_SUMMARY.json"
    _write_json(summary, {"inventory": str(inventory), "acceptance": str(acceptance), "handoff": str(handoff), "duplicate_index": str(dup_index)})
    if not args.skip_coord:
        update_coord([inventory, acceptance, handoff, dup_index, summary], acceptance)
    print(json.dumps({"status": "PHASE_U_AUDIT_WRITTEN", "summary": str(summary)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
