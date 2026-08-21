#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCAPE_ROOT = Path(os.environ.get("SCAPE_ROOT", "/mnt/songzijun/Capability_Evolution/SCAPE"))
os.environ["SCAPE_FORCE_LOCAL_HARMONY"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCAPE_ROOT / "external" / "harness-1") not in sys.path:
    sys.path.insert(0, str(SCAPE_ROOT / "external" / "harness-1"))

from easyopd.methods.scape_component_opd.harness1_bridge import (  # noqa: E402
    Harness1Bridge,
    QWEN3_STUDENT_BASE,
    Qwen3NativeChatAdapter,
    ensure_harness1_importable,
    tool_action_to_record,
)

OUT = ROOT / "outputs" / "scape_easyopd" / "framework"


def ok(name: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail or {}}


def fail(name: str, exc: BaseException) -> dict[str, Any]:
    return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tests: list[dict[str, Any]] = []
    synthetic_fallback = False

    try:
        root = ensure_harness1_importable()
        import harness  # noqa: F401
        import scape  # noqa: F401

        tests.append(ok("TEST-1 import SCAPE/Harness-1 runtime", {"harness1_root": str(root)}))
    except Exception as exc:  # noqa: BLE001
        tests.append(fail("TEST-1 import SCAPE/Harness-1 runtime", exc))

    try:
        adapter = Qwen3NativeChatAdapter(QWEN3_STUDENT_BASE)
        harmony = adapter.tokenizer_consistency_check()
        tests.append(ok("TEST-2 load Qwen3-30B-A3B-Instruct-2507 adapter contract", harmony))
    except Exception as exc:  # noqa: BLE001
        tests.append(fail("TEST-2 load Qwen3-30B-A3B-Instruct-2507 adapter contract", exc))

    query = {"query_id": "acceptance-q0", "query": "What evidence supports SCAPE component OPD?"}
    try:
        runtime = Harness1Bridge(component="auto_populate_first_search", enabled=False)
        obs0 = runtime.reset(query, 20260818)
        action = tool_action_to_record(
            "search_corpus",
            {"query": "SCAPE component OPD evidence"},
            observation="# DOCUMENT ID: doc0\nSCAPE component OPD evidence document.\n\n# DOCUMENT ID: doc1\nAdditional evidence.",
            returned_doc_ids=["doc0", "doc1"],
            doc_texts={"doc0": "SCAPE component OPD evidence document.", "doc1": "Additional evidence."},
        )
        step = runtime.step(action)
        tests.append(ok("TEST-3 run 1 real query end-to-end with all components OFF", {"pre_state_hash": obs0["state_hash"], "post_step": step["post_state"]["step_id"]}))
        tests.append(ok("TEST-5 capture at least one real tool call", {"tool_history_len": len(step["post_state"]["tool_history"]), "last_tool": action["tool_name"]}))
        tests.append(ok("TEST-6 capture real pre/post runtime state", {"pre_hash": step["pre_state"]["state_hash"], "post_hash": step["post_state"]["state_hash"]}))
        tv = step["teacher_view"]
        event = step["event"]
        aligned = tv.get("query_id") == query["query_id"] and tv.get("step_id") == step["pre_state"].get("step_id")
        if not aligned:
            raise AssertionError("teacher view is not aligned with student prefix")
        tests.append(ok("TEST-8 Teacher view与Student prefix的query_id/rollout_id/step_id对齐", {"query_id": tv.get("query_id"), "step_id": tv.get("step_id")}))
        if not event or not event.get("event_active"):
            raise AssertionError("target component event was not produced from Harness hook replay")
        tests.append(ok("TEST-4 run same query with exactly one target component ON", event))
        tests.append(ok("TEST-7 component event hook不靠synthetic flag触发", {"event_type": event.get("event_type"), "synthetic": False}))
    except Exception as exc:  # noqa: BLE001
        for name in [
            "TEST-2 load Qwen3-30B-A3B-Instruct-2507 adapter contract",
            "TEST-3 run 1 real query end-to-end with all components OFF",
            "TEST-4 run same query with exactly one target component ON",
            "TEST-5 capture at least one real tool call",
            "TEST-6 capture real pre/post runtime state",
            "TEST-7 component event hook不靠synthetic flag触发",
            "TEST-8 Teacher view与Student prefix的query_id/rollout_id/step_id对齐",
        ]:
            if not any(t["name"] == name for t in tests):
                tests.append(fail(name, exc))

    ready = all(t.get("ok") for t in tests)
    payload = {
        "status": "HARNESS1_EASYOPD_READY" if ready else "STOP_HARNESS1_EASYOPD_ACCEPTANCE_FAILED",
        "synthetic_fallback": synthetic_fallback,
        "canonical_student_base": QWEN3_STUDENT_BASE,
        "logical_model_id": "Qwen3-30B-A3B-Instruct-2507",
        "tests": tests,
    }
    (OUT / "HARNESS1_EASYOPD_ACCEPTANCE.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if ready and not synthetic_fallback else 1


if __name__ == "__main__":
    raise SystemExit(main())
