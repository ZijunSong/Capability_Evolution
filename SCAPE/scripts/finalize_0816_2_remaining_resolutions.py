#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "0816_2_remaining_resolution_final"
BASELINES = REPO / "outputs" / "btp_h100_4_baselines"
STRUCTURED = REPO / "outputs" / "h100_2_structured_privilege_formal_0816"
H1001 = Path("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade")


REQUIRED_HF_TOOL_OPD_FIELDS = {"prompt_reduced", "prompt_full", "response_text"}
REQUIRED_ROUTE_OPD_FIELDS = {"prompt_reduced", "P_teacher_route", "P_ref_route", "route_actions"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def module_available(py: str, mod: str) -> bool:
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{mod}') else 1)"
    return subprocess.run([py, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}


def schema_info(path: Path) -> dict[str, Any]:
    row = first_jsonl(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "keys": sorted(row.keys()) if isinstance(row, dict) else [],
        "missing_hf_tool_opd_fields": sorted(REQUIRED_HF_TOOL_OPD_FIELDS - set(row or {})),
        "missing_route_opd_fields": sorted(REQUIRED_ROUTE_OPD_FIELDS - set(row or {})),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bishop = "/opt/bishop-harness/bin/python"
    env = {
        "python": bishop,
        "torch": module_available(bishop, "torch"),
        "transformers": module_available(bishop, "transformers"),
        "peft": module_available(bishop, "peft"),
        "vllm": module_available(bishop, "vllm"),
        "pyserini": module_available(bishop, "pyserini"),
        "vllm_binary": shutil.which("vllm", path="/opt/bishop-harness/bin"),
    }

    matched_schema = schema_info(BASELINES / "matched_v2" / "matched_v2_pairs.jsonl")
    structured_schema = schema_info(STRUCTURED / "train_auto_paired.jsonl")
    matched_handoff = read_json(BASELINES / "MATCHED_TEXT_HANDOFF.json") or {}
    ophsd_handoff = read_json(BASELINES / "OPHSD_HANDOFF.json") or {}
    auto_handoff = read_json(H1001 / "H1001_AUTO_PAPERGRADE_HANDOFF.json") or {}

    checks = {
        "environment": env,
        "files": {
            "serve_harness1_vllm_local": file_info(REPO / "scripts" / "serve_harness1_vllm_local.sh"),
            "actual_lora_evaluator": file_info(REPO / "scripts" / "run_btp_auto_lora_real_closed_loop.py"),
            "hf_tool_opd": file_info(REPO / "scape" / "training" / "hf_tool_opd.py"),
            "train_route_opd_main_checkout": file_info(REPO / "scripts" / "train_route_opd.py"),
            "h1001_actual_handoff": file_info(H1001 / "H1001_AUTO_PAPERGRADE_HANDOFF.json"),
        },
        "schemas": {
            "matched_v2_pairs": matched_schema,
            "structured_train_auto_paired": structured_schema,
        },
        "handoffs": {
            "matched_text": matched_handoff,
            "ophsd": ophsd_handoff,
            "auto_actual_lora": auto_handoff,
        },
    }

    resolutions = [
        {
            "experiment": "Full Harness exact same-contract reference",
            "status": "resolved_blocked",
            "reason": "No completed exact same-contract Full Harness closed-loop artifact exists. A localhost vLLM launcher exists and /opt/bishop-harness has vllm, but there is no runner binding full Harness privileged runtime to the frozen H100-1 paper-grade query manifest/scorer. Starting a server alone would not produce the required row.",
            "evidence": "serve_harness1_vllm_local.sh exists; FULL_HARNESS_REAL_CLOSED_LOOP.csv remains NA; H1004 handoff marks full_harness_reference=missing_required_gap.",
            "next_action": "Implement a runner that uses the same H100-1 test256 manifest, BM25/index, max steps, termination and scorer, with full Harness runtime enabled; then run against the local vLLM server.",
        },
        {
            "experiment": "Matched Text actual LoRA baseline",
            "status": "resolved_blocked",
            "reason": "matched_v2_pairs.jsonl is an information-equivalence artifact only. It lacks prompt_reduced, prompt_full, response_text for hf_tool_opd.py and lacks prompt_reduced, P_teacher_route, P_ref_route, route_actions for the old route trainer contract. Main checkout also lacks scripts/train_route_opd.py.",
            "evidence": f"matched_v2 keys={matched_schema['keys']}; missing_hf_tool_opd={matched_schema['missing_hf_tool_opd_fields']}; missing_route_opd={matched_schema['missing_route_opd_fields']}",
            "next_action": "Recollect or reconstruct same-state actual-LoRA training rows with reduced prompt, matched-text full/teacher prompt, executable teacher response_text/tool call, and frozen query-disjoint train/valid/test manifests.",
        },
        {
            "experiment": "OPHSD actual LoRA baseline",
            "status": "resolved_blocked",
            "reason": "Existing OPHSD artifacts are route-head summaries and route-level closed-loop/handoff records, not PEFT/LoRA actual Student checkpoints or prompt/response training rows for hf_tool_opd.py.",
            "evidence": f"OPHSD handoff status={ophsd_handoff.get('status')}; real_bm25_closed_loop_for_ophsd={ophsd_handoff.get('real_bm25_closed_loop_for_ophsd')}; existing cells are route_head.pt summaries.",
            "next_action": "Implement faithful OPHSD collection: on-policy student states, whole-harness teacher context, actual teacher response_text/tool calls, matched update budget, and no-privilege actual Student closed-loop eval.",
        },
        {
            "experiment": "Structured actual Student V1/V2 redesign",
            "status": "resolved_blocked_after_audit",
            "reason": "The available structured AUTO dataset is sufficient for route-head residual diagnostics but not for actual Student LoRA. It contains P_tool_name_full/reduced and structured/textual fields, but lacks prompt_reduced, prompt_full, and response_text required by hf_tool_opd.py. AUTO actual-LoRA old recipe already failed the real closed-loop gate, so route-head-only V1/V2 redesign would violate 0816-2.",
            "evidence": f"structured keys={structured_schema['keys']}; missing_hf_tool_opd={structured_schema['missing_hf_tool_opd_fields']}; AUTO real_closed_loop_pass={auto_handoff.get('real_closed_loop_pass')}",
            "next_action": "Start a substantive actual-model redesign only after collecting prompt/response rows for structured residual/typed teacher branches; do not launch another route-head sweep as the final result.",
        },
    ]

    with (OUT / "0816_2_REMAINING_RESOLUTION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resolutions[0]))
        writer.writeheader()
        writer.writerows(resolutions)

    (OUT / "0816_2_REMAINING_RESOLUTION.json").write_text(json.dumps({"checks": checks, "resolutions": resolutions}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 0816-2 Remaining Experiment Resolution",
        "",
        "Status: all remaining items have been attempted through contract audit. None can be launched as a valid actual-model experiment from the current artifacts without creating a new data/runner contract.",
        "",
        "## Environment",
        "",
        f"- /opt environment checked: `{bishop}`",
        f"- torch/transformers/peft/vllm/pyserini: `{env['torch']}/{env['transformers']}/{env['peft']}/{env['vllm']}/{env['pyserini']}`",
        f"- vLLM binary: `{env['vllm_binary']}`",
        "",
        "The environment is not the main blocker; missing experiment contracts/data are.",
        "",
        "## Resolutions",
        "",
    ]
    for r in resolutions:
        lines += [
            f"### {r['experiment']}",
            "",
            f"- status: `{r['status']}`",
            f"- reason: {r['reason']}",
            f"- evidence: {r['evidence']}",
            f"- next valid action: {r['next_action']}",
            "",
        ]
    lines += [
        "## Status Buckets",
        "",
        "```text",
        "已完成:",
        "- AUTO actual-LoRA real closed-loop + controls: completed, failed gate.",
        "- importance_tagging proper K4/K8 fork: completed, failed gate.",
        "- Full Harness exact reference: resolved blocked by missing same-contract runner binding, not by GPU/env.",
        "- Matched Text actual LoRA: resolved blocked by missing prompt/teacher/response training contract.",
        "- OPHSD actual LoRA: resolved blocked by missing faithful prompt/response actual-LoRA contract.",
        "- Structured actual-model redesign: resolved blocked after audit; existing data supports route-head only, not valid actual-LoRA V1/V2.",
        "",
        "正在进行:",
        "- None.",
        "",
        "未开始:",
        "- None under the current 0816-2 artifact set. New work requires new data/runner-contract implementation, not simply launching existing experiments.",
        "```",
        "",
    ]
    (OUT / "0816_2_REMAINING_RESOLUTION.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "stage": "0816_2_remaining_resolution_final",
        "status": "completed_with_resolved_blockers",
        "generated_files": [
            "0816_2_REMAINING_RESOLUTION.md",
            "0816_2_REMAINING_RESOLUTION.json",
            "0816_2_REMAINING_RESOLUTION.csv",
            "RUN_MANIFEST.json",
            "SHA256SUMS",
        ],
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in files) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
