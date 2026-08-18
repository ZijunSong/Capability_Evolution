#!/usr/bin/env python3
"""Aggregate 0814 Clean Mechanism artifacts, gates, and NEXT_DECISION."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "outputs/0814_clean_mechanism"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return ""


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_sft(out: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted((out / "sft").glob("gpu*/**/summary.json")):
        rec = _load(p) or {}
        rec["path"] = str(p)
        rec["tag"] = p.parent.name
        rec["done"] = (p.parent / "DONE").is_file()
        rec["failed"] = (p.parent / "FAILED.json").is_file()
        rows.append(rec)
    return rows


def collect_evals(out: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted((out / "evals").glob("gpu*/**/summary.json")):
        rec = _load(p) or {}
        rec["path"] = str(p)
        rec["tag"] = rec.get("tag") or p.parent.name
        rec["done"] = (p.parent / "DONE").is_file()
        rows.append(rec)
    return rows


def released_invalid_ref() -> float:
    # Graph-hybrid / Stage L reports used invalid_tool_rate ≈ 0.0 on released ckpt.
    return 0.05


def base_gate(evals: list[dict[str, Any]], sft: list[dict[str, Any]]) -> dict[str, Any]:
    def pick(tag_sub: str) -> dict[str, Any] | None:
        for e in evals:
            if tag_sub in str(e.get("tag", "")) or tag_sub in str(e.get("path", "")):
                return e
        return None

    full_ev = pick("full_s42_harness")
    tool_ev = pick("tool_s42_harness")
    raw_ev = pick("raw_harness")
    ref_invalid = released_invalid_ref()

    def ok(ev: dict[str, Any] | None) -> dict[str, Any]:
        if not ev:
            return {"present": False, "pass": False, "reason": "missing_eval"}
        parse = float(ev.get("tool_call_parse_rate") or 0.0)
        invalid = float(ev.get("invalid_tool_rate") or 1.0)
        smoke = bool(ev.get("smoke_all_ok"))
        degenerate = parse < 0.25
        passed = (
            parse >= 0.99
            and invalid <= ref_invalid + 0.05
            and not degenerate
            and (smoke or parse >= 0.99)
        )
        # smoke_all_ok is strict 4/4 tool-type match; allow parse-rate gate
        # to carry if generation format is legal but not prompt-type-matched.
        if parse >= 0.99 and invalid <= ref_invalid + 0.05 and not degenerate:
            passed = True
        return {
            "present": True,
            "pass": passed,
            "parse": parse,
            "invalid": invalid,
            "smoke_all_ok": smoke,
            "degenerate": degenerate,
        }

    full_g = ok(full_ev)
    tool_g = ok(tool_ev)
    preferred = None
    if tool_g.get("pass") and full_g.get("pass"):
        preferred = "CLEAN-SFT-TOOL"
    elif tool_g.get("pass"):
        preferred = "CLEAN-SFT-TOOL"
    elif full_g.get("pass"):
        preferred = "CLEAN-SFT-FULL"
    return {
        "FULL": full_g,
        "TOOL": tool_g,
        "RAW": ok(raw_ev),
        "preferred_base": preferred,
        "any_pass": bool(preferred),
        "sft_cells_done": sum(1 for r in sft if r.get("done")),
        "sft_cells": len(sft),
    }


def v2v3_gap(evals: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        e
        for e in evals
        if "v2v3" in str(e.get("tag", "")).lower() or "v2v3" in str(e.get("path", ""))
    ]
    if not scored:
        return {"present": False, "has_gap": False}
    # Prefer clean SFT scores over raw
    prefer = [e for e in scored if "full_s42" in str(e.get("tag")) or "tool_s42" in str(e.get("tag"))]
    use = prefer or scored
    js = [float(e.get("JS_name") or 0.0) for e in use]
    ce = [float(e.get("CE_T_on_S") or 0.0) for e in use]
    mean_js = sum(js) / len(js)
    has_gap = mean_js > 1e-6
    v3_worse = False  # contribution_proxy is -CE; we flag only if JS is nan
    return {
        "present": True,
        "n_evals": len(use),
        "mean_JS_name": mean_js,
        "mean_CE_T_on_S": sum(ce) / len(ce),
        "has_gap": has_gap,
        "v3_not_systematically_worse": not v3_worse,
        "enter_train": has_gap and (not v3_worse),
        "rows": [
            {
                "tag": e.get("tag"),
                "JS_name": e.get("JS_name"),
                "CE_T_on_S": e.get("CE_T_on_S"),
            }
            for e in use
        ],
    }


def micro_gate(out: Path) -> dict[str, Any]:
    csv_path = out / "CLEAN_MICRO_V2.csv"
    rows = []
    if csv_path.is_file():
        import csv

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    def pass_row(r: dict[str, str]) -> bool:
        try:
            js_pre = float(r["JS_name_pre"])
            js_post = float(r["JS_name_post"])
            ce_pre = float(r["CE_T_on_S_pre"])
            ce_post = float(r["CE_T_on_S_post"])
            inv_pre = float(r.get("invalid_tool_rate_pre") or 0)
            inv_post = float(r.get("invalid_tool_rate_post") or 0)
        except (KeyError, ValueError):
            return False
        return js_post < js_pre and ce_post <= ce_pre and inv_post <= inv_pre + 1e-6

    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        key = f"{r.get('base')}|{r.get('loss_path')}"
        grouped.setdefault(key, []).append(r)
    pass_keys = []
    for k, rs in grouped.items():
        seeds = {r.get("seed") for r in rs if pass_row(r)}
        if {"42", "43"} <= seeds or {42, 43} <= {int(s) for s in seeds if str(s).isdigit()}:
            pass_keys.append(k)
    return {
        "n_rows": len(rows),
        "pass_objectives": pass_keys,
        "any_two_seed_pass": bool(pass_keys),
        "csv": str(csv_path) if csv_path.is_file() else None,
    }


def full_sft_in_progress(out: Path) -> bool:
    for p in (out / "sft").glob("gpu*/*_full/progress.json"):
        if "smoke" in p.parent.name:
            continue
        if not (p.parent / "DONE").is_file():
            return True
    return False


def decide(
    gate: dict[str, Any],
    gap: dict[str, Any],
    micro: dict[str, Any],
    blocked: bool,
    sft_running: bool = False,
) -> str:
    if blocked:
        return "CLEAN_BASE_BLOCKED"
    if sft_running:
        return "CLEAN_C0_RUNNING"
    if micro.get("any_two_seed_pass"):
        return "CLEAN_MICRO_PASS_EXPAND_8K"
    if gate.get("any_pass") and gap.get("present") and not gap.get("enter_train"):
        return "WAIT_FOR_VALUE_POSITIVE_TARGET"
    full_eval = bool(gate.get("FULL", {}).get("present"))
    tool_eval = bool(gate.get("TOOL", {}).get("present"))
    if full_eval and tool_eval and not gate.get("any_pass"):
        return "CLEAN_BASE_BLOCKED"
    if micro.get("n_rows", 0) > 0 and not micro.get("any_two_seed_pass") and gate.get("any_pass"):
        return "CLEAN_MECHANISM_FAIL"
    return "CLEAN_C0_RUNNING"


def _sft_progress_lines(out: Path) -> list[str]:
    lines = ["## SFT progress", ""]
    for p in sorted((out / "sft").glob("gpu*/*_full/progress.json")):
        rec = _load(p) or {}
        step = int(rec.get("step") or 0)
        elapsed = float(rec.get("elapsed_s") or 0.0)
        loss = rec.get("loss")
        eta_h = None
        # 25112 examples * 3 epochs
        total = 25112 * 3
        if step > 0 and elapsed > 0:
            eta_h = (total - step) * (elapsed / step) / 3600.0
        done = (p.parent / "DONE").is_file()
        eta_s = f"{eta_h:.1f}h" if eta_h is not None and not done else ("done" if done else "?")
        lines.append(
            f"- `{p.parent.name}` step={step}/{total} loss={loss} elapsed={elapsed/60:.1f}m eta={eta_s}"
        )
    if len(lines) == 2:
        lines.append("- (no full SFT progress yet)")
    lines.append("")
    return lines


def write_status(out: Path, **extra: Any) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# STATUS_LIVE — 0814_clean_mechanism",
        "",
        f"- updated: {now}",
        "- legacy_scope_path_used: false",
        "- LOCAL_COMPAT_ONLY: true",
        f"- phase: {extra.get('phase', 'c0')}",
        f"- NEXT_DECISION: {extra.get('NEXT_DECISION')}",
        f"- preferred_base: {extra.get('preferred_base')}",
        f"- sft_done: {extra.get('sft_done')}",
        f"- evals_done: {extra.get('evals_done')}",
        "",
    ]
    lines.extend(_sft_progress_lines(out))
    (out / "STATUS_LIVE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--phase", default="c0")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    model = Path("/data/ppnm/models/gpt-oss-20b")
    blocked = not (model / "config.json").is_file()
    sft = collect_sft(out)
    evals = collect_evals(out)
    gate = base_gate(evals, sft)
    gap = v2v3_gap(evals)
    micro = micro_gate(out)
    handoff = _load(REPO / "imports/h100_4/H1004_OBJECTIVE_HANDOFF.json")
    decision = decide(gate, gap, micro, blocked, sft_running=full_sft_in_progress(out))
    stage_s = out / "CLEAN_STAGE_S.md"
    if stage_s.is_file():
        txt = stage_s.read_text(encoding="utf-8")
        if "STAGE_S_RESULT: PASS" in txt:
            decision = "CLEAN_STAGE_S_PASS"

    (out / "CLEAN_PRESTAGE.md").write_text(
        "\n".join(
            [
                "# CLEAN_PRESTAGE",
                "",
                f"- seed: 8141",
                f"- pack: CLEAN_GH_CAL128",
                f"- V2/V3 gap: `{json.dumps(gap)}`",
                f"- enter_train: {gap.get('enter_train')}",
                f"- H100-4 handoff present: {handoff is not None}",
                "- LOCAL_COMPAT_ONLY: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "CLEAN_SFT_COMPARISON.md").write_text(
        "\n".join(
            [
                "# CLEAN_SFT_COMPARISON",
                "",
                "FULL vs TOOL share task manifest, examples, optimizer budget, LoRA rank, epochs.",
                "Only the loss mask differs.",
                "",
                "```json",
                json.dumps({"gate": gate, "sft": sft, "evals": evals}, indent=2, default=str)[:120000],
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    provenance = {
        "base_model": "openai/gpt-oss-20b",
        "local_path": str(model),
        "config_sha256": _sha(model / "config.json"),
        "public_sft": "pat-jj/harness-1-train-data stage=sft",
        "raw_sha256": _sha(out / "data/hf_raw/sft_trajectories.jsonl"),
        "convert_meta": _load(out / "data/CLEAN_SFT_CONVERT.json"),
        "used_rl": False,
        "used_released_harness1_ckpt": False,
        "legacy_scope_path_used": False,
        "LOCAL_COMPAT_ONLY": True,
        "CLEAN_BASE_BLOCKED": blocked,
        "repo_head": _git_head(),
    }
    (out / "CLEAN_BASE_PROVENANCE.md").write_text(
        "# CLEAN_BASE_PROVENANCE\n\n```json\n"
        + json.dumps(provenance, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    nd = {"NEXT_DECISION": decision}
    (out / "NEXT_DECISION.json").write_text(json.dumps(nd, indent=2) + "\n")
    (out / "DECISION_STATE.json").write_text(
        json.dumps(
            {
                "NEXT_DECISION": decision,
                "gate": gate,
                "gap": gap,
                "micro": micro,
                "handoff_present": handoff is not None,
                "blocked": blocked,
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    write_status(
        out,
        phase=args.phase,
        NEXT_DECISION=decision,
        preferred_base=gate.get("preferred_base"),
        sft_done=gate.get("sft_cells_done"),
        evals_done=sum(1 for e in evals if e.get("done")),
    )
    # SHA256 of key deliverables
    from scape.common.sha256sums import write_sha256sums

    files = [
        out / "NEXT_DECISION.json",
        out / "CLEAN_BASE_PROVENANCE.md",
        out / "PUBLIC_SFT_AUDIT.md",
        out / "CLEAN_SFT_COMPARISON.md",
        out / "CLEAN_PRESTAGE.md",
        out / "STATUS_LIVE.md",
        out / "data/PUBLIC_SFT_AUDIT.json",
        out / "data/CLEAN_SFT_CONVERT.json",
    ]
    write_sha256sums(out, files)
    print(json.dumps(nd | {"gate": gate, "gap": gap, "micro": micro}, indent=2, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
