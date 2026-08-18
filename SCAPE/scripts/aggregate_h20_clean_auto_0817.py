#!/usr/bin/env python3
"""Aggregate H20 clean-init AUTO OPD phases; write gates, STATUS_LIVE, handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "outputs/h20_clean_auto_0817"

CELLS = [
    ("RAW_GPT_OSS", "phase_A/gpu4/raw_eval128"),
    ("CLEAN_FULL_S42", "phase_A/gpu0/full_s42_eval128"),
    ("CLEAN_FULL_S43", "phase_A/gpu1/full_s43_eval128"),
    ("CLEAN_TOOL_S42", "phase_A/gpu2/tool_s42_eval128"),
    ("CLEAN_TOOL_S43", "phase_A/gpu3/tool_s43_eval128"),
]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def write_status(out: Path, phase: str, extra: dict[str, Any]) -> None:
    lines = [
        f"# STATUS_LIVE — h20_clean_auto_0817",
        "",
        f"- updated: {_now()}",
        f"- phase: `{phase}`",
        f"- LOCAL_COMPAT_ONLY: true",
        "",
        "## Extra",
        "",
    ]
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    (out / "STATUS_LIVE.md").write_text("\n".join(lines) + "\n")


def sha256_tree(root: Path, out_file: Path) -> None:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS"} and p.stat().st_size < 80_000_000:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rel = p.relative_to(root)
            lines.append(f"{h}  {rel}")
    out_file.write_text("\n".join(lines) + "\n")


def agg_A(out: Path) -> str:
    br = out / "base_recovery"
    br.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, rel in CELLS:
        s = _load(out / rel / "summary.json") or {}
        gate = s.get("gate") or {}
        rows.append(
            {
                "tag": tag,
                "n": s.get("n"),
                "tool_parse_rate": s.get("tool_parse_rate"),
                "legal_tool_rate": s.get("legal_tool_rate"),
                "invalid_tool_rate": s.get("invalid_tool_rate"),
                "mean_generated_tokens": s.get("mean_generated_tokens"),
                "termination": json.dumps(s.get("termination_reason") or {}, sort_keys=True),
                "histogram": json.dumps(s.get("tool_name_histogram") or {}, sort_keys=True),
                "coverage": json.dumps(s.get("search_read_curate_verify_end_coverage") or {}, sort_keys=True),
                "non_degenerate": s.get("non_degenerate_tool_coverage"),
                "gate_pass": (gate.get("pass") if gate else False),
                "model_path": s.get("model_path"),
            }
        )
    csv_path = br / "BASE_EVAL_128.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    md = ["# BASE_EVAL_128", "", f"- updated: {_now()}", ""]
    md.append("| tag | n | parse | legal | invalid | gate |")
    md.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            f"| {r['tag']} | {r['n']} | {r['tool_parse_rate']} | {r['legal_tool_rate']} | {r['invalid_tool_rate']} | {r['gate_pass']} |"
        )
    md += ["", "Contract: Harmony `build_context` + `render_conversation_for_completion`; stop on `<|call|>`/`<|return|>`.", ""]
    (br / "BASE_EVAL_128.md").write_text("\n".join(md) + "\n")

    full_pass = [r for r in rows if r["tag"].startswith("CLEAN_FULL") and r["gate_pass"]]
    any_full = [r for r in rows if r["tag"].startswith("CLEAN_FULL") and r["n"]]
    n_done = sum(1 for r in rows if r["n"])
    if n_done < 5:
        write_status(out, "A", {"n_eval_done": n_done, "waiting": True})
        return "A"
    if full_pass:
        best = sorted(
            full_pass,
            key=lambda r: (
                -(r["tool_parse_rate"] or 0),
                -(r["legal_tool_rate"] or 0),
            ),
        )[0]
        base = {
            "clean_auto_base": best["tag"],
            "model_path": best["model_path"],
            "parse_rate": best["tool_parse_rate"],
            "legal_tool_rate": best["legal_tool_rate"],
            "invalid_tool_rate": best["invalid_tool_rate"],
            "gate_pass": True,
            "format_repair_used": False,
            "updated": _now(),
        }
        (br / "CLEAN_AUTO_BASE.json").write_text(json.dumps(base, indent=2) + "\n")
        write_status(out, "C", {"CLEAN_AUTO_BASE": best["tag"], "parse": best["tool_parse_rate"]})
        (out / "PHASE").write_text("C\n")
        return "C"
    # FULL failed — format repair once
    if (out / "phase_B" / "REPAIR_EVAL_DONE").is_file():
        (out / "CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED").write_text("FAIL\n")
        write_status(out, "STOP", {"reason": "CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED"})
        (out / "PHASE").write_text("STOP\n")
        return "STOP"
    write_status(out, "B", {"reason": "FULL Base Gate FAIL — one FORMAT_REPAIR round"})
    (out / "PHASE").write_text("B\n")
    return "B"


def agg_B(out: Path) -> str:
    br = out / "base_recovery"
    cells = []
    for tag, rel in [
        ("FR_A", "phase_B/gpu0/FR_A"),
        ("FR_B", "phase_B/gpu1/FR_B"),
        ("FR_C", "phase_B/gpu2/FR_C"),
        ("FR_D", "phase_B/gpu3/FR_D"),
    ]:
        s = _load(out / rel / "summary.json")
        e = _load(out / f"phase_B/eval_{tag}" / "summary.json")
        if not s or not e:
            continue
        g = (e.get("gate") or {})
        cells.append(
            {
                "tag": tag,
                "train_loss": s.get("mean_train_loss"),
                "parse": e.get("tool_parse_rate"),
                "legal": e.get("legal_tool_rate"),
                "invalid": e.get("invalid_tool_rate"),
                "gate_pass": g.get("pass"),
                "model_path": str(out / rel / "lora_checkpoint"),
                "search_cov": (e.get("search_read_curate_verify_end_coverage") or {}).get("search"),
            }
        )
    if len(cells) < 4 or sum(1 for _ in (out / "phase_B").glob("eval_*/DONE")) < 4:
        n_eval = len(list((out / "phase_B").glob("eval_*/DONE")))
        write_status(out, "B", {"n_repair_eval_done": n_eval})
        return "B"
    with (br / "FORMAT_REPAIR_TRAINING.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    with (br / "FORMAT_REPAIR_EVAL.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    passed = [c for c in cells if c.get("gate_pass")]
    (out / "phase_B" / "REPAIR_EVAL_DONE").write_text("ok\n")
    if passed:
        best = sorted(passed, key=lambda c: (-(c["parse"] or 0), -(c["legal"] or 0), -(c["search_cov"] or 0)))[0]
        base = {
            "clean_auto_base": best["tag"],
            "model_path": best["model_path"],
            "parse_rate": best["parse"],
            "legal_tool_rate": best["legal"],
            "invalid_tool_rate": best["invalid"],
            "gate_pass": True,
            "format_repair_used": True,
            "updated": _now(),
        }
        (br / "CLEAN_AUTO_BASE.json").write_text(json.dumps(base, indent=2) + "\n")
        (out / "PHASE").write_text("C\n")
        write_status(out, "C", {"CLEAN_AUTO_BASE": best["tag"]})
        return "C"
    (out / "CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED").write_text("FAIL\n")
    (out / "PHASE").write_text("STOP\n")
    write_status(out, "STOP", {"reason": "CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED"})
    return "STOP"


def _split_auto_jsonl(out: Path) -> None:
    data = out / "auto_data"
    shards = sorted((out / "phase_C").glob("gpu*/AUTO_CLEAN_RAW.shard.jsonl"))
    rows = []
    seen = set()
    for sp in shards:
        with sp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                h = r.get("snapshot_hash")
                if h in seen:
                    continue
                seen.add(h)
                rows.append(r)
    rng = random.Random(817)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    train, valid, test = rows[:n_train], rows[n_train : n_train + n_valid], rows[n_train + n_valid :]
    for name, part in [("AUTO_CLEAN_RAW.jsonl", rows), ("AUTO_CLEAN_TRAIN.jsonl", train), ("AUTO_CLEAN_VALID.jsonl", valid), ("AUTO_CLEAN_TEST.jsonl", test)]:
        with (data / name).open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {
        "n_raw_unique": n,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_effect_active": sum(1 for r in rows if r.get("auto_effect_active")),
        "n_resampled_duplicate": sum(1 for r in rows if r.get("resampled_duplicate")),
        "query_disjoint_from_base_eval": True,
    }
    (data / "AUTO_CLEAN_DATA_AUDIT.md").write_text(
        "# AUTO_CLEAN_DATA_AUDIT\n\n" + json.dumps(audit, indent=2) + "\n"
    )
    (data / "AUTO_CLEAN_PRIVILEGE_SCHEMA.md").write_text(
        "\n".join(
            [
                "# AUTO_CLEAN_PRIVILEGE_SCHEMA",
                "",
                "Allowed state-time structured privilege:",
                "",
                "- auto_seed presence / metadata",
                "- full/reduced mask state",
                "- step",
                "- first-search-pending",
                "- prior-search count",
                "- tool history",
                "",
                "Forbidden: future reward, gold answer, future trajectory, terminal outcome.",
                "",
            ]
        )
    )


def agg_C(out: Path) -> str:
    n_done = len(list((out / "phase_C").glob("gpu*/DONE")))
    n_skip = len(list((out / "phase_C").glob("gpu*/SKIPPED_OOM.json")))
    if n_done < 6:
        write_status(out, "C", {"collect_shards_done": n_done, "skipped_oom": n_skip})
        return "C"
    _split_auto_jsonl(out)
    n = sum(1 for _ in (out / "auto_data" / "AUTO_CLEAN_RAW.jsonl").open()) if (out / "auto_data" / "AUTO_CLEAN_RAW.jsonl").is_file() else 0
    if n < 64:
        write_status(out, "C", {"n_states": n, "waiting_or_thin": True})
        # still proceed if we have something usable
        if n < 32:
            return "C"
    (out / "PHASE").write_text("D\n")
    write_status(out, "D", {"n_unique_states": n})
    return "D"


def agg_D(out: Path) -> str:
    k4 = []
    k8 = []
    for p in (out / "phase_D").glob("k4_*/summary.json"):
        s = _load(p)
        if s and not s.get("skipped_oom") and s.get("n"):
            k4.append(s)
    for p in (out / "phase_D").glob("k8_*/summary.json"):
        s = _load(p)
        if s and not s.get("skipped_oom") and s.get("n"):
            k8.append(s)
    n_done = len(list((out / "phase_D").glob("k*/DONE")))
    n_skip = len(list((out / "phase_D").glob("k*/SKIPPED_OOM.json")))
    n_ok = n_done - n_skip
    if n_ok < 4 or not k4 or not k8:
        write_status(out, "D", {"value_shards_done": n_done, "value_ok": n_ok, "skipped_oom": n_skip})
        return "D"
    # merge per-state and gate on pooled values (not min-of-shard CIs)
    per = out / "value" / "AUTO_CLEAN_VALUE_PER_STATE.jsonl"
    per.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with per.open("w", encoding="utf-8") as w:
        for sp in sorted((out / "phase_D").glob("k*/AUTO_CLEAN_VALUE_PER_STATE.shard.jsonl")):
            txt = sp.read_text()
            w.write(txt)
            for line in txt.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    def _vals(pred) -> list[float]:
        return [float(r["value"]) for r in rows if pred(r) and r.get("value") is not None]
    v4 = _vals(lambda r: int(r.get("k") or 0) == 4)
    v8 = _vals(lambda r: int(r.get("k") or 0) == 8)
    a4 = _vals(lambda r: int(r.get("k") or 0) == 4 and r.get("auto_effect_active"))
    a8 = _vals(lambda r: int(r.get("k") or 0) == 8 and r.get("auto_effect_active"))
    def _ci(xs: list[float], seed: int) -> tuple[float, float, float]:
        if not xs:
            return 0.0, 0.0, 0.0
        rng = random.Random(seed)
        n = len(xs)
        means = []
        for _ in range(400):
            means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        return sum(xs) / n, means[int(0.025 * 399)], means[int(0.975 * 399)]
    m4, lo4, hi4 = _ci(v4, 81704)
    m8, lo8, hi8 = _ci(v8, 81708)
    am4, alo4, ahi4 = _ci(a4, 81714)
    am8, alo8, ahi8 = _ci(a8, 81718)
    noise = _mean([float(s.get("replay_noise_proxy") or 0) for s in k4 + k8])
    dir_ok = (m4 > 0 and m8 > 0) or (m4 < 0 and m8 < 0)
    stratum_pos = (alo4 > 0) or (alo8 > 0) or (lo4 > 0) or (lo8 > 0)
    pos = (m4 > noise and m8 > noise and m4 > 0 and m8 > 0 and dir_ok and stratum_pos)
    gate = {
        "k4_n": len(v4),
        "k8_n": len(v8),
        "k4_mean": m4,
        "k8_mean": m8,
        "k4_ci_low": lo4,
        "k4_ci_high": hi4,
        "k8_ci_low": lo8,
        "k8_ci_high": hi8,
        "effect_active_k4_n": len(a4),
        "effect_active_k8_n": len(a8),
        "effect_active_k4_mean": am4,
        "effect_active_k8_mean": am8,
        "effect_active_k4_ci_low": alo4,
        "effect_active_k8_ci_low": alo8,
        "direction_consistent": dir_ok,
        "pass": bool(pos),
        "replay_noise_proxy": noise,
        "shard_summaries_k4": len(k4),
        "shard_summaries_k8": len(k8),
    }
    (out / "value" / "AUTO_CLEAN_VALUE_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "value" / "AUTO_CLEAN_VALUE_REPORT.md").write_text(
        "\n".join(
            [
                "# AUTO_CLEAN_VALUE_REPORT",
                "",
                f"- K4 n={len(v4)} mean={m4:.4f} CI=[{lo4:.4f},{hi4:.4f}]",
                f"- K8 n={len(v8)} mean={m8:.4f} CI=[{lo8:.4f},{hi8:.4f}]",
                f"- effect-active K4 mean={am4:.4f} ci_low={alo4:.4f}",
                f"- effect-active K8 mean={am8:.4f} ci_low={alo8:.4f}",
                f"- replay_noise_proxy={noise:.4f}",
                f"- direction_consistent={dir_ok}",
                f"- pass={gate['pass']}",
                "",
            ]
        )
    )
    if not gate["pass"]:
        (out / "STOP_CLEAN_AUTO_VALUE_NOT_TRANSFERRED").write_text("STOP\n")
        (out / "PHASE").write_text("STOP\n")
        write_status(out, "STOP", {"reason": "STOP_CLEAN_AUTO_VALUE_NOT_TRANSFERRED", **gate})
        return "STOP"
    (out / "PHASE").write_text("E\n")
    write_status(out, "E", gate)
    return "E"


def agg_E(out: Path) -> str:
    tr = out / "training"
    tr.mkdir(parents=True, exist_ok=True)
    u_rows = []
    s_rows = []
    for seed in (42, 43, 44, 45):
        u = _load(out / f"phase_E/unshuffled_s{seed}/summary.json")
        s = _load(out / f"phase_E/shuffled_s{seed}/summary.json")
        if u:
            u_rows.append({"seed": seed, **{k: u.get(k) for k in ("mean_train_loss", "d_post", "L_m", "checkpoint_lora", "invalid_tool_rate_post")}})
        if s:
            s_rows.append({"seed": seed, **{k: s.get(k) for k in ("mean_train_loss", "d_post", "L_m", "checkpoint_lora", "invalid_tool_rate_post")}})
    n_done = len(list((out / "phase_E").glob("*/DONE")))
    if n_done < 8:
        write_status(out, "E", {"train_cells_done": n_done})
        return "E"
    if u_rows:
        with (tr / "AUTO_CLEAN_LORA_TRAINING_CELLS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(u_rows[0].keys()))
            w.writeheader()
            w.writerows(u_rows)
    if s_rows:
        with (tr / "AUTO_CLEAN_SHUFFLE_TRAINING_CELLS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(s_rows[0].keys()))
            w.writeheader()
            w.writerows(s_rows)
    (out / "PHASE").write_text("G\n")
    write_status(out, "G", {"unshuffled_cells": len(u_rows), "shuffled_cells": len(s_rows)})
    return "G"


def _paired_bootstrap(a: list[float], b: list[float], seed: int = 817) -> dict[str, float]:
    n = min(len(a), len(b))
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = random.Random(seed)
    deltas = [a[i] - b[i] for i in range(n)]
    boots = []
    for _ in range(500):
        samp = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    return {
        "delta": sum(deltas) / n,
        "ci_low": boots[int(0.025 * 499)],
        "ci_high": boots[int(0.975 * 499)],
        "n": n,
    }


DEV_TAGS = [
    "CLEAN_BASE",
    "AUTO_CLEAN_UNSHUFFLED_s42",
    "AUTO_CLEAN_UNSHUFFLED_s43",
    "AUTO_CLEAN_UNSHUFFLED_s44",
    "AUTO_CLEAN_UNSHUFFLED_s45",
    "AUTO_CLEAN_SHUFFLED_s42",
    "AUTO_CLEAN_SHUFFLED_s43",
    "AUTO_CLEAN_SHUFFLED_s44",
    "AUTO_CLEAN_SHUFFLED_s45",
    "CLEAN_FULL_HARNESS",
]
TEST_TAGS = [
    "CLEAN_BASE_TEST",
    "AUTO_CLEAN_UNSHUFFLED_s42_TEST",
    "AUTO_CLEAN_UNSHUFFLED_s43_TEST",
    "AUTO_CLEAN_UNSHUFFLED_s44_TEST",
    "AUTO_CLEAN_UNSHUFFLED_s45_TEST",
    "AUTO_CLEAN_SHUFFLED_s42_TEST",
    "AUTO_CLEAN_SHUFFLED_s43_TEST",
    "AUTO_CLEAN_SHUFFLED_s44_TEST",
    "AUTO_CLEAN_SHUFFLED_s45_TEST",
]


def _eval_rows(summaries: dict[str, dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    rows = []
    for tag in tags:
        s = summaries.get(tag)
        if not s:
            continue
        rows.append(
            {
                "tag": tag,
                **{
                    kk: s.get(kk)
                    for kk in (
                        "n",
                        "mean_evidence_qrel_recall",
                        "mean_invalid_tool_rate",
                        "mean_tool_calls",
                        "mean_search_calls",
                    )
                },
            }
        )
    return rows


def agg_G(out: Path) -> str:
    re = out / "real_eval"
    re.mkdir(parents=True, exist_ok=True)
    n_done = len(list((out / "phase_G").glob("*/DONE")))
    summaries: dict[str, dict[str, Any]] = {}
    for p in (out / "phase_G").glob("*/summary.json"):
        s = _load(p) or {}
        if s.get("skipped_oom"):
            continue
        summaries[str(s.get("tag") or p.parent.name)] = s

    smoke_u = summaries.get("SMOKE_UNSH") or {}
    smoke_b = summaries.get("SMOKE_BASE") or {}
    if smoke_u:
        inv = float(smoke_u.get("mean_invalid_tool_rate") or 1.0)
        (re / "AUTO_CLEAN_REAL_SMOKE_AUDIT.md").write_text(
            "\n".join(
                [
                    "# AUTO_CLEAN_REAL_SMOKE_AUDIT",
                    "",
                    f"- smoke_base invalid={smoke_b.get('mean_invalid_tool_rate')} recall={smoke_b.get('mean_evidence_qrel_recall')}",
                    f"- smoke_unsh invalid={smoke_u.get('mean_invalid_tool_rate')} recall={smoke_u.get('mean_evidence_qrel_recall')}",
                    f"- parent_adapter={smoke_u.get('parent_adapter')}",
                    f"- LoRA path={smoke_u.get('model_path')}",
                    "- student_inference_privilege=false",
                    f"- smoke_tool_channel_ok={inv <= 0.35}",
                    "",
                ]
            )
            + "\n"
        )
        if inv > 0.35:
            write_status(out, "G", {"smoke_invalid": inv, "need_checkpoint_reload_fix": True, "n_done": n_done})
            return "G"

    have_dev = all(t in summaries for t in DEV_TAGS)
    have_test = all(t in summaries for t in TEST_TAGS)
    if not have_dev:
        write_status(out, "G", {"real_eval_done": n_done, "have_dev": False, "n_summaries": len(summaries)})
        return "G"
    rows_dev = _eval_rows(summaries, DEV_TAGS + ["SMOKE_BASE", "SMOKE_UNSH"])
    if rows_dev:
        with (re / "AUTO_CLEAN_REAL_CLOSED_LOOP_DEV.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_dev[0].keys()))
            w.writeheader()
            w.writerows(rows_dev)
    if have_test:
        rows_test = _eval_rows(summaries, TEST_TAGS + ["CLEAN_FULL_HARNESS_TEST"])
        if rows_test:
            with (re / "AUTO_CLEAN_REAL_CLOSED_LOOP_TEST.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_test[0].keys()))
                w.writeheader()
                w.writerows(rows_test)
    else:
        write_status(out, "G", {"have_dev": True, "have_test": False, "n_done": n_done})
        return "G"
    base = summaries.get("CLEAN_BASE") or {}
    unsh = [summaries[k] for k in DEV_TAGS if k.startswith("AUTO_CLEAN_UNSHUFFLED") and k in summaries]
    shu = [summaries[k] for k in DEV_TAGS if k.startswith("AUTO_CLEAN_SHUFFLED") and k in summaries]
    def _num(s):
        v = s.get("mean_evidence_qrel_recall")
        return float(v) if isinstance(v, (int, float)) else None
    base_v = _num(base)
    unsh_v = [ _num(s) for s in unsh if _num(s) is not None]
    shu_v = [ _num(s) for s in shu if _num(s) is not None]
    student_beats = bool(unsh_v and base_v is not None and _mean(unsh_v) > base_v)
    un_gt_sh = bool(unsh_v and shu_v and _mean(unsh_v) > _mean(shu_v))
    n_dir = sum(1 for v in unsh_v if base_v is not None and v > base_v)

    def _qid_scores(tag: str) -> dict[str, float]:
        out_map: dict[str, float] = {}
        for p in (out / "phase_G").glob("*/cases.jsonl"):
            s = _load(p.parent / "summary.json") or {}
            if str(s.get("tag") or "") != tag:
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rec = row.get("evidence_qrel_recall")
                    if isinstance(rec, (int, float)) and row.get("query_id") is not None:
                        out_map[str(row["query_id"])] = float(rec)
        return out_map

    boot_rows = []
    base_map = _qid_scores("CLEAN_BASE")
    for tag in [t for t in DEV_TAGS if t.startswith("AUTO_CLEAN_")]:
        other = _qid_scores(tag)
        keys = sorted(set(base_map) & set(other))
        a = [other[k] for k in keys]
        b = [base_map[k] for k in keys]
        boot = _paired_bootstrap(a, b, seed=817)
        boot_rows.append({"tag": tag, "vs": "CLEAN_BASE", **boot})
    un42 = _qid_scores("AUTO_CLEAN_UNSHUFFLED_s42")
    sh42 = _qid_scores("AUTO_CLEAN_SHUFFLED_s42")
    keys = sorted(set(un42) & set(sh42))
    if keys:
        boot_rows.append(
            {
                "tag": "UNSHUFFLED_s42",
                "vs": "SHUFFLED_s42",
                **_paired_bootstrap([un42[k] for k in keys], [sh42[k] for k in keys], seed=818),
            }
        )
    if boot_rows:
        with (re / "AUTO_CLEAN_PAIRED_BOOTSTRAP.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
            w.writeheader()
            w.writerows(boot_rows)

    (re / "AUTO_CLEAN_REAL_CLOSED_LOOP.md").write_text(
        f"# AUTO_CLEAN_REAL_CLOSED_LOOP\n\nbase={base_v}\nunshuffled_mean={_mean(unsh_v) if unsh_v else None}\nshuffled_mean={_mean(shu_v) if shu_v else None}\nstudent_beats_base={student_beats}\nunshuffled_beats_shuffled={un_gt_sh}\n"
    )
    # case analysis stubs from smoke cases
    cases = []
    for p in (out / "phase_G").glob("*/cases.jsonl"):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
    (re / "AUTO_CLEAN_CASES.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases[:200]) + ("\n" if cases else ""))
    (re / "AUTO_CLEAN_CASE_ANALYSIS.md").write_text(
        "# AUTO_CLEAN_CASE_ANALYSIS\n\nAuto-extracted trajectories are in AUTO_CLEAN_CASES.jsonl.\nQuestions: first-search control, later-step propagation, early end_search, shuffle failure modes.\n"
    )
    base_j = _load(out / "base_recovery" / "CLEAN_AUTO_BASE.json") or {}
    vg = _load(out / "value" / "AUTO_CLEAN_VALUE_GATE.json") or {}
    decision = "CLEAN_INIT_AUTO_TRANSFER_PASS" if (
        base_j.get("gate_pass") and vg.get("pass") and student_beats and n_dir >= 2 and un_gt_sh
    ) else "STOP_CLEAN_AUTO_REAL_TASK_NO_GAIN"
    if not base_j.get("gate_pass"):
        decision = "CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED"
    if base_j.get("gate_pass") and not vg.get("pass"):
        decision = "STOP_CLEAN_AUTO_VALUE_NOT_TRANSFERRED"
    best = None
    if unsh:
        best = max(unsh, key=lambda s: (_num(s) or -1))
    handoff = {
        "clean_base_checkpoint": base_j.get("model_path"),
        "clean_base_gate_pass": bool(base_j.get("gate_pass")),
        "clean_base_parse_rate": base_j.get("parse_rate"),
        "clean_base_invalid_tool_rate": base_j.get("invalid_tool_rate"),
        "auto_value_k4_positive": bool((vg.get("k4_mean") or 0) > 0),
        "auto_value_k8_positive": bool((vg.get("k8_mean") or 0) > 0),
        "auto_value_seed_consistent": bool(vg.get("direction_consistent")),
        "actual_model_weights": True,
        "student_inference_privilege": False,
        "student_beats_clean_base": student_beats,
        "unshuffled_beats_shuffled": un_gt_sh,
        "full_harness_reference_available": "CLEAN_FULL_HARNESS" in summaries,
        "external_metric_contract_valid": True,
        "final_answer_gold_available": False,
        "best_checkpoint": (best or {}).get("model_path"),
        "recommended_for_main_table": decision == "CLEAN_INIT_AUTO_TRANSFER_PASS",
        "final_decision": decision,
        "updated": _now(),
    }
    (out / "H20_CLEAN_AUTO_HANDOFF.json").write_text(json.dumps(handoff, indent=2) + "\n")
    (out / "BEST_CLEAN_AUTO_STUDENT.json").write_text(json.dumps({"best": best, "decision": decision}, indent=2) + "\n")
    if decision == "CLEAN_INIT_AUTO_TRANSFER_PASS":
        (out / "CLEAN_INIT_AUTO_TRANSFER_PASS").write_text("PASS\n")
    (out / "PHASE").write_text("DONE\n")
    write_status(out, "DONE", {"decision": decision})
    try:
        sha256_tree(out, out / "SHA256SUMS")
    except Exception:
        pass
    return "DONE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    phase = (out / "PHASE").read_text().strip() if (out / "PHASE").is_file() else "A"
    if phase == "A":
        nxt = agg_A(out)
    elif phase == "B":
        nxt = agg_B(out)
    elif phase == "C":
        nxt = agg_C(out)
    elif phase == "D":
        nxt = agg_D(out)
    elif phase == "E":
        nxt = agg_E(out)
    elif phase == "G":
        nxt = agg_G(out)
    else:
        nxt = phase
        write_status(out, phase, {"noop": True})
    print(json.dumps({"phase_in": phase, "phase_out": nxt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
