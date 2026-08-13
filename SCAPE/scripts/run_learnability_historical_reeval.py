#!/usr/bin/env python3
"""Re-evaluate historical checkpoints with canonical KL/JS metrics on VALID split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.training.canonical_metrics import NUMERIC_FLOOR
from scape.training.hf_tool_opd import ScapeHFToolOPD, LossPath


CSV_FIELDS = [
    "family", "checkpoint_id", "checkpoint_path", "valid_jsonl", "loss_path",
    "forward_KL", "reverse_KL", "JS", "signed_gap",
    "tool_name_KL", "arg_key_KL", "arg_value_KL",
    "legacy_div", "n_valid", "kl_nonneg_ok",
]


def score_with_dual_backend(
    teacher: ScapeHFToolOPD,
    student: ScapeHFToolOPD,
    rows: list[dict[str, Any]],
    *,
    loss_path: LossPath,
    max_rows: int | None = None,
) -> dict[str, float]:
    keys = [
        "forward_KL", "reverse_KL", "JS", "signed_gap",
        "tool_name_KL", "arg_key_KL", "arg_value_KL", "div",
    ]
    acc = {k: 0.0 for k in keys}
    legacy_acc = 0.0
    subset = rows[:max_rows] if max_rows else rows
    for row in subset:
        resp_ids = student.encode(row["response_text"])
        if not resp_ids:
            continue
        red_ids = student.encode(row["prompt_reduced"])
        full_ids = teacher.encode(row["prompt_full"])
        with __import__("torch").no_grad():
            s_logits = student._response_position_logits(red_ids, resp_ids, require_grad=False)
            t_logits = teacher._response_position_logits(full_ids, resp_ids, require_grad=False)
        from scape.training.canonical_metrics import (
            aggregate_token_metrics,
            js_from_logits,
            kl_from_logits,
            signed_logprob_gap,
        )
        fwd = kl_from_logits(t_logits, s_logits, forward=True)
        rev = kl_from_logits(t_logits, s_logits, forward=False)
        js = js_from_logits(t_logits, s_logits)
        gap = signed_logprob_gap(t_logits, s_logits, resp_ids)
        spans = student.span_token_masks(row["response_text"], len(resp_ids))
        token_mask = student.response_token_mask(row["response_text"], loss_path=loss_path)
        if len(token_mask) != len(resp_ids):
            token_mask = spans["tool"]
        import torch
        m_tool = torch.tensor(token_mask, device=fwd.device, dtype=fwd.dtype)
        m_name = torch.tensor(spans["name"], device=fwd.device, dtype=fwd.dtype)
        m_key = torch.tensor(spans["key"], device=fwd.device, dtype=fwd.dtype)
        m_val = torch.tensor(spans["value"], device=fwd.device, dtype=fwd.dtype)
        m = aggregate_token_metrics(
            fwd, rev, js, gap, m_tool,
            name_mask=m_name, key_mask=m_key, value_mask=m_val,
        )
        legacy = student.score_divergence(
            prompt_reduced=row["prompt_reduced"],
            prompt_full=row["prompt_full"],
            response_text=row["response_text"],
            loss_path=loss_path,
        )
        acc["forward_KL"] += m["forward_KL"]
        acc["reverse_KL"] += m["reverse_KL"]
        acc["JS"] += m["JS"]
        acc["signed_gap"] += m["signed_gap"]
        acc["tool_name_KL"] += m["tool_name_KL"]
        acc["arg_key_KL"] += m["arg_key_KL"]
        acc["arg_value_KL"] += m["arg_value_KL"]
        acc["div"] += m["signed_gap"]
        legacy_acc += legacy["div"]
    n = max(1, len(subset))
    out = {k: acc[k] / n for k in keys}
    out["legacy_div"] = legacy_acc / n
    return out


def build_checkpoint_inventory(repo: Path) -> list[dict[str, Any]]:
    """Checkpoint list per SCAPE-0813-H20 §4."""
    items: list[dict[str, Any]] = []
    eg = repo / "outputs/true_scape_evidence_graph"
    eg_data = eg / "data/EG_VALID_1K.jsonl"
    eg_retry = eg / "stage_l_retry"
    eg_main = eg / "stage_l"

    def add(family: str, cid: str, path: str, valid: Path, loss: str = "tool_token_kl") -> None:
        p = Path(path)
        if not p.is_absolute():
            p = repo / p
        items.append({
            "family": family,
            "checkpoint_id": cid,
            "checkpoint_path": str(p),
            "valid_jsonl": str(valid),
            "loss_path": loss,
        })

    base = "/data/ppnm/models/harness-1"
    add("evidence_graph", "base", base, eg_data)

    for tag in [
        "main_L512_s42", "main_L2K_s42", "main_L8K_s42",
        "main_L512_s43", "main_L2K_s43", "main_L8K_s43",
        "main_L2K_s44", "main_L8K_s44",
    ]:
        for gpu_dir in eg_main.glob("gpu*"):
            ck = gpu_dir / tag / "hf_merged"
            if ck.exists():
                add("evidence_graph_uniform", tag, str(ck), eg_data)
                break

    for ck in sorted(eg_retry.glob("gpu*/weighted_*/hf_merged")):
        tag = ck.parent.name
        add("evidence_graph_weighted", tag, str(ck), eg_data, "weighted_tool_token_kl")

    for tag in ["baseline_name_only_L2K", "baseline_name_only_L8K"]:
        for gpu_dir in eg_main.glob("gpu*"):
            ck = gpu_dir / tag / "hf_merged"
            if ck.exists():
                add("evidence_graph_name_only", tag, str(ck), eg_data, "tool_name_only_kl")
                break

    tour = repo / "outputs/true_scape_candidate_b_tournament"
    tour_data = tour / "data"
    comp_map = {
        "subtractive_curation": "SC",
        "importance_tagging": "IT",
        "verify_tool": "VT",
    }
    for comp, prefix in comp_map.items():
        valid = tour_data / f"{comp}_VALID_512.jsonl"
        for tag_pat in [f"{prefix}_L512_s42", f"{prefix}_L2K_s42",
                        f"{prefix}_L512_s43", f"{prefix}_L2K_s43"]:
            for ck in tour.glob(f"stage_l_micro/gpu*/{tag_pat}/hf_merged"):
                add(comp, tag_pat, str(ck), valid)

    return items


def filter_inventory(items: list[dict[str, Any]], families: list[str] | None) -> list[dict[str, Any]]:
    if not families:
        return items
    allowed = set(families)
    return [x for x in items if x["family"] in allowed]


def load_done_keys(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("family") and row.get("checkpoint_id"):
                done.add((row["family"], row["checkpoint_id"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--teacher-path", default="/data/ppnm/models/harness-1")
    ap.add_argument("--families", nargs="*", default=None,
                    help="Filter: evidence_graph, evidence_graph_uniform, subtractive_curation, ...")
    ap.add_argument("--checkpoint-ids", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-valid", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--inventory-json", type=Path, default=None)
    args = ap.parse_args()

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.inventory_json and args.inventory_json.exists():
        items = json.loads(args.inventory_json.read_text())
    else:
        items = build_checkpoint_inventory(args.repo)

    items = filter_inventory(items, args.families)
    if args.checkpoint_ids:
        ids = set(args.checkpoint_ids)
        items = [x for x in items if x["checkpoint_id"] in ids]

    if args.skip_existing:
        done_keys = load_done_keys(out_csv)
        items = [x for x in items if (x["family"], x["checkpoint_id"]) not in done_keys]
    if not items:
        print(json.dumps({"status": "skip", "reason": "all checkpoints already in csv"}))
        return

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0

    device_map = f"cuda:{args.gpu}"
    teacher = ScapeHFToolOPD(
        model_path=args.teacher_path,
        device_map=device_map,
        use_lora=False,
    )

    rows_written = []
    for item in items:
        ck_path = item["checkpoint_path"]
        valid_rows = load_same_state_jsonl(Path(item["valid_jsonl"]))
        loss_path = item.get("loss_path", "tool_token_kl")

        is_base = Path(ck_path).resolve() == Path(args.teacher_path).resolve()
        if is_base:
            student = teacher
        else:
            student = ScapeHFToolOPD(
                model_path=ck_path,
                device_map=device_map,
                use_lora=False,
            )

        metrics = score_with_dual_backend(
            teacher, student, valid_rows,
            loss_path=loss_path,  # type: ignore[arg-type]
            max_rows=args.max_valid,
        )
        kl_ok = (
            metrics["forward_KL"] >= NUMERIC_FLOOR
            and metrics["reverse_KL"] >= NUMERIC_FLOOR
            and metrics["JS"] >= NUMERIC_FLOOR
        )
        row = {
            "family": item["family"],
            "checkpoint_id": item["checkpoint_id"],
            "checkpoint_path": ck_path,
            "valid_jsonl": item["valid_jsonl"],
            "loss_path": loss_path,
            "forward_KL": metrics["forward_KL"],
            "reverse_KL": metrics["reverse_KL"],
            "JS": metrics["JS"],
            "signed_gap": metrics["signed_gap"],
            "tool_name_KL": metrics["tool_name_KL"],
            "arg_key_KL": metrics["arg_key_KL"],
            "arg_value_KL": metrics["arg_value_KL"],
            "legacy_div": metrics["legacy_div"],
            "n_valid": len(valid_rows[:args.max_valid] if args.max_valid else valid_rows),
            "kl_nonneg_ok": kl_ok,
        }
        rows_written.append(row)
        print(json.dumps(row))

        with out_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        if not is_base:
            del student
            gc.collect()
            __import__("torch").cuda.empty_cache()

    inventory_path = out_csv.parent / "CHECKPOINT_INVENTORY.json"
    inventory_path.write_text(
        json.dumps(build_checkpoint_inventory(args.repo), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
