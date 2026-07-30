#!/usr/bin/env python3
"""Round 5 B1 — DecisionState observability audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope.compact_target import compact_target_from_sample
from training.scope.dup_diagnostics import load_jsonl, write_json
from training.scope_round5.effective_input import dump_effective_inputs
from transformers import AutoTokenizer


def _gold_op(sample: dict) -> str:
    ct = compact_target_from_sample(sample)
    return ct.operation.value if ct else ""


def collision_audit(records: list) -> dict:
    by_hash: dict[str, list] = defaultdict(list)
    for rec in records:
        by_hash[rec.prompt_sha256].append(rec)

    n_conflicts = 0
    n_keep_conf = n_skip_conf = 0
    conflict_examples: list[dict] = []

    for sha, group in by_hash.items():
        labels = {r.gold_operation for r in group}
        if len(labels) > 1:
            n_conflicts += 1
            for r in group:
                if r.gold_operation == DupOperation.KEEP_EVIDENCE.value:
                    n_keep_conf += 1
                elif r.gold_operation == DupOperation.SKIP_DUPLICATE.value:
                    n_skip_conf += 1
            conflict_examples.append({
                "prompt_sha256": sha,
                "labels": sorted(labels),
                "sample_ids": [r.sample_id for r in group],
            })

    near_dup_conflicts: list[dict] = []
    by_key: dict[tuple, set] = defaultdict(set)
    for rec in records:
        ds = rec.raw_decision_state
        cand = str((ds.get("candidate_evidence_ids") or [""])[0] if isinstance(
            ds.get("candidate_evidence_ids"), list
        ) else "")
        tgt = compact_target_from_sample({"target_action": {"operation": rec.gold_operation}})
        key = (rec.query_id, cand, tuple(sorted(ds.get("observation_ids") or [])))
        by_key[key].add(rec.gold_operation)
    for key, labels in by_key.items():
        if len(labels) > 1:
            near_dup_conflicts.append({"key": key, "labels": sorted(labels)})

    return {
        "n_unique_effective_inputs": len(by_hash),
        "n_exact_collision_groups": sum(1 for g in by_hash.values() if len(g) > 1),
        "n_conflicting_label_groups": n_conflicts,
        "n_KEEP_in_conflicts": n_keep_conf,
        "n_SKIP_in_conflicts": n_skip_conf,
        "conflict_examples": conflict_examples[:20],
        "near_dup_conflicts": near_dup_conflicts[:20],
    }


def shadow_replay_audit(samples: list[dict]) -> dict:
    agree = disagree = 0
    disagreements: list[dict] = []
    for s in samples:
        ct = compact_target_from_sample(s)
        if ct is None:
            continue
        replay_op, prov = DupBilateralShadow.from_serialized_student_state(s)
        if replay_op == ct.operation:
            agree += 1
        else:
            disagree += 1
            if len(disagreements) < 20:
                disagreements.append({
                    "sample_id": s.get("sample_id"),
                    "gold": ct.operation.value,
                    "replay": replay_op.value,
                    "provenance": prov,
                })
    total = agree + disagree
    rate = agree / max(total, 1)
    return {
        "n_total": total,
        "n_agree": agree,
        "n_disagree": disagree,
        "agreement_rate": rate,
        "agreement_pass": rate >= 0.99,
        "disagreements": disagreements,
    }


def truncation_audit(records: list) -> dict:
    keep_total = skip_total = 0
    keep_trunc = skip_trunc = 0
    for rec in records:
        if rec.gold_operation == DupOperation.KEEP_EVIDENCE.value:
            keep_total += 1
            keep_trunc += int(rec.truncated)
        elif rec.gold_operation == DupOperation.SKIP_DUPLICATE.value:
            skip_total += 1
            skip_trunc += int(rec.truncated)
    return {
        "KEEP_truncation_rate": keep_trunc / max(keep_total, 1),
        "SKIP_truncation_rate": skip_trunc / max(skip_total, 1),
        "n_KEEP_truncated": keep_trunc,
        "n_SKIP_truncated": skip_trunc,
        "required_evidence_truncation_rate": sum(int(r.truncated) for r in records) / max(len(records), 1),
    }


def write_collision_report(path: Path, collision: dict, shadow: dict, trunc: dict) -> None:
  lines = [
      "# Round 5 Label Collision & Observability Report",
      "",
      "## B1.2 Effective-input label collisions",
      "",
      f"- n_unique_effective_inputs: {collision['n_unique_effective_inputs']}",
      f"- n_exact_collision_groups: {collision['n_exact_collision_groups']}",
      f"- n_conflicting_label_groups: {collision['n_conflicting_label_groups']}",
      f"- n_KEEP_in_conflicts: {collision['n_KEEP_in_conflicts']}",
      f"- n_SKIP_in_conflicts: {collision['n_SKIP_in_conflicts']}",
      "",
      "## B1.4 Serialized-state shadow replay",
      "",
      f"- agreement_rate: {shadow['agreement_rate']:.4f}",
      f"- agreement_pass (>=99%): {shadow['agreement_pass']}",
      f"- n_disagree: {shadow['n_disagree']}",
      "",
      "## B1.5 Truncation audit",
      "",
      f"- KEEP truncation rate: {trunc['KEEP_truncation_rate']:.4f}",
      f"- SKIP truncation rate: {trunc['SKIP_truncation_rate']:.4f}",
      f"- overall truncation rate: {trunc['required_evidence_truncation_rate']:.4f}",
      "",
  ]
  if collision["conflict_examples"]:
      lines.append("## Conflicting examples")
      lines.append("")
      for ex in collision["conflict_examples"]:
          lines.append(f"- sha={ex['prompt_sha256'][:16]}... labels={ex['labels']} ids={ex['sample_ids']}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs/scope_round5/observability",
    )
    args = p.parse_args()

    datasets = {
        "overfit128": _REPO / "artifacts/datasets/dup_sdi_round4_overfit128/train.jsonl",
        "train1807": _REPO / "artifacts/datasets/dup_sdi_round3/train.jsonl",
        "valid522": _REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl",
    }

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    all_records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, path in datasets.items():
        rows = load_jsonl(path)
        out = args.output_dir / f"effective_inputs_{name}.jsonl"
        recs = dump_effective_inputs(rows, tokenizer, out, max_length=args.max_length)
        all_records.extend(recs)

    combined_out = args.output_dir / "effective_inputs.jsonl"
    with combined_out.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    all_samples = []
    for path in datasets.values():
        all_samples.extend(load_jsonl(path))

    collision = collision_audit(all_records)
    shadow = shadow_replay_audit(all_samples)
    trunc = truncation_audit(all_records)

    b1_pass = (
        collision["n_conflicting_label_groups"] == 0
        and shadow["agreement_pass"]
        and trunc["required_evidence_truncation_rate"] < 0.01
    )

    report = {
        "collision": collision,
        "shadow_replay": shadow,
        "truncation": trunc,
        "B1_PASS": b1_pass,
    }
    write_json(args.output_dir / "observability_report.json", report)
    write_collision_report(
        args.output_dir / "LABEL_COLLISION_REPORT.md", collision, shadow, trunc,
    )
    (args.output_dir.parent / "B1_PASS").write_text(str(b1_pass) + "\n")
    print(json.dumps({"B1_PASS": b1_pass, **{k: report[k] for k in ("collision", "shadow_replay", "truncation")}}, indent=2))


if __name__ == "__main__":
    main()
