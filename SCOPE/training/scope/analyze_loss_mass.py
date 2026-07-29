#!/usr/bin/env python3
"""Round 1 loss-mass audit: sample balance vs target-token / loss mass."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from training.scope.dup_diagnostics import action_bucket, load_jsonl


def _operation_bucket(action: dict[str, Any] | None) -> str:
    if not action:
        return "other"
    cap = CapabilityAction.from_dict(action)
    at = cap.action_type.value
    if at == "curate_document":
        return action_bucket(action)
    if at in {"continue_search", "search", "rewrite_query"}:
        return "continue"
    if at in {"stop_and_answer", "answer", "abstain"}:
        return "stop"
    return at


def _count_target_tokens(sample: dict[str, Any], tokenizer=None) -> int:
    text = str(sample.get("target_action_text") or "")
    if not text and sample.get("target_action"):
        from harness.capability.adapters import render_capability_action

        text = render_capability_action(
            CapabilityAction.from_dict(sample["target_action"])
        )
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def _group_key(sample: dict[str, Any], *, by_route: bool, by_operation: bool) -> str:
    parts: list[str] = []
    if by_route:
        parts.append(str(sample.get("route", "UNKNOWN")).upper())
    if by_operation:
        parts.append(_operation_bucket(sample.get("target_action")))
    return "|".join(parts) if parts else "all"


def _aggregate_stats(
    samples: list[dict[str, Any]],
    tokenizer=None,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        route = str(s.get("route", "UNKNOWN")).upper()
        op = _operation_bucket(s.get("target_action"))
        tok = _count_target_tokens(s, tokenizer)
        groups[f"route:{route}"].append(tok)
        groups[f"operation:{op}"].append(tok)
        groups[f"route:{route}|operation:{op}"].append(tok)

    total_tokens = sum(sum(v) for v in groups.values() if "|" in next(iter(groups), ""))
    # Recompute total from per-sample
    all_tokens = [_count_target_tokens(s, tokenizer) for s in samples]
    total_tokens = sum(all_tokens)

    def _stats(vals: list[int]) -> dict[str, Any]:
        if not vals:
            return {
                "n_samples": 0,
                "total_target_tokens": 0,
                "mean_target_tokens": 0.0,
                "median_target_tokens": 0.0,
                "p90_target_tokens": 0.0,
                "token_share": 0.0,
            }
        sv = sorted(vals)
        n = len(sv)
        p90_idx = min(n - 1, int(n * 0.9))
        return {
            "n_samples": n,
            "total_target_tokens": sum(sv),
            "mean_target_tokens": sum(sv) / n,
            "median_target_tokens": sv[n // 2],
            "p90_target_tokens": sv[p90_idx],
            "token_share": sum(sv) / max(total_tokens, 1),
        }

    by_route: dict[str, Any] = {}
    by_operation: dict[str, Any] = {}
    by_route_operation: dict[str, Any] = {}
    for key, vals in groups.items():
        st = _stats(vals)
        if key.startswith("route:") and "|" not in key:
            by_route[key.split(":", 1)[1]] = st
        elif key.startswith("operation:"):
            by_operation[key.split(":", 1)[1]] = st
        elif "|" in key:
            by_route_operation[key] = st

    n = len(samples)
    endorse_n = sum(1 for s in samples if str(s.get("route", "")).upper() == "ENDORSE")
    correct_n = sum(1 for s in samples if str(s.get("route", "")).upper() == "CORRECT")
    endorse_tok = sum(
        _count_target_tokens(s, tokenizer)
        for s in samples
        if str(s.get("route", "")).upper() == "ENDORSE"
    )
    correct_tok = sum(
        _count_target_tokens(s, tokenizer)
        for s in samples
        if str(s.get("route", "")).upper() == "CORRECT"
    )

    return {
        "n_samples": n,
        "total_target_tokens": total_tokens,
        "by_route": by_route,
        "by_operation": by_operation,
        "by_route_operation": by_route_operation,
        "endorse_sample_share": endorse_n / max(n, 1),
        "endorse_target_token_share": endorse_tok / max(total_tokens, 1),
        "correct_sample_share": correct_n / max(n, 1),
        "correct_target_token_share": correct_tok / max(total_tokens, 1),
    }


def _loss_mass_with_ce(
    samples: list[dict[str, Any]],
    model_path: str | None,
) -> dict[str, Any] | None:
    if not model_path:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from training.scope.collator import collate_sdi_batch
        from training.scope.losses import sdi_cross_entropy
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    per_group_nll: dict[str, float] = defaultdict(float)
    per_group_tok: dict[str, int] = defaultdict(int)
    per_group_n: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for s in samples:
            try:
                batch = collate_sdi_batch([s], tokenizer, max_length=4096)
            except ValueError:
                continue
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            active = shift_labels != -100
            n_tok = int(active.sum().item())
            if n_tok == 0:
                continue
            loss, _ = sdi_cross_entropy(logits, labels)
            route = str(s.get("route", "UNKNOWN")).upper()
            op = _operation_bucket(s.get("target_action"))
            key = f"{route}|{op}"
            per_group_nll[key] += float(loss.item()) * n_tok
            per_group_tok[key] += n_tok
            per_group_n[key] += 1

    total_nll = sum(per_group_nll.values())
    groups: dict[str, Any] = {}
    for key in per_group_nll:
        groups[key] = {
            "n_samples": per_group_n[key],
            "sum_nll": per_group_nll[key],
            "mean_nll_per_token": per_group_nll[key] / max(per_group_tok[key], 1),
            "mean_nll_per_sample": per_group_nll[key] / max(per_group_n[key], 1),
            "estimated_loss_share": per_group_nll[key] / max(total_nll, 1e-8),
        }
    return {"by_route_operation": groups, "total_nll": total_nll}


def _imbalance_check(stats: dict[str, Any]) -> dict[str, Any]:
    es = stats["endorse_sample_share"]
    et = stats["endorse_target_token_share"]
    cs = stats["correct_sample_share"]
    ct = stats["correct_target_token_share"]
    sample_balanced = abs(es - cs) < 0.15
    token_imbalanced = abs(et - ct) > 0.15 or max(et, ct) > 0.65
    curate_ops = stats.get("by_operation", {})
    curate_share = curate_ops.get("curate_add", {}).get("token_share", 0)
    curate_share += curate_ops.get("curate_replace", {}).get("token_share", 0)
    return {
        "sample_near_50_50": sample_balanced,
        "token_mass_skewed": token_imbalanced,
        "long_curate_dominates_loss": curate_share > 0.5,
        "endorse_sample_share": es,
        "endorse_token_share": et,
        "correct_sample_share": cs,
        "correct_token_share": ct,
        "diagnosis": (
            "sample balance ≠ loss-token balance detected"
            if sample_balanced and token_imbalanced
            else "no severe imbalance"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Round 1 Loss-Mass Audit\n"]
    lines.append(f"- n_samples: **{report['n_samples']}**")
    lines.append(f"- total_target_tokens: **{report['total_target_tokens']}**\n")
    imb = report.get("imbalance_check", {})
    lines.append("## Route balance vs token mass\n")
    lines.append(
        f"| Route | Sample share | Target-token share |\n"
        f"|-------|-------------|-------------------|\n"
        f"| ENDORSE | {imb.get('endorse_sample_share', 0):.3f} | "
        f"{imb.get('endorse_token_share', 0):.3f} |\n"
        f"| CORRECT | {imb.get('correct_sample_share', 0):.3f} | "
        f"{imb.get('correct_token_share', 0):.3f} |\n"
    )
    lines.append(f"\n**Diagnosis:** {imb.get('diagnosis', 'n/a')}\n")
    lines.append("## By operation\n")
    for op, st in sorted((report.get("by_operation") or {}).items()):
        lines.append(
            f"- **{op}**: n={st['n_samples']}, "
            f"mean_tok={st['mean_target_tokens']:.1f}, "
            f"token_share={st['token_share']:.3f}"
        )
    if report.get("ce_loss_mass"):
        lines.append("\n## CE loss mass (optional)\n")
        for key, st in sorted(report["ce_loss_mass"]["by_route_operation"].items()):
            lines.append(
                f"- {key}: loss_share={st['estimated_loss_share']:.3f}, "
                f"n={st['n_samples']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/dup_sdi_round1"),
    )
    p.add_argument(
        "--natural-samples",
        type=Path,
        default=Path("outputs/scope_v3_audit_100q/natural_100q/samples.jsonl"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/scope_round2/diagnostics/round1_loss_mass.json"),
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/scope_round2/diagnostics/round1_loss_mass.md"),
    )
    p.add_argument("--model-path", type=str, default=None)
    args = p.parse_args()

    samples: list[dict[str, Any]] = []
    for path in [args.dataset_dir / "train.jsonl", args.dataset_dir / "valid.jsonl"]:
        if path.exists():
            samples.extend(load_jsonl(path))
    if args.natural_samples.exists():
        samples.extend(load_jsonl(args.natural_samples))
    # Deduplicate by sample_id
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for s in samples:
        sid = str(s.get("sample_id") or s.get("event_id") or id(s))
        if sid not in seen:
            seen.add(sid)
            unique.append(s)
    samples = unique

    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    stats = _aggregate_stats(samples, tokenizer)
    stats["imbalance_check"] = _imbalance_check(stats)
    ce = _loss_mass_with_ce(samples, args.model_path)
    if ce:
        stats["ce_loss_mass"] = ce

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(stats), encoding="utf-8")
    print(json.dumps(stats["imbalance_check"], indent=2))


if __name__ == "__main__":
    main()
