#!/usr/bin/env python3
"""Transfer evaluation hooks: schema swap, reason rename, retriever/corpus shift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.artifacts.reason_codes import ALL_REASON_CODES


def perturb_schema_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Evidence schema field-order / rename perturbation for transfer tests."""
    art = dict(row.get("artifact") or {})
    # Rename keys commonly used in evidence payloads
    payload = dict(art.get("metadata") or {})
    if "supporting_document_ids" in payload:
        payload["support_docs"] = payload.pop("supporting_document_ids")
    if "claim_id" in payload:
        payload["cid"] = payload.pop("claim_id")
    art["metadata"] = payload
    # Field order change via rebuild
    reordered = {
        "module_id": art.get("module_id"),
        "reason_code": art.get("reason_code"),
        "mode": art.get("mode"),
        "evidence_ids": art.get("evidence_ids"),
        "document_ids": art.get("document_ids"),
        "metadata": art.get("metadata"),
    }
    out = dict(row)
    out["artifact"] = reordered
    out["transfer_tag"] = "schema_perturb"
    return out


def rename_reason_codes(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out = dict(row)
    code = out.get("reason_code")
    if code in mapping:
        out["reason_code"] = mapping[code]
        art = dict(out.get("artifact") or {})
        art["reason_code"] = mapping[code]
        out["artifact"] = art
    out["transfer_tag"] = "reason_rename"
    return out


DEFAULT_REASON_RENAME = {
    "PREMATURE_STOP": "EARLY_STOP",
    "MISSING_DIRECT_EVIDENCE": "NO_DIRECT_EVIDENCE",
}


def apply_transfer(
    rows: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "schema":
        return [perturb_schema_fields(r) for r in rows]
    if mode == "reason_rename":
        return [rename_reason_codes(r, DEFAULT_REASON_RENAME) for r in rows]
    if mode == "retriever":
        # Marker only — actual BM25↔Chroma swap is handled by env_factory / eval queues
        out = []
        for r in rows:
            rr = dict(r)
            rr["transfer_tag"] = "retriever_bm25_to_chroma"
            rr["retriever"] = "chroma"
            out.append(rr)
        return out
    if mode == "fresh_corpus":
        out = []
        for r in rows:
            rr = dict(r)
            rr["transfer_tag"] = "fresh_corpus"
            rr["corpus"] = "fresh"
            out.append(rr)
        return out
    raise ValueError(f"Unknown transfer mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input transitions JSONL")
    p.add_argument(
        "--mode",
        choices=["schema", "reason_rename", "retriever", "fresh_corpus"],
        default="schema",
    )
    p.add_argument("--out", type=str, default="outputs/scope_transfer/out.jsonl")
    args = p.parse_args(argv)

    rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    transferred = apply_transfer(rows, mode=args.mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in transferred:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "mode": args.mode,
        "n": len(transferred),
        "known_reason_codes": sorted(ALL_REASON_CODES),
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
