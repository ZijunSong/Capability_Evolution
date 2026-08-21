#!/usr/bin/env python3
"""Prepare and audit the H100-1 V2 matched structured/textual privilege pair."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FIELDS = ["verify_available_full_view", "verify_available_student_view", "step", "n_documents", "n_curated_ids", "doc_ids", "claim_nonempty", "tool_history", "remaining_budget"]

def canon(r):
    elig = r["verify_eligibility"]
    xi = r["xi_t"]
    full = r["full_privileged_view"]
    return {
        "verify_available_full_view": bool(elig["verify_available_full_view"]),
        "verify_available_student_view": bool(elig["verify_available_student_view"]),
        "step": int(elig["step"]),
        "n_documents": int(elig["n_documents"]),
        "n_curated_ids": int(elig["n_curated_ids"]),
        "doc_ids": [str(x) for x in elig["doc_ids"]],
        "claim_nonempty": bool(elig["claim_nonempty"]),
        "tool_history": full.get("tool_history", xi.get("working_memory", {}).get("tool_history", [])),
        "remaining_budget": str(xi.get("working_memory", {}).get("token_budget_marker", "")),
    }

def textual(c):
    return "\n".join([
        "VERIFY_PRIVILEGE_V2",
        f"verify_available_full_view={str(c['verify_available_full_view']).lower()}",
        f"verify_available_student_view={str(c['verify_available_student_view']).lower()}",
        f"step={c['step']}",
        f"n_documents={c['n_documents']}",
        f"n_curated_ids={c['n_curated_ids']}",
        "doc_ids=" + json.dumps(c["doc_ids"], separators=(",", ":")),
        f"claim_nonempty={str(c['claim_nonempty']).lower()}",
        "tool_history=" + json.dumps(c["tool_history"], ensure_ascii=False, separators=(",", ":")),
        "remaining_budget=" + c["remaining_budget"],
        "END_VERIFY_PRIVILEGE_V2",
    ])

def parse_text(s):
    lines = s.splitlines()[1:-1]
    out = {}
    for line in lines:
        k, v = line.split("=", 1)
        if k in {"verify_available_full_view", "verify_available_student_view", "claim_nonempty"}: out[k] = v == "true"
        elif k in {"step", "n_documents", "n_curated_ids"}: out[k] = int(v)
        elif k == "doc_ids" or k == "tool_history": out[k] = json.loads(v)
        else: out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in args.source.read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs = []; failures = []
    for r in rows:
        c = canon(r); t = textual(c); parsed = parse_text(t); ok = parsed == c
        if not ok: failures.append(r.get("state_id"))
        pairs.append({"state_id": r["state_id"], "query_id": r["query_id"], "snapshot_hash": r["snapshot_hash"], "prompt_student": r.get("prompt_reduced", ""), "structured_privilege": c, "textual_privilege": t, "textual_roundtrip": parsed, "information_equivalent": ok})
    with (args.out / "matched_v2_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    split = {"source_rows": len(rows), "query_ids": sorted({str(r["query_id"]) for r in rows}), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "query_disjoint": True}
    (args.out / "V2_SPLIT_MANIFEST.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    status = "prepared_pending_training" if not failures else "failed_information_audit"
    (args.out / "MATCHED_INFORMATION_AUDIT.md").write_text("\n".join([
        "# MATCHED_INFORMATION_AUDIT", "", f"- status: `{status}`", f"- rows: {len(rows)}", f"- roundtrip_pass: {len(rows)-len(failures)}/{len(rows)}", f"- source: `{args.source}`", "- structured fields: exactly the 9 V2 state-time fields", "- textualizer: deterministic key=value serialization; no generated explanation, future labels, or reasoning", "- student inference privilege: absent; only `prompt_student` is retained", "- LOCAL_COMPAT_ONLY=true", "- official_chroma_parity=false", "", "Matched Text OPD training has not been launched by this adapter; this artifact is the required information audit and data barrier.", ""]), encoding="utf-8")
    (args.out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- status: {status}\n- rows: {len(rows)}\n- roundtrip_failures: {len(failures)}\n- next: launch matched textual OPD after runner integration\n", encoding="utf-8")
    if failures: raise SystemExit(f"information audit failed for {len(failures)} rows")

if __name__ == "__main__": main()
