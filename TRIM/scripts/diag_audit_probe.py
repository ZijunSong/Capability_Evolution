#!/usr/bin/env python3
"""Diagnostic: reproduce the RUNTIME_EFFECT_AUDIT 4-probe teacher failure in-container.

Prints intermediate state of _ProbeSearcher / rank_docs / _search so we can see
exactly where hits go empty (which causes auto_populate_first_search /
evidence_graph / sentence_compress / content_dedup to not fire).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TRIM = str(Path("/mnt/songzijun/Capability_Evolution/TRIM"))
for p in ("/mnt/songzijun/Capability_Evolution", _TRIM, "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD"):
    if p not in sys.path:
        sys.path.insert(0, p)

from trim.cli.launch import parse_train_args  # noqa: E402
from trim.eval.runtime_effect_audit import (  # noqa: E402
    _ProbeSearcher,
    _PROBE_STORE,
    audit_train_runtime_or_raise,
)
import trim.eval.runtime_effect_audit as _audit  # noqa: E402
from trim.eval.local_search_env import rank_docs, new_state  # noqa: E402


def main() -> int:
    sys.argv = [
        "run_train",
        "--harness", "Harness-1",
        "--benchmark", "bcplus_full",
        "--model_name", "/mnt/songzijun/models/Qwen3-4B-Instruct-2507",
        "--train_method", "trim",
        "--component", "all",
        "--train-env", "local_legacy",
        "--train-data", "sec",
        "--train-steps", "4",
        "--rl-data", "/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data",
        "--sec-corpus-root", "/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus",
        "--out", "/tmp/diag_out",
    ]
    args, _spec = parse_train_args(None)
    print("ARGS component=", getattr(args, "component", None), "training_mode=", getattr(args, "training_mode", None), flush=True)

    full_mask = {c: True for c in [
        "auto_populate_first_search", "evidence_graph", "sentence_compress",
        "content_dedup", "subtractive_curation", "importance_tagging",
        "chunk_neighbors", "verify_tool", "token_budget_marker",
        "adaptive_rerank_instruction",
    ]}

    print("STEP1 rank_docs direct:", rank_docs("Alice Smith Paris 2019", _PROBE_STORE, k=8), flush=True)
    ps = _ProbeSearcher()
    hits = ps.search("Alice Smith Paris 2019", 8)
    print("STEP2 _ProbeSearcher.search:", [(h.docid, h.score) for h in hits], flush=True)

    st, obs, ok = _audit._search(new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=full_mask))
    print(
        "STEP3 _search ok=", ok,
        " n_pool=", len(st.get("pool") or {}),
        " obs_len=", len(obs),
        " runtime_effects=", st.get("runtime_effects"),
        " auto_seed=", st.get("auto_seed"),
        " dedup_dropped=", st.get("dedup_dropped"),
        flush=True,
    )
    print("STEP3 obs[:400]=", repr(obs[:400]), flush=True)
    print("STEP3 pool keys=", list((st.get("pool") or {}).keys()), flush=True)

    os.makedirs("/tmp/diag_out", exist_ok=True)
    try:
        p = audit_train_runtime_or_raise(args, out=Path("/tmp/diag_out"))
        print("AUDIT pass=", p.get("pass"), " teacher_n_on=", p.get("teacher_n_on"), flush=True)
        t = (p.get("teacher") or {}).get("wiring", {})
        for cid in ["auto_populate_first_search", "evidence_graph", "sentence_compress", "content_dedup"]:
            print("  ", cid, "->", t["components"][cid]["full_or_current"], flush=True)
    except RuntimeError as e:
        print("AUDIT_RAISED:", e, flush=True)
    print("DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
