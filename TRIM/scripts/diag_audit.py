#!/usr/bin/env python3
"""Audit reproducer that mirrors the training job's exact audit call path.

Run with the SAME PYTHONPATH as the training job. If the training pip install
changed audit behavior, this will show it.
"""
import sys, json, traceback, types

print("=== PYTHON ===", sys.executable, sys.version, flush=True)
print("=== sys.path ===", sys.path[:4], flush=True)

# Show where trim comes from (catch a shadowing package from pip install)
try:
    import trim
    print("trim.__file__ =", getattr(trim, "__file__", None), flush=True)
    print("trim.__path__ =", list(getattr(trim, "__path__", []) or []), flush=True)
except Exception as e:
    print("trim import FAIL", repr(e), flush=True)

try:
    import trim.eval.runtime_effect_audit as ra
    print("runtime_effect_audit.__file__ =", ra.__file__, flush=True)
    import trim.eval.local_search_env as lse
    print("local_search_env.__file__ =", lse.__file__, flush=True)
    import trim.eval.h1_component_runtime as h1
    print("h1_component_runtime.__file__ =", h1.__file__, flush=True)
except Exception as e:
    print("AUDIT CHAIN IMPORT FAIL", repr(e), flush=True)
    traceback.print_exc()
    sys.exit(2)

from trim.eval.runtime_effect_audit import (
    _PROBE_STORE, _ProbeSearcher, _search, _probe_auto_populate,
    _probe_evidence_graph, _probe_sentence_compress, _probe_content_dedup,
    audit_mask_wiring, audit_train_runtime_or_raise,
)
from trim.eval.local_search_env import execute_tool, new_state, rank_docs
from trim.eval.h1_component_runtime import filter_dedup_hits, mask_on, resolve_mask, empty_effects

# Mirror the training job's mask resolution exactly.
from trim.training.four_cell_runtime import student_mask_for, teacher_mask_for
tm = teacher_mask_for(["all"], harness="Harness-1")
sm = student_mask_for(["all"], harness="Harness-1")
print("=== teacher_mask_for(['all']) ===", json.dumps(tm), flush=True)
print("=== student_mask_for(['all']) ===", json.dumps(sm), flush=True)

print("=== rank_docs direct ===", flush=True)
r = rank_docs("Alice Smith Paris 2019", _PROBE_STORE, k=8)
print("rank_docs:", [(d, round(s, 3)) for d, _t, s in r], flush=True)

s = _ProbeSearcher()
print("searcher.name =", repr(getattr(s, "name", "NONE")), flush=True)
hits = s.search("Alice Smith Paris 2019", 8)
print("n_hits =", len(hits), "docids =", [h.docid for h in hits], flush=True)

print("=== execute_tool search_corpus with TEACHER mask ===", flush=True)
st = new_state("Alice Smith Paris", dict(_PROBE_STORE), harness_mask=tm)
print("mask_on auto_populate =", mask_on(st, "auto_populate_first_search"), flush=True)
print("mask_on content_dedup =", mask_on(st, "content_dedup"), flush=True)
st2, obs, ok = execute_tool(st, "search_corpus", {"query": "Alice Smith Paris 2019"}, searcher=_ProbeSearcher())
print("ok =", ok, "| obs_len =", len(obs), flush=True)
print("obs_head =", repr(obs[:220]), flush=True)
print("pool =", list((st2.get("pool") or {}).keys()), flush=True)
print("curated =", list((st2.get("curated") or {}).keys()), flush=True)
print("auto_seed =", st2.get("auto_seed"), flush=True)
print("dedup_dropped =", st2.get("dedup_dropped"), flush=True)
print("runtime_effects =", st2.get("runtime_effects"), flush=True)

print("=== per-probe (teacher mask) ===", flush=True)
for name, fn in [("auto_populate", _probe_auto_populate),
                 ("evidence_graph", _probe_evidence_graph),
                 ("sentence_compress", _probe_sentence_compress),
                 ("content_dedup", _probe_content_dedup)]:
    try:
        rep = fn(tm)
        print(name, "=>", json.dumps(rep, default=str), flush=True)
    except Exception as e:
        print(name, "EXC", repr(e), flush=True)
        traceback.print_exc()

print("=== audit_mask_wiring(teacher) ===", flush=True)
res = audit_mask_wiring(tm)
print("pass =", res.get("pass"), "n_on =", res.get("n_on"), "n_fail =", res.get("n_fail"), flush=True)
print("failures =", res.get("failures"), flush=True)

print("=== audit_train_runtime_or_raise (training path) ===", flush=True)
# Build a minimal args namespace mirroring the failed job.
args = types.SimpleNamespace(
    component=["all"], harness="Harness-1", training_mode="rl",
    lambda_opd=None, opd_loss="sr_opd_ce",
)
try:
    out = audit_train_runtime_or_raise(args, out=None)
    print("RAISE did NOT fire; payload pass =", out.get("pass"), flush=True)
    print("summary =", out.get("summary"), flush=True)
except RuntimeError as e:
    print("RAISED:", str(e), flush=True)
except Exception as e:
    print("OTHER EXC", repr(e), flush=True)
    traceback.print_exc()

print("=== DONE ===", flush=True)
