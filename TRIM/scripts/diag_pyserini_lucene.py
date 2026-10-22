#!/usr/bin/env python3
"""Diagnostic + self-fix: locate the Anserini fatjar shipped with Pyserini,
set ANSERINI_CLASSPATH to the jar *directory* (not the package root), force
``pyserini.pyclass.configure_classpath`` to reload it, then open the local SEC
Lucene BM25 index.

Root cause being fixed: setting ``ANSERINI_CLASSPATH=<pyserini pkg root>``
points the JVM at a directory with no jars, so
``autoclass('io.anserini.eval.Qrels')`` raises ``ClassNotFoundException``.
The correct value is ``<pyserini pkg root>/resources/jars`` (where
``anserini-2.0.0-fatjar.jar`` lives). This script discovers that path
robustly so we never depend on a hard-coded install location.
"""
from __future__ import annotations

import glob
import os
import traceback


def _resolve_jar_dir() -> str:
    if os.environ.get("ANSERINI_CLASSPATH"):
        cp = os.environ["ANSERINI_CLASSPATH"]
        if glob.glob(os.path.join(cp, "*.jar")):
            return cp
        # If someone pointed at the package root, descend to resources/jars.
        cand = os.path.join(cp, "resources", "jars")
        if glob.glob(os.path.join(cand, "*.jar")):
            return cand
    try:
        import pyserini

        root = os.path.dirname(pyserini.__file__)
        cand = os.path.join(root, "resources", "jars")
        if glob.glob(os.path.join(cand, "*.jar")):
            return cand
    except Exception:
        pass
    return ""


def main() -> int:
    jar_dir = _resolve_jar_dir()
    print(f"[diag] resolved jar_dir={jar_dir}", flush=True)
    if not jar_dir:
        print("[diag] FATAL: no anserini jar directory found", flush=True)
        return 2
    jars = glob.glob(os.path.join(jar_dir, "*.jar"))
    print(f"[diag] jars: {jars}", flush=True)

    os.environ["ANSERINI_CLASSPATH"] = jar_dir
    try:
        import pyserini.pyclass as pyclass

        # Force reconfiguration with the correct classpath. configure_classpath
        # reads ANSERINI_CLASSPATH at call time.
        pyclass.configure_classpath(jar_dir)
        print("[diag] configure_classpath ok", flush=True)
    except Exception:
        print("[diag] configure_classpath failed:", flush=True)
        traceback.print_exc()

    # pyserini.search.lucene imports pyserini.encode._openai, which constructs
    # openai.OpenAI() at module load. Fill a placeholder OPENAI_API_KEY first
    # (mirrors trim.eval.offline_credentials.ensure_local_offline_credentials).
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "offline-placeholder"
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:0/v1"
    print("[diag] OPENAI_API_KEY placeholder set", flush=True)

    idx = "/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25"
    print(f"[diag] index_dir={idx} is_dir={os.path.isdir(idx)}", flush=True)
    try:
        from pyserini.search.lucene import LuceneSearcher

        print("[diag] LuceneSearcher import ok", flush=True)
        s = LuceneSearcher(idx)
        n_docs = getattr(s, "num_docs", 0)
        if callable(n_docs):
            n_docs = n_docs()
        print(f"[diag] LuceneSearcher opened; num_docs={n_docs}", flush=True)
        hits = s.search("annual report", 3)
        print(f"[diag] probe search hits: {len(hits)}", flush=True)
        for h in hits[:3]:
            print(f"[diag]   hit docid={getattr(h, 'docid', h)}", flush=True)
        print("[diag] PYPSERINI_LUCENE_OK", flush=True)
        return 0
    except Exception:
        print("[diag] PYPSERINI_LUCENE_EXCEPTION_BELOW", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
