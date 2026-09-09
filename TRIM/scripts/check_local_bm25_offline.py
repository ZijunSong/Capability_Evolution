#!/usr/bin/env python3
"""L12-1: import original tool types without chromadb / Chroma credentials."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))


def main() -> int:
    os.environ["HARNESS1_FORBID_CHROMA"] = "1"
    os.environ.pop("CHROMA_API_KEY", None)
    os.environ.pop("CHROMA_DATABASE", None)
    from trim.upstream_harness1.pin import ensure_harness1_on_path
    from trim.upstream_harness1.retrieval import RetrievalConfig

    ensure_harness1_on_path()
    if "chromadb" in sys.modules:
        raise SystemExit("chromadb was already imported before the offline check")
    import harness.config  # noqa: F401
    import harness.tools  # noqa: F401

    if "chromadb" in sys.modules:
        raise SystemExit("importing harness.config/tools loaded chromadb")
    try:
        RetrievalConfig(backend="local_bm25").assert_ready()
    except RuntimeError as exc:
        if "index-path" not in str(exc):
            raise
    from harness.config import Config

    cfg = Config()  # type: ignore[call-arg]
    try:
        cfg.get_chroma_client()
    except RuntimeError as exc:
        if "HARNESS1_FORBID_CHROMA" not in str(exc):
            raise
    else:
        raise SystemExit("get_chroma_client must refuse local_bm25 process")
    print("ok: harness.tools importable without chromadb; chroma client forbidden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
