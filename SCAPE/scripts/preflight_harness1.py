#!/usr/bin/env python3
"""Harness-1 / SCAPE preflight checks. Does not start training."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def main() -> int:
    report: dict = {"ok": True, "checks": {}, "blocked": []}

    py_ok = sys.version_info >= (3, 11)
    report["checks"]["python"] = {
        "version": sys.version,
        "ok": py_ok,
        "executable": sys.executable,
    }
    if not py_ok:
        report["ok"] = False
        report["blocked"].append("Python < 3.11")

    uv_path = shutil.which("uv")
    report["checks"]["uv"] = {"ok": bool(uv_path), "path": uv_path}
    if not uv_path:
        report["ok"] = False
        report["blocked"].append("uv not found")

    code, drv = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    report["checks"]["cuda_driver"] = {"ok": code == 0, "detail": drv[:500]}
    if code != 0:
        report["blocked"].append("nvidia-smi failed")

    for pkg in ("torch", "vllm", "transformers"):
        try:
            mod = __import__(pkg)
            report["checks"][pkg] = {"ok": True, "version": getattr(mod, "__version__", "?")}
        except Exception as exc:  # noqa: BLE001
            report["checks"][pkg] = {"ok": False, "error": str(exc)}
            # vllm optional at code-bootstrap time
            if pkg != "vllm":
                report["blocked"].append(f"{pkg} missing")

    harness = REPO / "external" / "harness-1"
    report["checks"]["harness1_checkout"] = {
        "ok": harness.exists(),
        "path": str(harness),
    }
    if harness.exists():
        code, head = _run(["git", "-C", str(harness), "rev-parse", "HEAD"])
        report["checks"]["harness1_commit"] = {"ok": code == 0, "head": head}
    else:
        report["ok"] = False
        report["blocked"].append("external/harness-1 missing")

    # Retrieval backend: do not silently fall back to SCOPE BM25
    chroma = os.environ.get("SCAPE_CHROMA_PATH") or os.environ.get("HARNESS1_CHROMA_PATH")
    report["checks"]["retrieval_backend"] = {
        "ok": bool(chroma and Path(chroma).exists()),
        "path": chroma,
    }
    if not (chroma and Path(chroma).exists()):
        report["blocked"].append("retrieval backend missing")
        blocked_doc = REPO / "docs" / "BLOCKED_RETRIEVAL_BACKEND.md"
        blocked_doc.parent.mkdir(parents=True, exist_ok=True)
        if not blocked_doc.exists():
            blocked_doc.write_text(
                "# BLOCKED_RETRIEVAL_BACKEND\n\n"
                "Compatible Chroma retrieval backend / index is not available on this host.\n"
                "Do NOT substitute SCOPE BM25 for H100 contribution runs.\n"
                "Allowed without backend: model/harness smoke, instrumentation, static tests.\n",
                encoding="utf-8",
            )

    report["platform"] = platform.platform()
    report["ok"] = len(report["blocked"]) == 0 or (
        # code-only bootstrap may lack GPU/vllm/retrieval
        set(report["blocked"]) <= {"nvidia-smi failed", "vllm missing", "retrieval backend missing", "torch missing"}
    )
    # For repository bootstrap we treat missing heavy deps as WARN, not hard fail,
    # unless harness checkout or python is broken.
    hard = [b for b in report["blocked"] if b.startswith("Python") or "harness-1" in b or b == "uv not found"]
    report["hard_fail"] = bool(hard)
    report["ok"] = not report["hard_fail"]

    out = REPO / "outputs" / "preflight" / "PREFLIGHT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["hard_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
