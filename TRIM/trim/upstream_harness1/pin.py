"""Pinned upstream Harness-1 identity (E01)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRIM_ROOT = Path(__file__).resolve().parents[2]
HARNESS1_ROOT = TRIM_ROOT / "external" / "harness-1"
PIN_PATH = TRIM_ROOT / "external" / "harness-1.PIN.json"

UPSTREAM_REPO = "https://github.com/pat-jj/harness-1"
PINNED_UPSTREAM_COMMIT = "8ac4012167858f6478fb2a8fd840e4550e2af161"
TRIM_REVIEW_COMMIT = "9c6b3a74d4c191b080254e13c0606f8fb9a55078"

INTERFACE_PATCHES: tuple[dict[str, str], ...] = (
    {
        "file": "training/train_rl.py",
        "symbol": "SlidingWindowSearchEnv.step_action",
        "kind": "io_boundary",
        "reason": "Accept a structured Action after token or API parse; keep original step lifecycle.",
    },
    {
        "file": "training/train_rl.py",
        "symbol": "SlidingWindowSearchEnv.selected_context_window",
        "kind": "io_boundary",
        "reason": "Expose the original RECENT_K / WM selection used by _render_next_context.",
    },
    {
        "file": "training/train_rl.py",
        "symbol": "SlidingWindowSearchEnv.__init__.openai_client",
        "kind": "io_boundary",
        "reason": "Inject a local verifier client so _exec_verify does not call get_config().",
    },
    {
        "file": "harness/config.py",
        "symbol": "get_chroma_client lazy import / HARNESS1_FORBID_CHROMA",
        "kind": "io_boundary",
        "reason": "Do not import chromadb or construct CloudClient unless the Chroma backend is selected.",
    },
    {
        "file": "harness/tools.py",
        "symbol": "lazy chromadb + Reranker type",
        "kind": "io_boundary",
        "reason": "Tool schema/metadata importable without chromadb or harness.rerank/get_config.",
    },
)


def load_pin() -> dict[str, Any]:
    if PIN_PATH.is_file():
        return json.loads(PIN_PATH.read_text(encoding="utf-8"))
    return {
        "upstream_repo": UPSTREAM_REPO,
        "pinned_commit": PINNED_UPSTREAM_COMMIT,
        "pin_method": "vendored_tree",
        "vendored_path": "TRIM/external/harness-1",
        "trim_review_commit": TRIM_REVIEW_COMMIT,
    }


def pin_manifest() -> dict[str, Any]:
    pin = load_pin()
    return {
        "upstream_repo": pin.get("upstream_repo") or UPSTREAM_REPO,
        "pinned_commit": pin.get("pinned_commit") or PINNED_UPSTREAM_COMMIT,
        "pin_method": pin.get("pin_method") or "vendored_tree",
        "vendored_path": str(HARNESS1_ROOT),
        "vendored_exists": HARNESS1_ROOT.is_dir(),
        "core_files": {
            "env": str(HARNESS1_ROOT / "training" / "train_rl.py"),
            "agent_api": str(HARNESS1_ROOT / "harness" / "agent.py"),
            "ultra_core": str(HARNESS1_ROOT / "harness" / "ultra_core.py"),
            "eval_tinker": str(HARNESS1_ROOT / "inference" / "evaluate_harness1.py"),
        },
        "interface_patches": [dict(p) for p in INTERFACE_PATCHES],
        "trim_review_commit": pin.get("trim_review_commit") or TRIM_REVIEW_COMMIT,
        "is_git_submodule": (HARNESS1_ROOT / ".git").exists(),
    }


def ensure_harness1_on_path() -> Path:
    import sys

    root = str(HARNESS1_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    cookbook = str(HARNESS1_ROOT / "tinker-cookbook")
    if cookbook not in sys.path:
        sys.path.insert(0, cookbook)
    return HARNESS1_ROOT
