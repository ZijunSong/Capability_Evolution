#!/usr/bin/env python3
"""Check official Harness-1 environment variables without printing secret values."""

from __future__ import annotations

import json
import os
from pathlib import Path

REQUIRED = [
    "OPENAI_API_KEY",
    "CHROMA_API_KEY",
    "CHROMA_DATABASE",
    "BROWSECOMPPLUS_QUERIES_PATH",
    "BROWSECOMPPLUS_QRELS_GOLD_PATH",
    "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH",
    "BROWSECOMPPLUS_ANSWERS_PATH",
]
OPTIONAL = [
    "HUGGINGFACE_TOKEN",
    "BASETEN_API_KEY",
    "BASETEN_MODEL_URL",
    "ANTHROPIC_API_KEY",
    "TINKER_API_KEY",
    "MOONSHOT_API_KEY",
    "JINA_API_KEY",
    "CONTEXTUAL_API_KEY",
]
PATH_VARS = [
    "BROWSECOMPPLUS_QUERIES_PATH",
    "BROWSECOMPPLUS_QRELS_GOLD_PATH",
    "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH",
    "BROWSECOMPPLUS_ANSWERS_PATH",
]


def present(name: str) -> bool:
    return bool(os.environ.get(name))


def main() -> None:
    checks = {name: present(name) for name in REQUIRED + OPTIONAL}
    path_checks = {name: (Path(os.environ[name]).exists() if present(name) else False) for name in PATH_VARS}
    out = {
        "required_present": {name: checks[name] for name in REQUIRED},
        "optional_present": {name: checks[name] for name in OPTIONAL},
        "path_exists": path_checks,
        "missing_required": [name for name in REQUIRED if not checks[name]],
        "missing_or_bad_paths": [name for name in PATH_VARS if not path_checks[name]],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
