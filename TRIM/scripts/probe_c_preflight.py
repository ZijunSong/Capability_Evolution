#!/usr/bin/env python3
"""Probe C preflight: loader/registry assertions before any GPU spend."""

from __future__ import annotations

import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))


def main() -> int:
    import importlib.metadata

    import torch
    import transformers
    import vllm
    from jnius import autoclass

    from trim.probes.probe_c_collect import model_identity
    from trim.training.four_cell_runtime import TEACHER_REGISTRY

    autoclass("java.lang.String")
    print(
        "VERSIONS",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "peft": importlib.metadata.version("peft"),
            "pyserini": importlib.metadata.version("pyserini"),
        },
    )
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    assert ver >= (5, 6), (
        f"transformers {transformers.__version__} lacks qwen3_5 registration"
    )
    ident = model_identity(Path("/tmp/qwen35-9b"))
    print("ARCH_OK", ident["architectures"], ident["model_revision"])
    reg = sorted(TEACHER_REGISTRY)
    print("TEACHER_REGISTRY", reg)
    for need in (
        "adaptive_rerank_instruction",
        "sentence_compress",
        "auto_populate_first_search",
        "verify_tool",
        "token_budget_marker",
    ):
        assert need in reg, f"missing registered teacher: {need}"
    print("IMPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
