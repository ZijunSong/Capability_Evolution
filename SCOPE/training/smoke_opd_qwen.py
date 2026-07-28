#!/usr/bin/env python3
"""Deprecated smoke entry — redirects to vLLM rollout + HF train smoke test."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.warn(
    "smoke_opd_qwen.py uses deprecated HF generate() rollout. "
    "Use training/smoke_opd_vllm_hf.py (vLLM rollout + HF train) instead.",
    DeprecationWarning,
    stacklevel=1,
)

from training.smoke_opd_vllm_hf import main

if __name__ == "__main__":
    main()
