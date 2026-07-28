#!/usr/bin/env python3
"""Import smoke test for the Harness-1 repository."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULES = [
    "harness.config",
    "harness.harness_config",
    "harness.graph",
    "harness.modules",
    "harness.telemetry",
    "harness.views",
    "harness.lifecycle",
    "harness.tools",
    "harness.ultra_core",
    "datagen.search_dataset",
    "training.generate_sft_data",
    "training.train_sft",
    "training.train_rl",
    "training.train_opd",
    "training.opd",
    "inference.evaluate_harness1",
    "inference.evaluate_modules",
    "inference.queue_module_ablation",
    "inference.queue_browsecomp_ablation",
    "inference.hf_inference",
    "inference.vllm_local_inference",
]


def main() -> None:
    for name in MODULES:
        importlib.import_module(name)
        print(f"ok {name}")


if __name__ == "__main__":
    main()
