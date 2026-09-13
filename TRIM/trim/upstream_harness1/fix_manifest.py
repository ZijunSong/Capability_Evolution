"""Public fix/version stamps for Harness-1 BM25 audit repairs (2026-09-13)."""

from __future__ import annotations

PUBLIC_FIX_VERSION = "harness1-bm25-fix-20260913"
METRIC_VERSION = "harness1-metrics-v2"
COUNT_VERSION = "harness1-count-v2"
DEFAULT_CURATE_NUDGE_POLICY = "legacy"


def fix_manifest() -> dict[str, str]:
    return {
        "public_fix_version": PUBLIC_FIX_VERSION,
        "metric_version": METRIC_VERSION,
        "count_version": COUNT_VERSION,
        "default_curate_nudge_policy": DEFAULT_CURATE_NUDGE_POLICY,
    }
