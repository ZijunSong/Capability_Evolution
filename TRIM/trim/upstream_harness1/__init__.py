"""Boundary between TRIM and the pinned upstream Harness-1 tree.

Official eval and training adapters import original environment classes from
``TRIM/external/harness-1``. This package holds version pins, V8D flag maps,
API message conversion, and retrieval manifests. It must not reimplement
search, curation, verify, or working-memory updates.
"""

from trim.upstream_harness1.pin import (
    HARNESS1_ROOT,
    INTERFACE_PATCHES,
    PINNED_UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    load_pin,
    pin_manifest,
)
from trim.upstream_harness1.v8d_flags import (
    ABLATE_FLAGS_CLEARED_FOR_BASELINE,
    EVALUATION_PATH_LEGACY_LOCAL,
    EVALUATION_PATH_UPSTREAM_API,
    V8D_COMPONENT_FLAGS,
    actual_eval_mask,
    all_enabled_mask,
    subprocess_env_for_mask,
    v8d_env_from_mask,
)

__all__ = [
    "ABLATE_FLAGS_CLEARED_FOR_BASELINE",
    "EVALUATION_PATH_LEGACY_LOCAL",
    "EVALUATION_PATH_UPSTREAM_API",
    "HARNESS1_ROOT",
    "INTERFACE_PATCHES",
    "PINNED_UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "V8D_COMPONENT_FLAGS",
    "actual_eval_mask",
    "all_enabled_mask",
    "load_pin",
    "pin_manifest",
    "subprocess_env_for_mask",
    "v8d_env_from_mask",
]
