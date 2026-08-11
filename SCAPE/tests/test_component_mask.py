from __future__ import annotations

import os

from scape.adapters.components import assert_mask_diff_only, full_mask, minus_mask
from scape.adapters.harness_mask import apply_component_mask, only_toggle


def test_component_mask_only_changes_target_flag():
    base = full_mask()
    after = minus_mask("evidence_graph", base)
    assert_mask_diff_only(base, after, expected_changed=["evidence_graph"])
    assert base["evidence_graph"] is True
    assert after["evidence_graph"] is False

    # Env application only toggles the target upstream flag relative to a clean apply
    with apply_component_mask(base):
        before_env = {k: os.environ.get(k) for k in os.environ if k.startswith("V8D_")}
    with apply_component_mask(after):
        after_env = {k: os.environ.get(k) for k in os.environ if k.startswith("V8D_")}

    changed = [k for k in set(before_env) | set(after_env) if before_env.get(k) != after_env.get(k)]
    assert changed == ["V8D_EVIDENCE_GRAPH"]

    toggled = only_toggle("verify_tool", enabled=False, base=base)
    assert_mask_diff_only(base, toggled, expected_changed=["verify_tool"])
