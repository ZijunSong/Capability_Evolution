"""Helpers for RSI bank tests. Not collected as a test module."""

from __future__ import annotations

import json
from pathlib import Path


COMPONENT_SOURCE = '''
def create(spec):
    class Component:
        def __init__(self):
            self.component_id = spec.id
            self.calls = 0
            self._private = {}

        def on_reset(self, view, private_state):
            self.calls += 1
            return {}

        def before_model(self, view, private_state):
            self.calls += 1
            return {"teacher_guidance": "use the visible documents before searching again"}

        def after_tool(self, view, tool_result, private_state):
            self.calls += 1
            return {}

        def before_finish(self, view, proposed_action, private_state):
            return {}

        def export_private_state(self):
            return dict(self._private)

        def import_private_state(self, state):
            self._private = dict(state or {})

    return Component()
'''


def write_bank(root: Path, *, component_ids: list[str] | None = None, extra: dict | None = None) -> Path:
    ids = list(component_ids or [])
    root.mkdir(parents=True, exist_ok=True)
    components = []
    for index, cid in enumerate(ids):
        slug = cid.split(".")[-1]
        (root / "components").mkdir(exist_ok=True)
        (root / "guidance").mkdir(exist_ok=True)
        (root / "recipes").mkdir(exist_ok=True)
        (root / "components" / f"{slug}.py").write_text(COMPONENT_SOURCE, encoding="utf-8")
        (root / "guidance" / f"{slug}.md").write_text("Use only visible evidence.\n", encoding="utf-8")
        (root / "recipes" / f"{slug}.json").write_text(
            json.dumps({"component_id": cid, "route": "native_anchor", "target_tools": ["search_corpus"]}),
            encoding="utf-8",
        )
        components.append(
            {
                "id": cid,
                "kind": "view",
                "factory": f"components/{slug}.py:create",
                "depends_on": [],
                "conflicts_with": [],
                "phase": "before_model",
                "priority": 20 + index,
                "reads": ["query"],
                "writes": ["teacher_guidance"],
                "distill_route": "native_anchor",
                "target_tools": ["search_corpus"],
                "guidance": f"guidance/{slug}.md",
                "recipe": f"recipes/{slug}.json",
                "max_extra_llm_calls_per_episode": 0,
                "max_extra_tool_calls_per_episode": 0,
            }
        )
    manifest = {
        "schema_version": "trim_bank_v1",
        "bank_id": root.name,
        "legacy_flags": {},
        "active_components": ids,
        "components": components,
        "hook_order": ids,
        "caps": {"active_components": 6, "added_prompt_tokens": 1536},
    }
    if extra:
        manifest.update(extra)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
