"""Fixed native-anchor interpreter. The evolved surface is the recipe and guidance, not this file."""

from __future__ import annotations

import json
from pathlib import Path


def create(spec):
    root = Path(__file__).resolve().parents[1]
    recipe_path = root / str(spec.recipe or "recipes/query_repair.json")
    recipe = {}
    if recipe_path.is_file():
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    class Component:
        def __init__(self):
            self.component_id = spec.id
            self._private = {}

        def on_reset(self, view, private_state):
            return {}

        def before_model(self, view, private_state):
            if str(recipe.get("trigger") or "") != "after_search":
                return {}
            history = list((view or {}).get("visible_search_history") or [])
            if not history:
                return {}
            text = str(recipe.get("guidance") or "").strip()
            if not text:
                return {}
            return {"teacher_guidance": text}

        def after_tool(self, view, tool_result, private_state):
            if str(recipe.get("trigger") or "") != "after_search":
                return {}
            name = str((tool_result or {}).get("name") or "")
            if name != "search_corpus":
                return {}
            query = str((view or {}).get("query") or "").strip()
            if not query:
                return {}
            return {
                "events": [
                    {
                        "component_id": spec.id,
                        "action_name": "search_corpus",
                        "arguments": {"query": query},
                        "information_provenance": [{"source": "visible_query", "available_to_student": True}],
                    }
                ]
            }

        def before_finish(self, view, proposed_action, private_state):
            return {}

        def export_private_state(self):
            return dict(self._private)

        def import_private_state(self, state):
            self._private = dict(state or {})

    return Component()
