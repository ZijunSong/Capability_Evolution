"""Harness-G component taxonomy: basic runtime tools vs advanced (v8d-like) flags.

Harness-1 keeps search / read / curate / end_search as the always-on runtime,
and toggles V8D_* features (evidence graph, verify, auto-populate, …) as
internalization targets. Harness-G is the same split on a graph menu:

  thin runtime  = INIT + SELECT + LOOKUP + ANSWER + working memory
  full harness  = thin runtime + the advanced components below

Student rollouts always use the thin menu. Teacher DualView / OPD projection
turn the advanced components back into SELECT / LOOKUP / ANSWER.
"""

from __future__ import annotations

from typing import Any

HARNESS_NAME = "Harness-G"

# Always-on student tools. Analogous to Harness-1 search/read/curate/end_search.
RUNTIME_TOOLS: tuple[str, ...] = ("init", "select", "lookup", "answer")

# Teacher-only extra tool. Analogous to Harness-1 verify.
TEACHER_ONLY_TOOLS: tuple[str, ...] = ("answer_with",)

# Upstream-style env flags applied by TRIM around the Harness-G adapter env.
COMPONENT_TAXONOMY: dict[str, dict[str, Any]] = {
    "answer_with": {
        "upstream_flag": "HARNESS_G_ANSWER_WITH",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "Harvest SELECT+ANSWER in one menu entry. Project to select(sid).",
    },
    "bridge_entities": {
        "upstream_flag": "HARNESS_G_BRIDGE_ENTITIES",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "Extra LOOKUP targets after SELECT (graph bridge proposal).",
    },
    "entity_synonyms": {
        "upstream_flag": "HARNESS_G_ENTITY_SYNONYMS",
        "semantic_or_runtime": "hybrid",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "LOOKUP expands via Syn(e). Analogous to Harness-1 chunk_neighbors.",
    },
    "sentence_neighbors": {
        "upstream_flag": "HARNESS_G_SENTENCE_NEIGHBORS",
        "semantic_or_runtime": "runtime",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "LOOKUP expands adjacent sentences. Analogous to chunk_neighbors.",
    },
    "hybrid_init_retrieve": {
        "upstream_flag": "HARNESS_G_HYBRID_INIT_RETRIEVE",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "INIT fuses paragraph / sentence / entity channels.",
    },
    "invalid_target_filter": {
        "upstream_flag": "HARNESS_G_INVALID_TARGET_FILTER",
        "semantic_or_runtime": "runtime",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": True,
        "note": "Drop dates / cardinals / nationality LOOKUP targets.",
    },
    "lookup_dedup": {
        "upstream_flag": "HARNESS_G_LOOKUP_DEDUP",
        "semantic_or_runtime": "runtime",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": True,
        "note": "Never re-offer an already-looked-up entity.",
    },
    "snc_frontier": {
        "upstream_flag": "HARNESS_G_SNC_FRONTIER",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": False,
        "note": "Privileged same-state action previews for teacher DualView.",
    },
}

COMPONENT_ALIASES: dict[str, str] = {
    "answer_with": "answer_with",
    "harvest": "answer_with",
    "answer-with": "answer_with",
    "bridge": "bridge_entities",
    "bridge_entities": "bridge_entities",
    "bridge-entities": "bridge_entities",
    "entity_synonyms": "entity_synonyms",
    "synonyms": "entity_synonyms",
    "synonym": "entity_synonyms",
    "sentence_neighbors": "sentence_neighbors",
    "neighbors": "sentence_neighbors",
    "adjacency": "sentence_neighbors",
    "hybrid_init_retrieve": "hybrid_init_retrieve",
    "hybrid_init": "hybrid_init_retrieve",
    "init_retrieve": "hybrid_init_retrieve",
    "invalid_target_filter": "invalid_target_filter",
    "bad_target": "invalid_target_filter",
    "lookup_dedup": "lookup_dedup",
    "dedup": "lookup_dedup",
    "snc": "snc_frontier",
    "snc_frontier": "snc_frontier",
    "frontier": "snc_frontier",
}

RUNTIME_ANCHORS: frozenset[str] = frozenset(
    {
        "graph_index",
        "menu_environment",
        "mixquery_construction",
        "working_memory",
        "invalid_target_filter",
        "lookup_dedup",
    }
)
