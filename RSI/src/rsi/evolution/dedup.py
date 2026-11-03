"""Candidate deduplication fingerprints (audit doc §8.2).

27B evolve runs proposed the same component repeatedly across generations;
each duplicate consumed a full panel evaluation.  Three fingerprint layers:

1. exact content  — executable source bytes identical: reuse the previous
   evaluation outright (cache hit);
2. AST near-dup   — comments/docstrings/formatting stripped AST equal: flagged
   near-duplicate, allowed a targeted small-panel recheck, not a full panel;
3. behavior probe — (future) trigger/effect signatures on fixed smoke traces.

Cache keys are content-derived, NOT candidate_id: the id mixes in parent
digests and hypothesis text, so re-describing the same code produced new ids
and evaded dedup.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Mapping


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_digest(source: str) -> str:
    """Exact-content fingerprint of one component file."""
    return _sha256(source)


def bank_content_digest(files: Mapping[str, str]) -> str:
    """Exact-content fingerprint over a bank's executable files
    (path -> source), path-normalized and sorted."""
    joined = "\n".join(f"{p}::{files[p]}" for p in sorted(files))
    return _sha256(joined)


def ast_digest(source: str) -> str:
    """AST fingerprint with comments/docstrings/annotations stripped.

    A parse failure returns the plain content digest (never a crash — an
    unparseable candidate is the validator's problem, not the dedup's)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return content_digest(source)

    class _Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):  # noqa: N802
            node.body = [ast.Pass()] if not node.body else [
                s for s in node.body if not isinstance(s, ast.Expr)
                or not isinstance(getattr(s, "value", None), ast.Constant)
                or not isinstance(getattr(s.value, "value", None), str)
            ] or [ast.Pass()]
            # drop docstring
            node.body = [
                s for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant)
                        and isinstance(getattr(s.value, "value", None), str))
            ] or [ast.Pass()]
            self.generic_visit(node)
            return node

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_ClassDef(self, node):  # noqa: N802
            return self.visit_FunctionDef(node)

        def visit_Module(self, node):  # noqa: N802
            return self.visit_FunctionDef(node)

        def visit_arg(self, node):  # noqa: N802
            node.annotation = None
            return node

        def visit_FunctionDef_defaults(self, node):  # pragma: no cover
            return node

    stripped = _Strip().visit(tree)
    ast.fix_missing_locations(stripped)
    return _sha256(ast.dump(stripped))


@dataclass
class DedupCache:
    """Content-addressed evaluation cache + near-duplicate ledger."""

    exact: dict[str, dict] = field(default_factory=dict)
    near: dict[str, list[str]] = field(default_factory=dict)

    def lookup_exact(self, bank_digest: str) -> dict | None:
        return self.exact.get(bank_digest)

    def record(
        self,
        *,
        bank_digest: str,
        ast_digests: Mapping[str, str],
        result: Mapping,
    ) -> str | None:
        """Record an evaluation; returns the near-duplicate original digest
        when this bank's ASTs match a previously evaluated bank."""
        self.exact[bank_digest] = dict(result)
        # near-dup: any component whose AST matches a previously-seen digest
        # from a DIFFERENT bank
        for prev_digest, prev_asts in self.exact.items():
            if prev_digest == bank_digest:
                continue
            prev_ast_set = set((prev_digest, ))  # placeholder; real check below
        # simpler: maintain ast -> bank index
        for aid in ast_digests.values():
            prior = self.near.get(aid, [])
            prior_filtered = [d for d in prior if d != bank_digest]
            if prior_filtered:
                return prior_filtered[0]
        for aid in ast_digests.values():
            self.near.setdefault(aid, []).append(bank_digest)
        return None


def dedup_verdict(
    *,
    bank_digest: str,
    ast_digests: Mapping[str, str],
    cache: DedupCache,
) -> tuple[str, dict | None]:
    """Returns (verdict, prior_result):
      - "new"           : evaluate fully;
      - "exact_dup"     : reuse prior_result, skip the panel;
      - "near_dup:<id>" : prior bank shares an AST-equal component — the
        caller decides on a targeted recheck or a rejection note.
    """
    prior = cache.lookup_exact(bank_digest)
    if prior is not None:
        return "exact_dup", prior
    for aid in ast_digests.values():
        for prev in cache.near.get(aid, []):
            if prev != bank_digest:
                return f"near_dup:{prev}", cache.lookup_exact(prev)
    return "new", None
