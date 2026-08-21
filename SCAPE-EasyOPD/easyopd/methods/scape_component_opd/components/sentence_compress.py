from __future__ import annotations

from easyopd.methods.scape_component_opd.component_registry import get_component_spec

SPEC = get_component_spec("sentence_compress")

__all__ = ["SPEC"]
