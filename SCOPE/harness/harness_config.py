"""Unified Harness module configuration loading and legacy env bridging."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.graph.module import ModuleConfig

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _module_from_dict(data: dict[str, Any], defaults: ModuleConfig) -> ModuleConfig:
    return ModuleConfig(
        module_id=str(data.get("module_id", defaults.module_id)),
        enabled=bool(data.get("enabled", defaults.enabled)),
        lifecycle_managed=bool(
            data.get("lifecycle_managed", defaults.lifecycle_managed)
        ),
        required=bool(data.get("required", defaults.required)),
        node_overrides=dict(data.get("node_overrides", defaults.node_overrides)),
        fallback_mode=str(data.get("fallback_mode", defaults.fallback_mode)),
        options={
            **defaults.options,
            **{k: v for k, v in data.items() if k not in ModuleConfig.__dataclass_fields__},
        },
    )


@dataclass
class HarnessConfig:
    retrieval: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            module_id="retrieval",
            enabled=True,
            required=True,
            lifecycle_managed=False,
        )
    )
    evidence_state: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            module_id="evidence_state",
            enabled=True,
            lifecycle_managed=True,
        )
    )
    verification: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            module_id="verification",
            enabled=True,
            lifecycle_managed=True,
        )
    )
    context_budget: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            module_id="context_budget",
            enabled=True,
            lifecycle_managed=True,
        )
    )
    recovery: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            module_id="recovery",
            enabled=False,
            lifecycle_managed=False,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": asdict(self.retrieval),
            "evidence_state": asdict(self.evidence_state),
            "verification": asdict(self.verification),
            "context_budget": asdict(self.context_budget),
            "recovery": asdict(self.recovery),
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save_resolved(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def default_full_config() -> HarnessConfig:
    return HarnessConfig(
        evidence_state=ModuleConfig(
            module_id="evidence_state",
            enabled=True,
            lifecycle_managed=True,
            options={
                "candidate_pool": True,
                "content_dedup": True,
                "evidence_graph": True,
                "importance_tagging": True,
                "subtractive_curation": True,
                "auto_seed": True,
                "review_memory": True,
                "render_structured_state": True,
                "preserve_minimal_selection": True,
            },
        ),
        verification=ModuleConfig(
            module_id="verification",
            enabled=True,
            lifecycle_managed=True,
            options={
                "expose_verify_tool": True,
                "store_records": True,
                "render_records": True,
                "verification_aware_curation": True,
            },
        ),
        context_budget=ModuleConfig(
            module_id="context_budget",
            enabled=True,
            lifecycle_managed=True,
            options={
                "sentence_compression": True,
                "structured_context_rendering": True,
                "recent_window": True,
                "token_budget_marker": True,
                "stop_budget_hint": True,
                "deterministic_truncation": False,
            },
        ),
    )


def from_legacy_env() -> HarnessConfig:
    """Map legacy V8D_* environment variables to HarnessConfig."""
    cfg = default_full_config()
    es = cfg.evidence_state.options
    es["content_dedup"] = _bool_env("V8D_CONTENT_DEDUP", es.get("content_dedup", True))
    es["evidence_graph"] = _bool_env("V8D_EVIDENCE_GRAPH", es.get("evidence_graph", True))
    es["importance_tagging"] = _bool_env(
        "V8D_IMPORTANCE_TAGGING", es.get("importance_tagging", True)
    )
    es["subtractive_curation"] = _bool_env(
        "V8D_SUBTRACTIVE_CURATION", es.get("subtractive_curation", True)
    )
    es["auto_seed"] = _bool_env(
        "V8D_AUTO_POPULATE_FIRST_SEARCH", es.get("auto_seed", True)
    )
    es["review_memory"] = not _bool_env("ABLATE_REVIEW_DOCS_UNAVAILABLE", False)
    es["render_structured_state"] = True
    es["preserve_minimal_selection"] = True

    ver = cfg.verification.options
    verify_on = _bool_env("V8D_VERIFY_TOOL", ver.get("expose_verify_tool", True))
    ablate_verify = _bool_env("ABLATE_VERIFY_UNAVAILABLE", False)
    cfg.verification.enabled = verify_on and not ablate_verify
    ver["expose_verify_tool"] = verify_on and not ablate_verify
    ver["store_records"] = verify_on
    ver["render_records"] = verify_on
    ver["verification_aware_curation"] = verify_on

    cb = cfg.context_budget.options
    cb["sentence_compression"] = _bool_env(
        "V8D_SENTENCE_COMPRESS", cb.get("sentence_compression", True)
    )
    cb["token_budget_marker"] = _bool_env(
        "V8D_TOKEN_BUDGET_MARKER", cb.get("token_budget_marker", True)
    )
    cb["structured_context_rendering"] = True
    cb["recent_window"] = True
    cb["stop_budget_hint"] = True

    cfg.retrieval.options["chunk_neighbors"] = _bool_env("V8D_CHUNK_NEIGHBORS", False)
    cfg.retrieval.options["rerank"] = True
    return cfg


def _apply_options_to_module(data: dict[str, Any], module_id: str) -> ModuleConfig:
    defaults = default_full_config().__getattribute__(module_id.replace("-", "_"))
    if module_id == "evidence_state":
        defaults = default_full_config().evidence_state
    elif module_id == "context_budget":
        defaults = default_full_config().context_budget
    elif module_id == "verification":
        defaults = default_full_config().verification
    elif module_id == "retrieval":
        defaults = default_full_config().retrieval
    elif module_id == "recovery":
        defaults = default_full_config().recovery
    return _module_from_dict({**data, "module_id": module_id}, defaults)


def load_harness_config(
    path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> HarnessConfig:
    """Load config with priority: CLI > YAML > legacy env > defaults."""
    if path is None:
        cfg = from_legacy_env()
    else:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = HarnessConfig(
            retrieval=_apply_options_to_module(raw.get("retrieval", {}), "retrieval"),
            evidence_state=_apply_options_to_module(
                raw.get("evidence_state", {}), "evidence_state"
            ),
            verification=_apply_options_to_module(
                raw.get("verification", {}), "verification"
            ),
            context_budget=_apply_options_to_module(
                raw.get("context_budget", {}), "context_budget"
            ),
            recovery=_apply_options_to_module(raw.get("recovery", {}), "recovery"),
        )

    if cli_overrides:
        for key, value in cli_overrides.items():
            if hasattr(cfg, key) and isinstance(value, dict):
                module = getattr(cfg, key)
                for opt_k, opt_v in value.items():
                    if opt_k in ModuleConfig.__dataclass_fields__:
                        setattr(module, opt_k, opt_v)
                    else:
                        module.options[opt_k] = opt_v

    return cfg


def apply_harness_config(config: HarnessConfig) -> dict[str, str]:
    """Apply HarnessConfig to process environment (legacy V8D_* bridge)."""
    es = config.evidence_state
    opts = es.options
    env: dict[str, str] = {
        "V8D_CONTENT_DEDUP": "1" if opts.get("content_dedup", False) and es.enabled else "0",
        "V8D_EVIDENCE_GRAPH": "1" if opts.get("evidence_graph", False) and es.enabled else "0",
        "V8D_IMPORTANCE_TAGGING": "1"
        if opts.get("importance_tagging", False) and es.enabled
        else "0",
        "V8D_SUBTRACTIVE_CURATION": "1"
        if opts.get("subtractive_curation", False) and es.enabled
        else "0",
        "V8D_AUTO_POPULATE_FIRST_SEARCH": "1"
        if opts.get("auto_seed", False) and es.enabled
        else "0",
        "ABLATE_REVIEW_DOCS_UNAVAILABLE": "0"
        if opts.get("review_memory", True) and es.enabled
        else "1",
    }

    ver = config.verification
    vopts = ver.options
    verify_on = ver.enabled and vopts.get("expose_verify_tool", False)
    env["V8D_VERIFY_TOOL"] = "1" if verify_on else "0"
    env["ABLATE_VERIFY_UNAVAILABLE"] = "0" if verify_on else "1"

    cb = config.context_budget
    copts = cb.options
    cb_on = cb.enabled
    env["V8D_SENTENCE_COMPRESS"] = (
        "1" if cb_on and copts.get("sentence_compression", False) else "0"
    )
    env["V8D_TOKEN_BUDGET_MARKER"] = (
        "1" if cb_on and copts.get("token_budget_marker", False) else "0"
    )
    env["V8D_CHUNK_NEIGHBORS"] = (
        "1" if config.retrieval.options.get("chunk_neighbors", False) else "0"
    )

    for key, value in env.items():
        os.environ[key] = value
    return env


def config_path(name: str) -> Path:
    return CONFIGS_DIR / name
