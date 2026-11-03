"""Campaign configuration: YAML in, validated resolved config out.

Three tiers are provided under ``configs/``:
- ``toy_smoke.yaml`` — no real benchmarks; every output marked synthetic
- ``pilot_multidomain.yaml`` — real models/environments, development budgets
- ``formal_multidomain.yaml`` — frozen thresholds, full data/statistics plan;
  refuses to run with placeholder values
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from rsi.contracts.common import canonical_json, digest_of
from rsi.contracts.errors import ProtocolViolation

SCHEMA_VERSION = "rsi_campaign_v1"
# rsi_campaign_v2 (audit doc §16) adds domain-role separation keys
# (evolution/opd/selection/sealed) and scoring/statistics sections; the
# runtime consumes them via section(), so the loader accepts both markers.
ACCEPTED_SCHEMA_VERSIONS = ("rsi_campaign_v1", "rsi_campaign_v2")


@dataclass
class CampaignConfig:
    raw: dict[str, Any]
    config_path: Path
    config_dir: Path

    # -- accessors -------------------------------------------------------
    @property
    def campaign_id(self) -> str:
        return str(self.raw.get("campaign_id", "campaign"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def max_rounds(self) -> int:
        return int(self.raw.get("max_rounds", 1))

    def section(self, name: str) -> dict[str, Any]:
        sec = self.raw.get(name)
        if sec is None:
            return {}
        if not isinstance(sec, dict):
            raise ProtocolViolation(f"config section {name!r} must be a mapping")
        return dict(sec)

    def resolve_path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        if path.is_absolute():
            return path
        # Relative paths resolve against the project root (parent of configs/),
        # matching the documented CLI usage `cd Capability_Evolution/RSI`.
        base = self.config_dir.parent if self.config_dir.name == "configs" else self.config_dir
        return (base / path).resolve()

    @property
    def project_root(self) -> Path:
        return self.config_dir.parent if self.config_dir.name == "configs" else self.config_dir

    def digest(self) -> str:
        return digest_of(self.raw)

    def to_resolved_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config_path": str(self.config_path),
            "raw": self.raw,
            "config_digest": self.digest(),
        }


REQUIRED_TOP_LEVEL = (
    "schema_version",
    "campaign_id",
    "seed",
    "max_rounds",
    "policy",
    "runtime",
    "suite",
    "evolution",
    "opd",
    "training",
    "promotion",
    "evaluation",
    "execution",
)


def load_config(path: str | Path) -> CampaignConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise ProtocolViolation(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ProtocolViolation("config root must be a mapping")
    version = raw.get("schema_version")
    if version not in ACCEPTED_SCHEMA_VERSIONS:
        raise ProtocolViolation(
            f"schema_version must be one of {ACCEPTED_SCHEMA_VERSIONS}, got {version!r}"
        )
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ProtocolViolation(
            f"config missing sections: {missing} — the v2 templates must carry "
            "a complete policy section (refactor guide §3 P0: the first v2 "
            "AppWorld template shipped without one and could not load)"
        )
    cfg = CampaignConfig(raw=raw, config_path=path, config_dir=path.parent)
    _validate_tier(cfg)
    _validate_domain_policy(cfg)
    return cfg


def _validate_domain_policy(cfg: CampaignConfig) -> None:
    """Refactor guide §7: in single-domain mode the role-separation keys must
    agree with suite.domains — a typo'd key (e.g. evolution_domains pointing
    at another environment) must fail loading, not silently run evolution on
    a different domain than declared."""
    suite = cfg.section("suite")
    domains = set(suite.get("domains") or [])
    if len(domains) != 1:
        return  # multi-domain legacy configs keep legacy behavior
    for key in ("evolution_domains", "opd_domains", "selection_domains"):
        vals = set(suite.get(key) or [])
        if vals and vals != domains:
            raise ProtocolViolation(
                f"suite.{key}={sorted(vals)} disagrees with suite.domains="
                f"{sorted(domains)} in single-domain mode — fix the config "
                "(per-environment campaigns declare one domain everywhere)"
            )
    sealed = set(suite.get("sealed_eval_domains") or [])
    if sealed & domains:
        raise ProtocolViolation(
            f"suite.sealed_eval_domains={sorted(sealed)} includes the campaign's "
            "own domain — a domain cannot be both evolved on and sealed"
        )


def _validate_tier(cfg: CampaignConfig) -> None:
    """formal configs must carry frozen, non-placeholder values."""
    tier = cfg.raw.get("tier")
    if tier != "formal":
        return
    problems: list[str] = []
    promo = cfg.section("promotion")
    policy = cfg.raw.get("promotion_policy_file") or promo.get("policy_file")
    if not policy:
        problems.append("formal config must point to a frozen promotion policy file")
    evo = cfg.section("evolution")
    if not evo.get("frozen_thresholds"):
        problems.append("formal config must declare evolution.frozen_thresholds")
    ev = cfg.section("evaluation")
    if ev.get("statistics_mode") == "pilot_reused_dev":
        problems.append("formal config must not use pilot_reused_dev statistics")
    if problems:
        raise ProtocolViolation("formal config invalid: " + "; ".join(problems))


def dump_resolved(cfg: CampaignConfig) -> str:
    return json.dumps(cfg.to_resolved_json(), indent=2, sort_keys=True, ensure_ascii=False)
