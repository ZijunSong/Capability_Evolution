"""Environment / dependency / identity doctor.

Reports what is available and what is missing.  Missing optional environment
dependencies are reported by name; nothing silently falls back to another
environment.  Also audits that the running process does not depend on the
legacy research runtimes.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from rsi.config import CampaignConfig

LEGACY_MARKERS = ("trim", "scape", "scope", "harness1", "upstream_harness1")

CORE_MODULES = (
    "rsi.contracts",
    "rsi.runtime",
    "rsi.harness",
    "rsi.environments",
    "rsi.providers",
    "rsi.config",
)


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def check_no_legacy_imports() -> dict[str, Any]:
    """No legacy research runtime may be loaded in this process."""
    loaded = []
    for name, mod in list(sys.modules.items()):
        top = name.split(".")[0]
        if top in ("trim", "scape", "scope") or name.startswith("harness1"):
            loaded.append(name)
    return {"ok": not loaded, "legacy_modules_loaded": loaded}


def run_doctor(cfg: CampaignConfig | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "checks": {}}

    py_version = sys.version.split()[0]
    report["checks"]["python"] = {"ok": tuple(sys.version_info[:2]) >= (3, 11), "version": py_version}

    core = {m: _module_available(m) for m in CORE_MODULES}
    report["checks"]["core_modules"] = {"ok": all(core.values()), "modules": core}
    if not all(core.values()):
        report["ok"] = False

    legacy = check_no_legacy_imports()
    report["checks"]["no_legacy_runtime"] = legacy
    if not legacy["ok"]:
        report["ok"] = False

    env_report: dict[str, Any] = {}
    try:
        from rsi.environments.registry import global_registry

        registry = global_registry()
        for domain in registry.domains():
            ok, missing = registry.check_deps(domain)
            env_report[domain] = {"deps_ok": ok, "missing": list(missing)}
    except Exception as e:  # noqa: BLE001
        env_report["error"] = repr(e)
        report["ok"] = False
    report["checks"]["environments"] = env_report

    if cfg is not None:
        suite = cfg.section("suite")
        domains = list(suite.get("domains") or [])
        suite_report = {}
        for d in domains:
            entry = env_report.get(d, {"deps_ok": False, "missing": ["<unregistered>"]})
            suite_report[d] = entry
        report["checks"]["suite_domains"] = suite_report
        policy = cfg.section("policy")
        init = policy.get("initial_artifact")
        init_path = cfg.resolve_path(init) if init else None
        report["checks"]["policy"] = {
            "initial_artifact": str(init_path) if init_path else None,
            "initial_artifact_exists": bool(init_path and init_path.exists()) if init else "unset",
        }
        report["checks"]["config"] = {
            "campaign_id": cfg.campaign_id,
            "config_digest": cfg.digest(),
            "tier": cfg.raw.get("tier", "unspecified"),
        }
        training = cfg.section("training")
        backend = training.get("backend", "torch_fsdp2")
        backend_mod = {
            "hf_reference": "rsi.training.hf_reference",
            "torch_fsdp2": "rsi.training.fsdp2_backend",
        }.get(str(backend))
        backend_ok = _module_available(backend_mod) if backend_mod else False
        report["checks"]["training_backend"] = {"backend": backend, "importable": backend_ok}
        if not backend_ok:
            report["ok"] = False

    torch_info: dict[str, Any] = {"available": _module_available("torch")}
    if torch_info["available"]:
        import torch

        torch_info["version"] = torch.__version__
        torch_info["cuda"] = torch.cuda.is_available()
    report["checks"]["torch"] = torch_info

    synth = cfg.raw.get("tier") == "toy" if cfg else True
    report["synthetic_environment"] = synth
    return report
