"""Environment registry: protocol-level indirection from domains to adapters.

Core code never routes on benchmark names.  A campaign config names domains;
this registry resolves them to (adapter factory, task provider, scorer).  An
unavailable optional dependency makes the factory raise
:class:`NotImplementedCapability` and ``doctor`` report the gap — it never
silently falls back to another environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rsi.contracts.errors import NotImplementedCapability
from rsi.environments.base import EnvironmentAdapter, Scorer, TaskProvider


@dataclass(frozen=True)
class EnvironmentBundle:
    domain: str
    adapter_factory: Callable[[], EnvironmentAdapter]
    task_provider_factory: Callable[[], TaskProvider]
    scorer_factory: Callable[[], Scorer]
    optional_deps: tuple[str, ...] = ()
    synthetic: bool = False


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._bundles: dict[str, EnvironmentBundle] = {}

    def register(self, bundle: EnvironmentBundle) -> None:
        self._bundles[bundle.domain] = bundle

    def domains(self) -> tuple[str, ...]:
        return tuple(sorted(self._bundles))

    def get(self, domain: str) -> EnvironmentBundle:
        if domain not in self._bundles:
            raise NotImplementedCapability(
                f"environment domain {domain!r} is not registered; "
                f"known: {sorted(self._bundles)}"
            )
        return self._bundles[domain]

    def check_deps(self, domain: str) -> tuple[bool, tuple[str, ...]]:
        bundle = self.get(domain)
        missing = []
        for mod in bundle.optional_deps:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        return (not missing, tuple(missing))

    def resolve(self, domain: str) -> tuple[EnvironmentAdapter, TaskProvider, Scorer]:
        bundle = self.get(domain)
        ok, missing = self.check_deps(domain)
        if not ok:
            raise NotImplementedCapability(
                f"environment {domain!r} missing optional dependencies: {list(missing)}; "
                "install them or drop the domain from the suite — no fallback"
            )
        return bundle.adapter_factory(), bundle.task_provider_factory(), bundle.scorer_factory()


_GLOBAL = EnvironmentRegistry()


def global_registry() -> EnvironmentRegistry:
    # Import side-effect registrations for toy domains.
    from rsi.environments import toy_registry  # noqa: F401

    # Real benchmark domains: registered here with lazy factories so that the
    # modules can be imported without their optional dependencies present.
    from rsi.environments.appworld_env import AppWorldAdapter
    from rsi.environments.spreadsheetbench_env import SpreadsheetBenchAdapter
    from rsi.environments.swebench_env import SWEBenchAdapter

    for domain, adapter_cls, deps in (
        ("appworld", AppWorldAdapter, ("appworld",)),
        ("spreadsheetbench", SpreadsheetBenchAdapter, ("openpyxl", "pandas")),
        ("swebench", SWEBenchAdapter, ("docker", "swebench")),
    ):
        if domain not in _GLOBAL._bundles:
            _GLOBAL.register(
                EnvironmentBundle(
                    domain=domain,
                    adapter_factory=lambda c=adapter_cls: c(),  # type: ignore[call-arg]
                    task_provider_factory=lambda d=domain: _missing_provider(d),
                    scorer_factory=lambda d=domain: _missing_scorer(d),
                    optional_deps=deps,
                    synthetic=False,
                )
            )
    return _GLOBAL


def _missing_provider(domain: str) -> TaskProvider:
    raise NotImplementedCapability(
        f"task provider for {domain!r} is wired when the optional dependency is "
        "installed and the frozen dataset revision is prepared (data prepare)"
    )


def _missing_scorer(domain: str) -> Scorer:
    raise NotImplementedCapability(
        f"official scorer for {domain!r} is wired when the optional dependency is "
        "installed and the frozen dataset revision is prepared (data prepare)"
    )


def register_bundle(bundle: EnvironmentBundle) -> None:
    _GLOBAL.register(bundle)
