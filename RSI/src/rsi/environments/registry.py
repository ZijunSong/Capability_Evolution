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

    def probe_executable(self, domain: str) -> tuple[bool, str]:
        """Can this domain actually execute + score right now?

        Tries the real constructors (scorer/adapter/provider).  Used to skip a
        domain loudly at panel level — same skip for baseline and every
        candidate keeps comparisons valid; the reason is recorded verbatim.
        """
        bundle = self.get(domain)
        ok, missing = self.check_deps(domain)
        if not ok:
            return False, f"missing optional deps: {list(missing)}"
        try:
            scorer = bundle.scorer_factory()
            adapter = bundle.adapter_factory()
            provider = bundle.task_provider_factory()
            probe_tasks = provider.public_tasks("work_train")
            for who, obj in (("adapter", adapter), ("scorer", scorer)):
                check_ready = getattr(obj, "check_ready", None)
                if callable(check_ready):
                    ready, why = check_ready()
                    if not ready:
                        try:
                            adapter.close()
                        except Exception:  # noqa: BLE001
                            pass
                        return False, f"{who} not ready: {why}"
        except NotImplementedCapability as e:
            return False, str(e)
        except Exception as e:  # noqa: BLE001
            return False, f"constructibility check failed: {e!r}"
        finally:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
        if not probe_tasks:
            return False, "no work_train tasks in provider"
        return True, "ok"


_GLOBAL = EnvironmentRegistry()


def global_registry() -> EnvironmentRegistry:
    # Import side-effect registrations for toy domains (unit-test fixtures only;
    # toy domains are never experiment targets — see docs/runbook.md).
    from rsi.environments import toy_registry  # noqa: F401

    if "appworld" not in _GLOBAL._bundles:
        from rsi.environments.appworld_env import AppWorldAdapter, AppWorldScorer
        from rsi.environments.task_providers import AppWorldTaskProvider

        _GLOBAL.register(
            EnvironmentBundle(
                domain="appworld",
                adapter_factory=lambda: AppWorldAdapter(),
                task_provider_factory=lambda: AppWorldTaskProvider(),
                scorer_factory=lambda: AppWorldScorer(),
                optional_deps=("appworld",),
                synthetic=False,
            )
        )
    if "spreadsheetbench" not in _GLOBAL._bundles:
        from rsi.environments.spreadsheetbench_env import (
            SpreadsheetBenchAdapter,
            SpreadsheetBenchScorer,
        )
        from rsi.environments.task_providers import SpreadsheetBenchTaskProvider

        _GLOBAL.register(
            EnvironmentBundle(
                domain="spreadsheetbench",
                adapter_factory=lambda: SpreadsheetBenchAdapter(),
                task_provider_factory=lambda: SpreadsheetBenchTaskProvider(),
                scorer_factory=lambda: SpreadsheetBenchScorer(),
                optional_deps=("openpyxl", "pandas"),
                synthetic=False,
            )
        )
    if "swebench" not in _GLOBAL._bundles:
        from rsi.environments.swebench_env import SWEBenchAdapter, SWEBenchScorer
        from rsi.environments.task_providers import SWEBenchTaskProvider

        _GLOBAL.register(
            EnvironmentBundle(
                domain="swebench",
                adapter_factory=lambda: SWEBenchAdapter(),
                task_provider_factory=lambda: SWEBenchTaskProvider(),
                scorer_factory=lambda: SWEBenchScorer(),
                optional_deps=("docker", "swebench"),
                synthetic=False,
            )
        )
    if "webshop" not in _GLOBAL._bundles:
        from rsi.environments.task_providers import WebShopTaskProvider
        from rsi.environments.webshop_env import WebShopAdapter, WebShopScorer

        _GLOBAL.register(
            EnvironmentBundle(
                domain="webshop",
                adapter_factory=lambda: WebShopAdapter(),
                task_provider_factory=lambda: WebShopTaskProvider(),
                scorer_factory=lambda: WebShopScorer(),
                optional_deps=("bs4", "thefuzz", "spacy", "pyserini"),
                synthetic=False,
            )
        )
    if "alfworld" not in _GLOBAL._bundles:
        from rsi.environments.alfworld_env import AlfWorldAdapter, AlfWorldScorer
        from rsi.environments.task_providers import AlfWorldTaskProvider

        _GLOBAL.register(
            EnvironmentBundle(
                domain="alfworld",
                adapter_factory=lambda: AlfWorldAdapter(),
                task_provider_factory=lambda: AlfWorldTaskProvider(),
                scorer_factory=lambda: AlfWorldScorer(),
                optional_deps=("textworld",),
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
