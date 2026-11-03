"""Multi-GPU parallel panel evaluation.

Shards panel tasks across N worker processes, each pinned to its own GPU
(``CUDA_VISIBLE_DEVICES`` set per spawned child before the lazy torch import),
each building its own policy instance from the campaign config.  Results merge
into the same :class:`EvalRunResult` shape as the serial path with identical
per-task seeds, so serial and parallel runs are comparable.

This is task/data parallelism for evaluation (one model replica per GPU).  The
proposer runs in the parent process on ``RSI_PROPOSER_DEVICE`` (default cpu) so
it never collides with a worker's GPU memory.

Module import must stay torch-free at top level: spawned children set the
device env before the first lazy ``import torch``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.errors import InfraError, ProtocolViolation
from rsi.contracts.events import CostRecord
from rsi.contracts.observation import ScoreRecord
from rsi.evaluation.aggregation import aggregate_scores
from rsi.evaluation.costs import CostLedger
from rsi.evaluation.service import EvalRunResult

DEFAULT_MAX_WORKERS = 8


def detect_gpu_ids() -> list[int]:
    """GPU ids visible to this process (empty list when CUDA is unavailable)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - torch may be absent in CPU-only installs
        return []


def resolve_worker_plan(
    *,
    workers: int | None = None,
    gpu_ids: Sequence[int] | None = None,
    require_gpu: bool = False,
) -> tuple[int, list[int | None]]:
    """Return (n_workers, gpu_id_per_worker).  ``None`` gpu = CPU worker."""
    gpus = list(gpu_ids) if gpu_ids is not None else detect_gpu_ids()
    if workers is None:
        workers = min(DEFAULT_MAX_WORKERS, len(gpus)) if gpus else 1
    workers = max(1, int(workers))
    if workers > 1 and not gpus:
        if require_gpu:
            raise ProtocolViolation(
                f"panel parallelism requested ({workers} workers) but no CUDA GPU is "
                "visible; refusing to pretend CPU workers are an 8-GPU run"
            )
        return workers, [None] * workers  # honest CPU multiprocessing
    if workers > len(gpus):
        raise ProtocolViolation(
            f"panel workers={workers} exceeds visible GPUs={gpus}; pin fewer workers "
            "(RSI_PANEL_WORKERS) or free devices"
        )
    return workers, [gpus[i] for i in range(workers)]


@dataclass(frozen=True)
class TaskAssignment:
    domain: str
    task_id: str
    task_index: int  # index within domain's role slice; fixes the episode seed
    gpu_id: int | None


def _build_assignments(
    *,
    domains: Sequence[str],
    role: str,
    round_index: int | None,
    max_per_domain: int | None,
    gpu_ids: Sequence[int | None],
) -> list[TaskAssignment]:
    from rsi.environments.registry import global_registry

    registry = global_registry()
    assignments: list[TaskAssignment] = []
    k = 0
    for domain in domains:
        provider = registry.get(domain).task_provider_factory()
        tasks = provider.public_tasks(role, round_index=round_index)
        if max_per_domain is not None:
            tasks = tasks[:max_per_domain]
        for i, task in enumerate(tasks):
            assignments.append(
                TaskAssignment(
                    domain=domain,
                    task_id=task.task_id,
                    task_index=i,
                    gpu_id=gpu_ids[k % len(gpu_ids)],
                )
            )
            k += 1
    return assignments


# ---------------------------------------------------------------------------
# worker (top-level for spawn pickling; must stay torch-free at import time)
# ---------------------------------------------------------------------------


def _panel_worker(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    gpu_id = payload["gpu_id"]
    if gpu_id is not None:
        # Before any lazy torch import: the child sees exactly one device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from rsi.config import load_config
    from rsi.contracts.observation import ArtifactRef
    from rsi.contracts.policy import SamplingParams
    from rsi.environments.registry import global_registry
    from rsi.evaluation.service import resolve_bank
    from rsi.harness.loader import load_bank_ref
    from rsi.providers.factory import build_policy
    from rsi.runtime.budgets import BudgetSpec
    from rsi.runtime.loop import MinimalLoop

    cfg = load_config(payload["config_path"])
    runtime_cfg = cfg.section("runtime")
    budgets = BudgetSpec(
        max_actions=int(runtime_cfg.get("max_actions", 20)),
        max_extra_llm_calls=int(runtime_cfg.get("max_extra_llm_calls", 2)),
    )
    sampling = SamplingParams(
        max_new_tokens=int(runtime_cfg.get("max_new_tokens", 256)),
        temperature=float(runtime_cfg.get("temperature", 1.0)),
    )
    # Bank/policy construction failures are candidate-caused (e.g. a candidate
    # whose component fails protocol validation): report every assignment as an
    # error record instead of killing the whole pool (which used to abort the
    # entire evolve run via InfraError).
    try:
        bank = load_bank_ref(payload["bank_ref"]) if payload["bank_ref"] else resolve_bank("minimal", None)
        device = "cuda:0" if gpu_id is not None else "cpu"
        # Stagger model loads: 8 workers hitting from_pretrained at the same
        # instant stacked host-RAM peaks and disk seeks (kygt9). Small offset.
        if gpu_id:
            time.sleep(float(gpu_id) * 2.0)
        print(f"[panel_worker gpu={gpu_id}] building policy on {device}", flush=True)
        policy = build_policy(cfg, device=device)
        print(f"[panel_worker gpu={gpu_id}] policy ready", flush=True)
    except Exception as e:  # noqa: BLE001
        return [
            {
                "record": {
                    "task_id": spec["task_id"],
                    "domain": spec["domain"],
                    "metric_name": "task_success",
                    "value": 0.0,
                    "success": False,
                    "details": {"error": repr(e)},
                    "scorer_version": "error",
                    "synthetic": True,
                },
                "cost": {},
                "wall_ms": 0.0,
                "terminated_reason": "worker_setup_error",
                "n_decisions": 0,
                "bank_digest": "",
                "policy_digest": "",
                "component_errors": {"worker_setup": repr(e)},
            }
            for spec in payload["assignments"]
        ]
    registry = global_registry()

    out: list[dict[str, Any]] = []
    n_assign = len(payload["assignments"])
    print(f"[panel_worker gpu={gpu_id}] start: {n_assign} assignments", flush=True)
    for spec in payload["assignments"]:
        executable, why = registry.probe_executable(spec["domain"])
        if not executable:
            out.append(
                {
                    "record": {
                        "task_id": spec["task_id"],
                        "domain": spec["domain"],
                        "metric_name": "skipped",
                        "value": 0.0,
                        "success": False,
                        "details": {"skipped": why},
                        "scorer_version": "skipped",
                        "synthetic": False,
                    },
                    "cost": {},
                    "wall_ms": 0.0,
                    "terminated_reason": "domain_skipped",
                    "n_decisions": 0,
                    "bank_digest": "",
                    "policy_digest": "",
                    "component_errors": {},
                }
            )
            continue
        try:
            bundle = registry.get(spec["domain"])
            provider = bundle.task_provider_factory()
            scorer = bundle.scorer_factory()
            tasks = provider.public_tasks(payload["role"], round_index=payload["round_index"])
            if payload["max_per_domain"] is not None:
                tasks = tasks[: payload["max_per_domain"]]
            task = next(t for t in tasks if t.task_id == spec["task_id"])
            env = bundle.adapter_factory()
            loop = MinimalLoop(
                env=env,
                policy=policy,
                bank=bank,
                budgets=budgets,
                sampling=sampling,
                campaign_id=payload["campaign_id"],
            )
            t0 = time.perf_counter()
            result = loop.run_episode(task, seed=payload["seed"] + spec["task_index"])
            wall_ms = (time.perf_counter() - t0) * 1000.0
            artifact = result.final_artifact or ArtifactRef(ref_id=task.task_id, kind="answer")
            record = scorer.evaluate(artifact, provider.private_ref(task.task_id))
            from rsi.contracts.events import total_cost

            out.append(
                {
                    "record": {
                        "task_id": record.task_id,
                        "domain": record.domain,
                        "metric_name": record.metric_name,
                        "value": record.value,
                        "success": record.success,
                        "details": dict(record.details),
                        "scorer_version": record.scorer_version,
                        "synthetic": record.synthetic,
                    },
                    "cost": total_cost(result.trace.events).to_json(),
                    "wall_ms": wall_ms,
                    "terminated_reason": result.trace.terminated_reason,
                    "n_decisions": len(result.trace.decisions),
                    "bank_digest": bank.bank_digest,
                    "policy_digest": policy.policy_artifact.policy_id,
                    "component_errors": dict(result.component_errors),
                }
            )
            print(
                f"[panel_worker gpu={gpu_id}] done {spec['task_id']} "
                f"({len(out)}/{n_assign}) value={record.value} "
                f"reason={result.trace.terminated_reason} wall_ms={wall_ms:.0f}",
                flush=True,
            )
            env.close()
        except Exception as e:  # noqa: BLE001 - one task must not kill the pool
            out.append(
                {
                    "record": {
                        "task_id": spec["task_id"],
                        "domain": spec["domain"],
                        "metric_name": "task_success",
                        "value": 0.0,
                        "success": False,
                        "details": {"error": repr(e)},
                        "scorer_version": "error",
                        "synthetic": True,
                    },
                    "cost": {},
                    "wall_ms": 0.0,
                    "terminated_reason": "task_error",
                    "n_decisions": 0,
                    "bank_digest": "",
                    "policy_digest": "",
                    "component_errors": {"task_error": repr(e)},
                }
            )
    return out


# ---------------------------------------------------------------------------


def run_panel_mp(
    cfg: Any,
    *,
    bank_ref: str | None = None,
    role: str,
    seed: int,
    round_index: int | None = None,
    max_per_domain: int | None = None,
    campaign_id: str | None = None,
    workers: int | None = None,
    gpu_ids: Sequence[int] | None = None,
    require_gpu: bool = False,
) -> EvalRunResult:
    """Parallel :func:`run_panel` across pinned GPU workers.

    ``bank_ref`` is a bank ref/path (``None`` = minimal/empty bank).  Seeds are
    ``seed + task_index`` exactly as in the serial path, so serial and parallel
    results are comparable per task.
    """
    from rsi.config import CampaignConfig

    if not isinstance(cfg, CampaignConfig):
        raise ProtocolViolation("run_panel_mp requires a loaded CampaignConfig")
    suite_cfg = cfg.section("suite")
    domains = list(suite_cfg.get("domains") or [])
    n_workers, gpu_plan = resolve_worker_plan(
        workers=workers, gpu_ids=gpu_ids, require_gpu=require_gpu
    )
    assignments = _build_assignments(
        domains=domains,
        role=role,
        round_index=round_index,
        max_per_domain=max_per_domain,
        gpu_ids=gpu_plan,
    )
    if not assignments:
        suite = aggregate_scores([], aggregate=str(suite_cfg.get("aggregate", "equal_domain_macro")))
        return EvalRunResult(suite=suite, records=(), costs=CostLedger(), per_task={})

    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    chunk_gpu: list[int | None] = [None] * n_workers
    for i, a in enumerate(assignments):
        w = i % n_workers
        chunks[w].append({"domain": a.domain, "task_id": a.task_id, "task_index": a.task_index})
        chunk_gpu[w] = a.gpu_id

    payloads = [
        {
            "config_path": str(cfg.config_path),
            "bank_ref": bank_ref,
            "role": role,
            "seed": seed,
            "round_index": round_index,
            "max_per_domain": max_per_domain,
            "campaign_id": campaign_id or cfg.campaign_id,
            "gpu_id": chunk_gpu[w],
            "assignments": chunks[w],
        }
        for w in range(n_workers)
        if chunks[w]
    ]

    ctx = mp.get_context("spawn")
    try:
        # ProcessPoolExecutor (unlike mp.Pool.map) raises BrokenProcessPool when
        # a child is SIGKILLed (e.g. OOM), instead of waiting on it forever —
        # the kygt9 hang was Pool.map spinning on dead workers with zero output.
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
            results = list(ex.map(_panel_worker, payloads))
    except Exception as e:  # noqa: BLE001 - surface worker crashes loudly
        raise InfraError(f"panel worker failed: {e!r}") from e

    records: list[ScoreRecord] = []
    costs = CostLedger()
    per_task: dict[str, dict[str, Any]] = {}
    skipped_domains: dict[str, str] = {}
    for batch in results:
        for item in batch:
            r = item["record"]
            if r.get("metric_name") == "skipped":
                skipped_domains[str(r["domain"])] = str((r.get("details") or {}).get("skipped", "unknown"))
                continue
            record = ScoreRecord(
                task_id=r["task_id"],
                domain=r["domain"],
                metric_name=r["metric_name"],
                value=r["value"],
                success=r["success"],
                details=r["details"],
                scorer_version=r["scorer_version"],
                synthetic=r["synthetic"],
            )
            records.append(record)
            cost = CostRecord(**item["cost"])
            costs.add_episode(
                success=record.success,
                protocol_valid=item["terminated_reason"] not in ("max_actions",),
                completed=item["terminated_reason"] == "finish",
                cost=cost,
                wall_ms=item["wall_ms"],
            )
            per_task[record.task_id] = {
                "domain": record.domain,
                "value": record.value,
                "terminated_reason": item["terminated_reason"],
                "n_decisions": item["n_decisions"],
                "wall_ms": item["wall_ms"],
                "bank_digest": item["bank_digest"],
                "policy_digest": item["policy_digest"],
                "synthetic": record.synthetic,
                "component_errors": dict(item.get("component_errors") or {}),
            }
    records.sort(key=lambda r: (r.domain, r.task_id))
    suite = aggregate_scores(records, aggregate=str(suite_cfg.get("aggregate", "equal_domain_macro")))
    return EvalRunResult(
        suite=suite, records=tuple(records), costs=costs, per_task=per_task, skipped_domains=skipped_domains
    )
