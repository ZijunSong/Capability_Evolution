"""Task providers for the registered benchmark domains.

Role splits follow task book §11.2.  Group clustering keys:

- AppWorld: ``scenario`` cluster (all tasks of one scenario stay in one role)
- SpreadsheetBench: question id (multi-workbook test cases stay together)
- SWE-bench: instance/issue family; train+dev pool is family-deduplicated
  against the Verified/Lite test panels (nothing from the final panels can
  leak into any development role).
- WebShop: product asin (all human instructions for one product stay in one
  role); the official 500-goal human test panel (goal index 0..499) is
  ``final_test`` and is asin-deduplicated out of every other role.
- ALFWorld: task instance (same scene/goal trials stay together); the official
  release carries train/valid_seen/valid_unseen (no test_unseen/OOD — see
  alfworld/data/README.md): ``final_test`` prefers ``test_unseen`` then
  ``valid_unseen``, ``transfer_test`` prefers ``out_of_distribution`` then
  ``valid_seen``; development roles come from ``train``.

The frozen split manifests are written by ``rsi.cli data prepare``; providers
re-derive the same deterministic assignment and the loader verifies digests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from rsi.contracts.errors import NotImplementedCapability, ProtocolViolation
from rsi.contracts.task import PrivateTaskRef, PublicTask

DEFAULT_APPWORLD_ROOT = "/mnt/songzijun/datasets/appworld"
DEFAULT_SSB_ROOT = "/mnt/songzijun/datasets/spreadsheetbench"
DEFAULT_SWE_ROOT = "/mnt/songzijun/datasets/swebench"
DEFAULT_WEBSHOP_ROOT = "/mnt/songzijun/datasets/webshop"
DEFAULT_ALFWORLD_ROOT = "/mnt/songzijun/datasets/alfworld"

# AppWorld's path store caches APPWORLD_ROOT at first package import — pin the
# dataset root BEFORE any `import appworld` can happen in this process.
import os as _os

_os.environ.setdefault("APPWORLD_ROOT", DEFAULT_APPWORLD_ROOT)
_os.environ.setdefault("RSI_SSB_ROOT", DEFAULT_SSB_ROOT)
_os.environ.setdefault("RSI_SWEBENCH_ROOT", DEFAULT_SWE_ROOT)
_os.environ.setdefault("RSI_WEBSHOP_ROOT", DEFAULT_WEBSHOP_ROOT)
_os.environ.setdefault("RSI_ALFWORLD_ROOT", DEFAULT_ALFWORLD_ROOT)

ROLE_DATASET = {
    # role -> AppWorld dataset_name
    "work_train": "train",
    "harness_select": "train",
    "weight_dev": "train",
    "confirm_h": "dev",
    "confirm_w": "dev",
    "final_test": "test_normal",
    "transfer_test": "test_challenge",
}


class AppWorldTaskProvider:
    """AppWorld tasks with scenario-level grouping.

    Task ids follow ``<generator_id>_<n>[:tag]``; the scenario (generator id)
    is the group key so a scenario never straddles split roles.  Tasks are
    enumerated from ``data/tasks/`` and split deterministically by group hash
    (generated universes have no official release split; the split is frozen at
    ``data prepare``).  The (heavier) per-task metadata is materialized only
    for the slice actually requested.
    """

    domain = "appworld"

    def __init__(self, data_root: str | None = None, *, panel_sizes: dict[str, int] | None = None):
        self._root = data_root or os.environ.get("APPWORLD_ROOT", DEFAULT_APPWORLD_ROOT)
        self._panel_sizes = dict(panel_sizes or {})
        self._task_cache: dict[str, PublicTask] = {}
        self._groups: dict[str, tuple[str, ...]] = {}
        self._roles: dict[str, list[str]] | None = None
        # "official" (datasets/*.txt gating) | "custom_generated" (explicit opt-in)
        self._split_source: str = "unresolved"

    def _all_task_ids(self) -> list[str]:
        import re

        tasks_dir = Path(self._root) / "data" / "tasks"
        if not tasks_dir.is_dir() or not any(tasks_dir.iterdir()):
            raise NotImplementedCapability(
                f"AppWorld tasks not found (or empty) under {tasks_dir} — run data prepare "
                "(official bundle download or the official generation pipeline)"
            )
        ids = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
        return [t for t in ids if re.match(r"^[A-Za-z0-9]+_\d+", t.split(":")[0]) or t == "base"]

    @staticmethod
    def _generator_id(task_id: str) -> str:
        base = task_id.split(":")[0]
        if base == "base":
            return "base"
        return base.rsplit("_", 1)[0]

    def _materialize(self, tid: str) -> PublicTask:
        if tid in self._task_cache:
            return self._task_cache[tid]
        os.environ.setdefault("APPWORLD_ROOT", self._root)
        from appworld.task import Task

        # Strict mode (default): Task.load failure must fail the preflight,
        # NOT degrade to a placeholder instruction (audit §8.2/§14).
        strict = os.environ.get("RSI_APPWORLD_STRICT_TASK_LOAD", "1") != "0"
        instruction = ""
        load_error: str | None = None
        try:
            raw = Task.load(tid, load_ground_truth=False)
            instruction = str(getattr(raw, "instruction", "") or getattr(raw, "main_goal", ""))
        except Exception as e:  # noqa: BLE001
            load_error = repr(e)
            if strict:
                raise ProtocolViolation(
                    f"appworld Task.load({tid!r}) failed: {e!r} — data preflight "
                    "must block rather than substitute placeholder instructions"
                ) from e
        if strict and not instruction.strip():
            raise ProtocolViolation(
                f"appworld task {tid!r} has an empty instruction — refusing to "
                "substitute a placeholder"
            )
        scenario = self._generator_id(tid)
        task = PublicTask(
            task_id=f"appworld:{tid}",
            domain=self.domain,
            instruction=instruction or f"(appworld task {tid})",
            group_id=f"appworld:{scenario}",
            metadata={
                "appworld_task_id": tid,
                "scenario": scenario,
                **({"load_error": load_error} if load_error else {}),
            },
        )
        self._task_cache[tid] = task
        self._groups.setdefault(task.group_id, tuple())
        self._groups[task.group_id] = tuple(sorted(set(self._groups[task.group_id]) | {task.task_id}))
        return task

    def _official_split_members(self) -> dict[str, set[str]] | None:
        """Official train/dev/test membership from ``data/datasets/*.txt``
        (shipped by ``appworld download datasets``).  None when the listings
        are absent — the caller then decides, loudly, what to do (audit
        guide §8.1: official split must gate candidates BEFORE role hashing)."""
        datasets_dir = Path(self._root) / "data" / "datasets"
        out: dict[str, set[str]] = {}
        for name in ("train", "dev", "test_normal", "test_challenge"):
            f = datasets_dir / f"{name}.txt"
            if not f.is_file():
                return None
            out[name] = {ln.strip() for ln in f.read_text("utf-8").splitlines() if ln.strip()}
        return out

    def _split_roles(self) -> dict[str, list[str]]:
        if self._roles is not None:
            return self._roles
        import hashlib

        ids = self._all_task_ids()
        official = self._official_split_members()
        allow_unofficial = os.environ.get("RSI_APPWORLD_ALLOW_UNOFFICIAL_SPLIT", "") == "1"
        if official is None:
            if not allow_unofficial:
                raise ProtocolViolation(
                    "appworld official dataset listings missing "
                    f"({self._root}/data/datasets/*.txt) — run `appworld download "
                    "datasets` first, or set RSI_APPWORLD_ALLOW_UNOFFICIAL_SPLIT=1 "
                    "to explicitly mark this a custom-generated split (recorded "
                    "as such in every manifest; NOT the official test protocol)"
                )
            self._split_source = "custom_generated"
        else:
            self._split_source = "official"
            dev_ids: set[str] = official["train"] | official["dev"]
            n_test = official["test_normal"]
            n_chal = official["test_challenge"]
            known = set(ids)
            missing_dev = dev_ids - known
            if missing_dev:
                raise ProtocolViolation(
                    f"official train/dev lists {len(missing_dev)} task ids not "
                    "present under data/tasks (dataset/tasks revision mismatch)"
                )
            # §8.1: restrict the candidate pool by official membership, then
            # hash-assign DEVELOPMENT roles inside the non-test pool only.
            ids = sorted(dev_ids)
            self._official_test_ids = {"test_normal": sorted(n_test), "test_challenge": sorted(n_chal)}
        # group tasks by scenario, then assign each scenario group to one role
        by_group: dict[str, list[str]] = {}
        for tid in ids:
            by_group.setdefault(self._generator_id(tid), []).append(tid)
        order = ["work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w"]
        buckets: dict[str, list[str]] = {k: [] for k in order}
        for gid in sorted(by_group):
            h = int(hashlib.sha256(gid.encode()).hexdigest(), 16)
            buckets[order[h % len(order)]].extend(sorted(by_group[gid]))
        self._roles = dict(buckets)
        if self._split_source == "custom_generated":
            # legacy behavior preserved only under the explicit opt-in: the
            # old 7-role hash split included test pools — keep the mapping but
            # record the source so nothing claims official status.
            legacy_order = order + ["final_test", "transfer_test"]
            legacy_buckets: dict[str, list[str]] = {k: [] for k in legacy_order}
            for gid in sorted(by_group):
                h = int(hashlib.sha256(gid.encode()).hexdigest(), 16)
                legacy_buckets[legacy_order[h % len(legacy_order)]].extend(sorted(by_group[gid]))
            self._roles = legacy_buckets
        else:
            # official mode: final_test = official test_normal verbatim;
            # transfer_test = official test_challenge verbatim.
            self._roles["final_test"] = sorted(getattr(self, "_official_test_ids", {}).get("test_normal", []))
            self._roles["transfer_test"] = sorted(getattr(self, "_official_test_ids", {}).get("test_challenge", []))
        return self._roles

    def materialize_task_id(self, tid: str) -> PublicTask:
        """Exact-id materialization for the frozen-manifest path (audit §8.2).

        Manifest ids carry the ``appworld:`` domain prefix while the internal
        materializer expects the raw AppWorld id; strip exactly one prefix and
        verify the round-trip so the returned PublicTask.task_id matches the
        manifest id verbatim.
        """
        raw = tid.split(":", 1)[1] if tid.startswith(f"{self.domain}:") else tid
        task = self._materialize(raw)
        if task.task_id != tid and not raw.startswith(f"{self.domain}:"):
            # _materialize re-prefixes raw ids; a manifest id WITH the prefix
            # must round-trip identically — anything else is a frozen-source
            # mismatch, not a silent substitution.
            if task.task_id != raw:
                raise ProtocolViolation(
                    f"appworld materialized task_id {task.task_id!r} does not match "
                    f"manifest id {tid!r}"
                )
        return task

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        ids = self._split_roles().get(role, [])
        n = self._panel_sizes.get(role)
        take = ids[:n] if n else ids
        return tuple(self._materialize(tid) for tid in take)

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def groups(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = dict(self._groups)
        for role in self._split_roles():
            for tid in self._split_roles()[role]:
                self._materialize(tid)
        return dict(self._groups)

    def all_tasks(self) -> tuple[PublicTask, ...]:
        for role in self._split_roles():
            for tid in self._split_roles()[role]:
                self._materialize(tid)
        return tuple(self._task_cache.values())


class SWEBenchTaskProvider:
    """SWE-bench tasks from frozen parquet panels with issue-family dedup."""

    domain = "swebench"

    def __init__(self, dataset_root: str | None = None, *, panel_sizes: dict[str, int] | None = None):
        self._root = Path(dataset_root or os.environ.get("RSI_SWEBENCH_ROOT", DEFAULT_SWE_ROOT))
        if panel_sizes is None and os.environ.get("RSI_PANEL_SIZES"):
            # e.g. RSI_PANEL_SIZES='{"final_test": 500, "transfer_test": 300}' to run
            # the full official sets instead of the small frozen slices.
            import json as _json

            panel_sizes = _json.loads(os.environ["RSI_PANEL_SIZES"])
        self._panel_sizes = dict(panel_sizes or {})
        self._by_id: dict[str, PublicTask] = {}
        self._private: dict[str, dict[str, Any]] = {}
        self._roles: dict[str, list[PublicTask]] = {}
        self._loaded = False

    def _parquet_rows(self, name: str) -> list[dict[str, Any]]:
        import pandas as pd

        p = self._root / name
        if not p.is_file():
            raise NotImplementedCapability(
                f"SWE-bench data missing: {p} — run data prepare (frozen parquet downloads)"
            )
        return pd.read_parquet(p).to_dict("records")

    def _load(self) -> None:
        if self._loaded:
            return
        import hashlib

        verified = {r["instance_id"] for r in self._parquet_rows("verified_test.parquet")}
        lite = {r["instance_id"] for r in self._parquet_rows("lite_test.parquet")}
        dev_pool: list[PublicTask] = []
        for name in ("swebench_train.parquet", "swebench_dev.parquet"):
            for r in self._parquet_rows(name):
                iid = r["instance_id"]
                if iid in verified or iid in lite:
                    continue  # family dedup: final panels never enter development
                task = self._make_task(r, dataset=name)
                dev_pool.append(task)
        final = [self._make_task(r, dataset="verified_test.parquet") for r in self._parquet_rows("verified_test.parquet")]
        transfer_ids = lite - verified
        transfer = [
            self._make_task(r, dataset="lite_test.parquet")
            for r in self._parquet_rows("lite_test.parquet")
            if r["instance_id"] in transfer_ids
        ]
        # Deterministic hash ordering — spreads repos across role slices
        # (plain id sort would pack the head of the alphabet into work_train).
        def hkey(t: PublicTask) -> str:
            return hashlib.sha256(t.task_id.encode()).hexdigest()

        dev_pool.sort(key=hkey)
        final.sort(key=hkey)
        transfer.sort(key=hkey)
        n = self._panel_sizes
        self._roles = {
            "work_train": dev_pool[: n.get("work_train") or 64],
            "harness_select": dev_pool[n.get("work_train") or 64 : (n.get("work_train") or 64) + (n.get("harness_select") or 32)],
            "weight_dev": dev_pool[(n.get("work_train") or 64) + (n.get("harness_select") or 32) :][: n.get("weight_dev") or 32],
            "confirm_h": dev_pool[-(n.get("confirm_h") or 16) :] if len(dev_pool) > 48 else [],
            "confirm_w": [],
            "final_test": final[: n.get("final_test") or 32],
            "transfer_test": transfer[: n.get("transfer_test") or 16],
        }
        # confirm_w: disjoint slice after confirm_h
        if len(dev_pool) > 64:
            self._roles["confirm_w"] = dev_pool[-(n.get("confirm_h") or 16) - (n.get("confirm_w") or 16) : -(n.get("confirm_h") or 16)]
        self._loaded = True

    def _make_task(self, r: dict[str, Any], *, dataset: str) -> PublicTask:
        iid = str(r["instance_id"])
        task = PublicTask(
            task_id=f"swebench:{iid}",
            domain=self.domain,
            instruction=str(r.get("problem_statement", "")),
            group_id=f"swebench:{iid}",  # issue family = instance; dedup handled above
            metadata={
                "instance_id": iid,
                "repo": str(r.get("repo", "")),
                "base_commit": str(r.get("base_commit", "")),
                "version": str(r.get("version", "")),
                "dataset": dataset,
            },
        )
        self._by_id[task.task_id] = task
        self._private[task.task_id] = {
            "patch": r.get("patch"),
            "test_patch": r.get("test_patch"),
            "FAIL_TO_PASS": r.get("FAIL_TO_PASS"),
            "PASS_TO_PASS": r.get("PASS_TO_PASS"),
        }
        return task

    def materialize_task_id(self, tid: str) -> PublicTask:
        """Exact-id materialization for the frozen-manifest path (audit §8.2).

        Unknown ids are a ProtocolViolation, never a placeholder task."""
        self._load()
        try:
            return self._by_id[tid]
        except KeyError as exc:
            raise ProtocolViolation(
                f"webshop task {tid!r} absent from the frozen goal manifest"
            ) from exc

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._roles.get(role, ()))

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def private_record(self, task_id: str) -> dict[str, Any]:
        self._load()
        return dict(self._private.get(task_id, {}))

    def groups(self) -> dict[str, tuple[str, ...]]:
        self._load()
        out: dict[str, list[str]] = {}
        for t in self._by_id.values():
            out.setdefault(t.group_id, []).append(t.task_id)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def all_tasks(self) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._by_id.values())


class SpreadsheetBenchTaskProvider:
    """SpreadsheetBench questions (all_data_912_v0.1 schema).

    Group = question id — all test-case workbooks of one question stay in one
    role.  Split assignment is deterministic by group hash.
    """

    domain = "spreadsheetbench"

    def __init__(self, dataset_root: str | None = None, *, panel_sizes: dict[str, int] | None = None):
        self._root = Path(dataset_root or os.environ.get("RSI_SSB_ROOT", DEFAULT_SSB_ROOT))
        self._panel_sizes = dict(panel_sizes or {})
        self._by_id: dict[str, PublicTask] = {}
        self._roles: dict[str, list[PublicTask]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from rsi.environments.spreadsheetbench_env import load_ssb_questions

        rows = load_ssb_questions(self._root)
        tasks: list[PublicTask] = []
        for r in rows:
            qid = str(r["id"])
            task = PublicTask(
                task_id=f"ssb:{qid}",
                domain=self.domain,
                instruction=str(r.get("instruction", "")),
                group_id=f"ssb:{qid}",
                metadata={
                    "question_id": qid,
                    "answer_position": str(r.get("answer_position", "")),
                    "answer_sheet": str(r.get("answer_sheet", "")),
                    "instruction_type": str(r.get("instruction_type", "")),
                },
            )
            tasks.append(task)
            self._by_id[task.task_id] = task
        tasks.sort(key=lambda t: t.task_id)
        import hashlib

        order = ["work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w", "final_test", "transfer_test"]
        buckets: dict[str, list[PublicTask]] = {k: [] for k in order}
        for t in tasks:
            h = int(hashlib.sha256(t.group_id.encode()).hexdigest(), 16)
            buckets[order[h % len(order)]].append(t)
        self._roles = {
            k: (v[: self._panel_sizes[k]] if self._panel_sizes.get(k) else v) for k, v in buckets.items()
        }
        self._loaded = True

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._roles.get(role, ()))

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def groups(self) -> dict[str, tuple[str, ...]]:
        self._load()
        return {t.group_id: (t.task_id,) for t in self._by_id.values()}

    def all_tasks(self) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._by_id.values())


class WebShopTaskProvider:
    """WebShop human goals from the frozen goal manifest.

    ``frozen_goals.json`` (materialized by ``scripts/prepare_webshop_data.py``)
    pins the official ``get_goals(human_goals=True)`` output under a fixed RNG
    seed in the official post-shuffle order — goal index ``i`` is exactly the
    official ``reset(session=i)`` index.  ``final_test`` is the official
    500-goal human test panel (indices 0..499); the transfer panel and the
    development roles are asin-grouped and asin-deduplicated against it.
    """

    domain = "webshop"

    DEV_ROLES = ("work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w")

    def __init__(self, data_root: str | None = None, *, panel_sizes: dict[str, int] | None = None):
        self._root = Path(data_root or os.environ.get("RSI_WEBSHOP_ROOT", DEFAULT_WEBSHOP_ROOT))
        self._panel_sizes = dict(panel_sizes or {})
        self._by_id: dict[str, PublicTask] = {}
        self._private: dict[str, dict[str, Any]] = {}
        self._roles: dict[str, list[PublicTask]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import hashlib

        from rsi.environments.webshop_env import _load_frozen

        manifest, goals, _prices = _load_frozen(self._root)
        panel = dict(manifest.get("panel") or {})
        n_final = int(panel.get("final_test_count") or 500)
        n_transfer = int(panel.get("transfer_test_count") or 250)
        if not goals:
            raise NotImplementedCapability(f"webshop frozen goals empty under {self._root}")
        if len(goals) <= n_final:
            raise NotImplementedCapability(
                f"webshop frozen goals ({len(goals)}) too small for the official "
                f"{n_final}-goal final panel"
            )

        def make(i: int) -> PublicTask:
            g = goals[i]
            asin = str(g.get("asin") or "")
            task = PublicTask(
                task_id=f"webshop:g{i:05d}",
                domain=self.domain,
                instruction=str(g.get("instruction_text") or f"(webshop goal {i})"),
                group_id=f"webshop:{asin}",
                metadata={"webshop_goal_index": i, "asin": asin},
            )
            self._by_id[task.task_id] = task
            self._private[task.task_id] = dict(g)
            return task

        final_tasks = [make(i) for i in range(n_final)]
        final_asins = {str(goals[i].get("asin") or "") for i in range(n_final)}

        # Remaining goals, whole asin groups, panel-asin families dropped
        # (same anti-leak rule as the SWE-bench family dedup).
        by_group: dict[str, list[int]] = {}
        for i in range(n_final, len(goals)):
            asin = str(goals[i].get("asin") or "")
            if asin in final_asins:
                continue
            by_group.setdefault(asin, []).append(i)
        for members in by_group.values():
            members.sort()

        # Deterministic hash order; whole groups fill the transfer panel first.
        group_order = sorted(by_group, key=lambda a: hashlib.sha256(a.encode()).hexdigest())
        transfer: list[PublicTask] = []
        transfer_groups: set[str] = set()
        for asin in group_order:
            if len(transfer) >= n_transfer:
                break
            transfer.extend(make(i) for i in by_group[asin])
            transfer_groups.add(asin)
        buckets: dict[str, list[PublicTask]] = {k: [] for k in self.DEV_ROLES}
        for asin in group_order:
            if asin in transfer_groups:
                continue
            h = int(hashlib.sha256(asin.encode()).hexdigest(), 16)
            buckets[self.DEV_ROLES[h % len(self.DEV_ROLES)]].extend(make(i) for i in by_group[asin])
        self._roles = {
            "work_train": buckets["work_train"],
            "harness_select": buckets["harness_select"],
            "weight_dev": buckets["weight_dev"],
            "confirm_h": buckets["confirm_h"],
            "confirm_w": buckets["confirm_w"],
            "final_test": final_tasks,
            "transfer_test": transfer,
        }
        n = self._panel_sizes
        self._roles = {k: (v[: n[k]] if n.get(k) else v) for k, v in self._roles.items()}
        self._loaded = True

    def materialize_task_id(self, tid: str) -> PublicTask:
        """Exact-id materialization for the frozen-manifest path (audit §8.2).

        Unknown ids are a ProtocolViolation, never a placeholder task."""
        self._load()
        try:
            return self._by_id[tid]
        except KeyError as exc:
            raise ProtocolViolation(
                f"webshop task {tid!r} absent from the frozen goal manifest"
            ) from exc

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._roles.get(role, ()))

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def private_record(self, task_id: str) -> dict[str, Any]:
        self._load()
        return dict(self._private.get(task_id, {}))

    def groups(self) -> dict[str, tuple[str, ...]]:
        self._load()
        out: dict[str, list[str]] = {}
        for t in self._by_id.values():
            out.setdefault(t.group_id, []).append(t.task_id)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def all_tasks(self) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._by_id.values())


class AlfWorldTaskProvider:
    """ALFWorld games from the prepared ``$ALFWORLD_DATA``-style tree.

    Development roles are hash-assigned from the official ``train`` split at
    task-instance level (all trials of one scene/goal stay together).
    ``final_test`` prefers the paper's ``test_unseen`` panel and falls back to
    the released ``valid_unseen``; ``transfer_test`` prefers
    ``out_of_distribution`` then ``valid_seen`` — which split fed each panel is
    recorded in the frozen manifest (``final_split``/``transfer_split``).
    ``frozen_manifest.json`` pins per-split counts and the tree digest.
    """

    domain = "alfworld"

    DEV_ROLES = ("work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w")
    # Official release (alfworld/data/README.md): train 3553 + valid_seen 140 +
    # valid_unseen 134 — the paper's test_unseen/OOD panels are not in the
    # public json_2.1.1 dump.  Prefer them when present, else the released
    # eval splits (eval_config.yaml evaluates on valid_seen/valid_unseen).
    # Which split fed each panel is recorded in the frozen manifest.
    FINAL_SPLITS = ("test_unseen", "valid_unseen", "valid_seen")
    TRANSFER_SPLITS = ("out_of_distribution", "valid_seen", "valid_train")

    def __init__(self, data_root: str | None = None, *, panel_sizes: dict[str, int] | None = None):
        self._root = Path(data_root or os.environ.get("RSI_ALFWORLD_ROOT", DEFAULT_ALFWORLD_ROOT))
        self._panel_sizes = dict(panel_sizes or {})
        self._by_id: dict[str, PublicTask] = {}
        self._private: dict[str, dict[str, Any]] = {}
        self._roles: dict[str, list[PublicTask]] = {}
        self._loaded = False

    def _frozen_manifest(self) -> dict[str, Any]:
        path = self._root / "frozen_manifest.json"
        if not path.is_file():
            raise NotImplementedCapability(
                f"ALFWorld frozen manifest missing: {path} — run "
                "scripts/prepare_alfworld_data.py (materialized by data prepare)"
            )
        return json.loads(path.read_text("utf-8"))

    def _walk_games(self) -> list[dict[str, Any]]:
        """Solve-checked game dirs: co-located traj_data.json + game.tw-pddl.

        Mirrors the official ``AlfredTWEnv.collect_game_files`` filter (skips
        movable/sliced trajectories).  Walks the official
        ``<root>/json_2.1.1/<split>/<task>/<trial>/`` layout with shallow
        per-task listings in a thread pool — sequential metadata walks take
        tens of minutes on loaded JuiceFS, and a full-tree rglob mis-reads the
        ``json_2.1.1`` wrapper as a split.
        """
        from concurrent.futures import ThreadPoolExecutor

        base = self._root / "json_2.1.1"
        if not base.is_dir():
            base = self._root

        def scan_task(task_dir: Path) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            split = task_dir.parent.name
            task_instance = str(task_dir.relative_to(base))
            if "movable" in task_instance or "Sliced" in task_instance:
                return out
            try:
                trials = [p for p in task_dir.iterdir() if p.is_dir()]
            except OSError:
                return out
            for trial in sorted(trials):
                if "movable" in trial.name or "Sliced" in trial.name:
                    continue
                traj = trial / "traj_data.json"
                game_file = trial / "game.tw-pddl"
                try:
                    if not (traj.is_file() and game_file.is_file()):
                        continue
                except OSError:
                    continue
                out.append(
                    {
                        "rel_dir": str(trial.relative_to(self._root)),
                        "split": split,
                        "task_instance": task_instance,
                        "trial": trial.name,
                        "traj_path": traj,
                        "game_path": game_file,
                    }
                )
            return out

        task_dirs: list[Path] = []
        for split_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if split_dir.name in ("rsi_outputs", "logic") or split_dir.name.startswith((".", "_")):
                continue
            task_dirs.extend(sorted(p for p in split_dir.iterdir() if p.is_dir()))
        games: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            for batch in pool.map(scan_task, task_dirs):
                games.extend(batch)
        return games

    # Load-time cache: every spawned panel worker calls public_tasks() → _load(),
    # and a cold _load() walks the full game tree plus ~5k traj reads on JuiceFS.
    # Eight workers doing that concurrently queue on metadata ops and stall the
    # whole panel (observed 2026-11-03 bwozi: 8 workers × 0% CPU × 1h, zero tasks).
    # The first process materializes the reconstructed game rows + traj payload
    # to this cache file; every later process reads it back instead of walking.
    # Keyed by the frozen manifest's tree digest so stale caches are ignored.
    CACHE_ENV = "RSI_ALFWORLD_LOAD_CACHE"
    CACHE_VERSION = 2

    def _load(self) -> None:
        if self._loaded:
            return
        import hashlib
        import json as _json
        import os as _os
        import tempfile

        manifest = self._frozen_manifest()
        cache_path = _os.environ.get(self.CACHE_ENV) or ""
        if not cache_path:
            cache_path = _os.path.join(tempfile.gettempdir(), "rsi_alfworld_load_cache.json")
        games: list[dict[str, Any]] = []
        traj_by_rel: dict[str, dict[str, Any]] = {}
        tree_digest = str(manifest.get("tree_digest") or "")
        cache_hit = False
        try:
            if _os.path.isfile(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    blob = _json.load(f)
                if (
                    blob.get("cache_version") == self.CACHE_VERSION
                    and blob.get("tree_digest") == tree_digest
                    and blob.get("alfworld_root") == str(self._root)
                ):
                    games = blob["games"]
                    traj_by_rel = blob["traj_by_rel"]
                    cache_hit = True
                    # stderr only: stdout carries the JSON payloads that entry
                    # scripts pipe through `python -c "json.load(sys.stdin)"`.
                    print(
                        f"[alfworld_provider] load cache hit: {len(games)} games from {cache_path}",
                        flush=True,
                        file=sys.stderr,
                    )
        except (OSError, ValueError, KeyError):
            cache_hit = False
        if not cache_hit:
            games = self._walk_games()
        if not games:
            raise NotImplementedCapability(
                f"no prepared ALFWorld games under {self._root} — run scripts/prepare_alfworld_data.py"
            )
        counts = dict(manifest.get("split_counts") or {})
        actual_counts: dict[str, int] = {}
        for g in games:
            actual_counts[g["split"]] = actual_counts.get(g["split"], 0) + 1
        for split, n in counts.items():
            if actual_counts.get(split, 0) != int(n):
                raise NotImplementedCapability(
                    f"ALFWorld split {split!r} count mismatch: manifest {n} != "
                    f"walked {actual_counts.get(split, 0)} — re-run data prepare"
                )

        # Preload traj metadata in parallel — ~4k small JuiceFS reads dominate
        # sequential task construction.
        if not cache_hit:
            from concurrent.futures import ThreadPoolExecutor

            def _read_traj(g: dict[str, Any]) -> dict[str, Any]:
                try:
                    return json.loads(Path(g["traj_path"]).read_text("utf-8"))
                except Exception:  # noqa: BLE001
                    return {}

            with ThreadPoolExecutor(max_workers=32) as pool:
                traj_by_rel = {g["rel_dir"]: t for g, t in zip(games, pool.map(_read_traj, games))}
            try:
                blob = {
                    "cache_version": self.CACHE_VERSION,
                    "tree_digest": tree_digest,
                    "alfworld_root": str(self._root),
                    "games": [{**g, "traj_path": str(g["traj_path"]), "game_path": str(g["game_path"])} for g in games],
                    "traj_by_rel": traj_by_rel,
                }
                tmp_path = cache_path + f".tmp.{_os.getpid()}"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    _json.dump(blob, f)
                _os.replace(tmp_path, cache_path)  # atomic: readers never see partial JSON
                print(
                    f"[alfworld_provider] load cache written: {len(games)} games -> {cache_path}",
                    flush=True,
                    file=sys.stderr,
                )
            except OSError:
                pass  # cache is best-effort; next process just walks again

        def make(g: dict[str, Any]) -> PublicTask:
            traj: dict[str, Any] = traj_by_rel.get(g["rel_dir"], {})
            from rsi.environments.alfworld_env import extract_instruction

            instruction = extract_instruction(traj) or f"(alfworld task {g['rel_dir']})"
            task_id = f"alfworld:{g['split']}/{g['rel_dir']}"
            task = PublicTask(
                task_id=task_id,
                domain=self.domain,
                instruction=instruction,
                group_id=f"alfworld:{g['task_instance']}",
                metadata={
                    "alfworld_game_file": str(Path(g["rel_dir"]) / "game.tw-pddl"),
                    "split": g["split"],
                    "task_type": str(traj.get("task_type") or ""),
                    "trial": g["trial"],
                },
            )
            self._by_id[task.task_id] = task
            self._private[task.task_id] = {"traj_data": traj, "game_file": task.metadata["alfworld_game_file"]}
            return task

        final_split = ""
        for cand in self.FINAL_SPLITS:
            if actual_counts.get(cand, 0) > 0:
                final_split = cand
                break
        transfer_split = ""
        for cand in self.TRANSFER_SPLITS:
            if cand != final_split and actual_counts.get(cand, 0) > 0:
                transfer_split = cand
                break
        final = [make(g) for g in games if g["split"] == final_split]
        if not final:
            raise NotImplementedCapability(
                f"ALFWorld final panel split not present under {self._root} "
                f"(tried {list(self.FINAL_SPLITS)})"
            )
        final_instances = {g["task_instance"] for g in games if g["split"] == final_split}
        # Same task instance never straddles two roles: panel instances are
        # dropped from every other split (same anti-leak rule as SWE-bench
        # family dedup) and from the sibling panel.
        transfer = (
            [
                make(g)
                for g in games
                if g["split"] == transfer_split and g["task_instance"] not in final_instances
            ]
            if transfer_split
            else []
        )
        panel_instances = final_instances | (
            {g["task_instance"] for g in games if g["split"] == transfer_split} if transfer_split else set()
        )

        by_group: dict[str, list[dict[str, Any]]] = {}
        for g in games:
            if g["split"] != "train":
                continue
            if g["task_instance"] in panel_instances:
                continue
            by_group.setdefault(g["task_instance"], []).append(g)
        buckets: dict[str, list[PublicTask]] = {k: [] for k in self.DEV_ROLES}
        for gid in sorted(by_group):
            h = int(hashlib.sha256(gid.encode()).hexdigest(), 16)
            buckets[self.DEV_ROLES[h % len(self.DEV_ROLES)]].extend(make(g) for g in by_group[gid])
        self._roles = {
            "work_train": buckets["work_train"],
            "harness_select": buckets["harness_select"],
            "weight_dev": buckets["weight_dev"],
            "confirm_h": buckets["confirm_h"],
            "confirm_w": buckets["confirm_w"],
            "final_test": final,
            "transfer_test": transfer,
        }
        n = self._panel_sizes
        self._roles = {k: (v[: n[k]] if n.get(k) else v) for k, v in self._roles.items()}
        self._loaded = True

    def materialize_task_id(self, tid: str) -> PublicTask:
        """Exact-id materialization for the frozen-manifest path (audit §8.2).

        Unknown ids are a ProtocolViolation, never a placeholder task."""
        self._load()
        try:
            return self._by_id[tid]
        except KeyError as exc:
            raise ProtocolViolation(
                f"alfworld task {tid!r} absent from the frozen ALFWorld manifest"
            ) from exc

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._roles.get(role, ()))

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def private_record(self, task_id: str) -> dict[str, Any]:
        self._load()
        return dict(self._private.get(task_id, {}))

    def groups(self) -> dict[str, tuple[str, ...]]:
        self._load()
        out: dict[str, list[str]] = {}
        for t in self._by_id.values():
            out.setdefault(t.group_id, []).append(t.task_id)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def all_tasks(self) -> tuple[PublicTask, ...]:
        self._load()
        return tuple(self._by_id.values())
