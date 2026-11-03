"""WebShop / ALFWorld task-provider contract tests (no heavy official deps).

Fixtures mirror the frozen artifacts written by ``scripts/prepare_webshop_data.py``
and ``scripts/prepare_alfworld_data.py``.  The invariants under test are the
task-book ones: deterministic role assignment, group disjointness across roles,
panel-asin / panel-instance anti-leak dedup, and private gold never entering
``PublicTask``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.contracts.errors import NotImplementedCapability
from rsi.contracts.task import SplitAssignment, SplitManifest
from rsi.environments.task_providers import AlfWorldTaskProvider, WebShopTaskProvider

WEB_ROLES = ["work_train", "harness_select", "weight_dev", "confirm_h", "confirm_w",
             "final_test", "transfer_test"]


def _write_webshop_frozen(root: Path, goals, panel=None) -> None:
    payload = {
        "schema_version": "rsi_webshop_goals_v1",
        "goals": goals,
        "product_prices": {g["asin"]: 9.99 for g in goals},
    }
    goals_path = root / "frozen_goals.json"
    goals_path.write_text(json.dumps(payload), "utf-8")
    from rsi.contracts.common import digest_of_file

    manifest = {
        "schema_version": "rsi_webshop_frozen_v1",
        "flavor": "small",
        "items_file": "data/items_shuffle_1000.json",
        "goals_file": "frozen_goals.json",
        "goals_sha256": digest_of_file(str(goals_path)),
        "panel": panel or {"final_test_count": 4, "transfer_test_count": 2},
    }
    (root / "frozen_manifest.json").write_text(json.dumps(manifest), "utf-8")


def _goal(i: int, asin: str) -> dict:
    return {
        "asin": asin,
        "category": "c",
        "query": "q",
        "name": f"product {asin}",
        "product_category": "pc",
        "instruction_text": f"instruction for goal {i} on {asin}",
        "attributes": ["blue"],
        "price_upper": 50.0,
        "goal_options": {"color": "blue"},
        "weight": 1,
    }


def test_webshop_roles_and_asin_dedup(tmp_path):
    # 3 goals share asin A (2 in the official final panel head, 1 later);
    # the later one must be dropped, not leaked into a dev role.
    goals = [_goal(0, "A"), _goal(1, "B"), _goal(2, "A"), _goal(3, "C"),
             _goal(4, "D"), _goal(5, "E"), _goal(6, "F"), _goal(7, "G"),
             _goal(8, "H"), _goal(9, "I"), _goal(10, "J")]
    _write_webshop_frozen(tmp_path, goals)
    p = WebShopTaskProvider(str(tmp_path))
    final = p.public_tasks("final_test")
    assert [t.metadata["webshop_goal_index"] for t in final] == [0, 1, 2, 3]
    transfer = p.public_tasks("transfer_test")
    assert len(transfer) == 2
    # goal 2 (asin A) is inside the final panel; the OTHER asin-A goals must
    # appear nowhere else (anti-leak).  Here only index 0..3 exist for A.
    all_roles = {r: [t.task_id for t in p.public_tasks(r)] for r in WEB_ROLES}
    seen = [tid for ids in all_roles.values() for tid in ids]
    assert len(seen) == len(set(seen)), "task ids must be unique across roles"
    # group disjointness enforced by the SplitManifest contract
    manifest = SplitManifest(
        manifest_id="t",
        domain="webshop",
        assignments=tuple(
            SplitAssignment(role=r, task_ids=tuple(all_roles[r])) for r in WEB_ROLES if all_roles[r]
        ),
        groups=p.groups(),
    )
    manifest.check_group_disjoint()
    # transfer and dev roles come only from indices >= 4
    for role in WEB_ROLES[:5] + ["transfer_test"]:
        for t in p.public_tasks(role):
            assert t.metadata["webshop_goal_index"] >= 4


def test_webshop_final_panel_is_head_of_frozen_goals(tmp_path):
    goals = [_goal(i, f"ASIN{i:03d}") for i in range(12)]
    _write_webshop_frozen(tmp_path, goals, panel={"final_test_count": 4, "transfer_test_count": 2})
    p = WebShopTaskProvider(str(tmp_path))
    final_idx = [t.metadata["webshop_goal_index"] for t in p.public_tasks("final_test")]
    assert final_idx == [0, 1, 2, 3]  # official reset(session=i) indices


def test_webshop_private_gold_not_public(tmp_path):
    goals = [_goal(i, f"ASIN{i:03d}") for i in range(8)]
    _write_webshop_frozen(tmp_path, goals)
    p = WebShopTaskProvider(str(tmp_path))
    t = p.public_tasks("final_test")[0]
    priv = p.private_record(t.task_id)
    assert priv["attributes"] == ["blue"] and priv["price_upper"] == 50.0
    blob = json.dumps({"i": t.instruction, "m": dict(t.metadata)})
    assert "blue" not in blob and "price_upper" not in blob


def test_webshop_missing_frozen_is_loud(tmp_path):
    with pytest.raises(NotImplementedCapability):
        WebShopTaskProvider(str(tmp_path)).public_tasks("work_train")


def test_webshop_panel_sizes_truncate(tmp_path):
    goals = [_goal(i, f"ASIN{i:03d}") for i in range(30)]
    _write_webshop_frozen(tmp_path, goals, panel={"final_test_count": 6, "transfer_test_count": 4})
    p = WebShopTaskProvider(str(tmp_path), panel_sizes={"final_test": 3, "work_train": 2})
    assert len(p.public_tasks("final_test")) == 3
    assert len(p.public_tasks("work_train")) == 2


def _write_alfworld_tree(root: Path) -> None:
    def game(split: str, inst: str, trial: str, desc: str) -> None:
        d = root / split / inst / trial
        d.mkdir(parents=True)
        (d / "traj_data.json").write_text(
            json.dumps({"task_type": "pick_and_place_simple", "task_desc": desc}), "utf-8"
        )
        (d / "game.tw-pddl").write_text(json.dumps({"solvable": True}), "utf-8")

    for i in range(6):
        game("train", f"inst{i}", "trial_T1", f"train goal {i}")
    # same instance as inst0 but in the final panel — inst0 must vanish from dev
    game("test_unseen", "inst0", "trial_T2", "panel goal")
    game("test_unseen", "inst9", "trial_T1", "panel goal 9")
    game("out_of_distribution", "inst8", "trial_T1", "ood goal")
    # missing game file is ignored by the walker
    d = root / "train" / "broken" / "trial_T1"
    d.mkdir(parents=True)
    (d / "traj_data.json").write_text("{}", "utf-8")
    # counts are over post-filter games (co-located traj_data.json + game.tw-pddl)
    counts = {"train": 6, "test_unseen": 2, "out_of_distribution": 1}
    (root / "frozen_manifest.json").write_text(
        json.dumps({"schema_version": "rsi_alfworld_frozen_v1", "split_counts": counts}), "utf-8"
    )


def test_alfworld_roles_and_instance_dedup(tmp_path):
    _write_alfworld_tree(tmp_path)
    p = AlfWorldTaskProvider(str(tmp_path))
    final = p.public_tasks("final_test")
    transfer = p.public_tasks("transfer_test")
    assert len(final) == 2 and len(transfer) == 1
    # inst0 appears in the final panel → its train trial must be dropped
    all_dev = [t for r in WEB_ROLES[:5] for t in p.public_tasks(r)]
    assert all(t.group_id != "alfworld:inst0" for t in all_dev)
    ids = [t.task_id for t in all_dev + list(final) + list(transfer)]
    assert len(ids) == len(set(ids))
    manifest = SplitManifest(
        manifest_id="t",
        domain="alfworld",
        assignments=tuple(
            (SplitAssignment(role=r, task_ids=tuple(t.task_id for t in p.public_tasks(r))))
            for r in WEB_ROLES
            if p.public_tasks(r)
        ),
        groups=p.groups(),
    )
    manifest.check_group_disjoint()


def test_alfworld_instruction_from_traj(tmp_path):
    _write_alfworld_tree(tmp_path)
    p = AlfWorldTaskProvider(str(tmp_path))
    t = p.public_tasks("final_test")[0]
    assert t.instruction.startswith("panel goal") or t.instruction.startswith("train goal")
    assert t.metadata["alfworld_game_file"].endswith("game.tw-pddl")


def test_alfworld_missing_manifest_is_loud(tmp_path):
    with pytest.raises(NotImplementedCapability):
        AlfWorldTaskProvider(str(tmp_path)).public_tasks("work_train")


def test_alfworld_count_mismatch_is_loud(tmp_path):
    _write_alfworld_tree(tmp_path)
    m = json.loads((tmp_path / "frozen_manifest.json").read_text("utf-8"))
    m["split_counts"]["train"] = 99
    (tmp_path / "frozen_manifest.json").write_text(json.dumps(m), "utf-8")
    with pytest.raises(NotImplementedCapability):
        AlfWorldTaskProvider(str(tmp_path)).public_tasks("work_train")
