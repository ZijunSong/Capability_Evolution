"""Coalition add-pair + collection plumbing tests (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.probes.probe_c_collect import assign_splits, select_decision_points
from trim.probes.probe_c_contract import (
    ContractError,
    load_config,
    make_add_pair,
)


def test_coalition_add_pair_diff_is_full_member_set():
    pair = make_add_pair(
        [
            "subtractive_curation",
            "importance_tagging",
            "adaptive_rerank_instruction",
        ],
        git_sha="g1",
    )
    assert set(pair.diff_components()) == {
        "subtractive_curation",
        "importance_tagging",
        "adaptive_rerank_instruction",
    }
    assert all(v is False for v in pair.student_mask.values())
    on = sorted(k for k, v in pair.teacher_mask.items() if v)
    assert on == [
        "adaptive_rerank_instruction",
        "importance_tagging",
        "subtractive_curation",
    ]


def test_config_coalition_members(tmp_path: Path):
    cfg = {
        "schema_version": "probe_c_v3",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "rev-1",
        "pair_mode": "add_from_h0",
        "components": ["esp", "msc"],
        "coalition_members": {
            "esp": ["subtractive_curation", "importance_tagging", "adaptive_rerank_instruction"],
            "msc": ["verify_tool", "token_budget_marker"],
        },
        "state_source": "real_h0_rollout",
        "projection_mode": "direct_only",
        "probe_objective": "frozen_projected_gap",
        "loss_scale": 1.0,
        "splits": {"frozen": True},
        "git_sha": "g1",
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = load_config(p)
    assert loaded.members_of("esp") == (
        "subtractive_curation",
        "importance_tagging",
        "adaptive_rerank_instruction",
    )
    assert loaded.members_of("msc") == ("verify_tool", "token_budget_marker")
    # single-component fallback
    assert loaded.members_of("unknown") == ("unknown",)

    bad = dict(cfg, coalition_members={"esp": ["not_a_component"]})
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ContractError, match="unknown members"):
        load_config(p2)


def test_assign_splits_disjoint_and_deterministic():
    ids = [f"q{i}" for i in range(20)]
    a = assign_splits(ids, n_train=8, n_select=6, n_audit=4, seed=7)
    b = assign_splits(list(reversed(ids)), n_train=8, n_select=6, n_audit=4, seed=7)
    assert a == b  # order-independent
    from collections import Counter

    counts = Counter(a.values())
    assert counts["C_train"] == 8
    assert counts["C_select"] == 6
    assert counts["C_audit"] == 4
    assert counts[""] == 2  # remainder not collected
    # each query appears in exactly one split
    assert len(a) == 20


class _Point:
    def __init__(self, turn: int):
        self.turn_id = turn


def test_select_decision_points_caps_at_4():
    pts = [_Point(t) for t in range(12)]
    kept = select_decision_points(pts)
    assert len(kept) == 4
    assert kept[0].turn_id == 0
    assert kept[-1].turn_id == 11
    small = select_decision_points([_Point(t) for t in range(3)])
    assert len(small) == 3


def test_cache_build_accepts_coalition_executor():
    """Coalition executor keeps real events and reports unregistered members."""
    from trim.probes.probe_c_cache import build_real_teacher_cache
    from trim.probes.probe_c_data import validate_state_row
    from trim.probes.probe_c_cache import TeacherBranchOutcome
    from tests.probe_c_fixtures import make_state_row

    pair = make_add_pair(["verify_tool", "token_budget_marker"], git_sha="g1")
    state = validate_state_row(
        make_state_row(student_mask=pair.student_mask, teacher_mask=pair.teacher_mask),
        pair=pair,
    )

    def executor(s, p):
        return TeacherBranchOutcome(
            triggered=True,
            source_type="capability_effect",
            events=[object()],
            projection_kind="direct",
            projected_steps=[
                type(
                    "Step",
                    (),
                    {"target_action": {"name": "search_corpus", "arguments": {"query": "q"}}},
                )()
            ],
            realizable=True,
            detail={"members": [{"component_id": "verify_tool", "source_type": "capability_effect"}]},
        )

    rows, funnel = build_real_teacher_cache(
        [state],
        pair,
        component_id="msc",
        executor=executor,
        encoder=lambda s, a: ([1, 2], [True, True]),
        teacher_logp_fn=lambda prompt, target: [-0.1] * len(target),
    )
    assert len(rows) == 1
    assert rows[0].funnel.detail["members"][0]["component_id"] == "verify_tool"
    assert funnel.status(min_covered_queries=1) == "READY"


def test_make_candidate_set_handles_teacher_target_objects(monkeypatch):
    """CachedSupervisionRow.target is a TeacherTarget object, not a dict."""
    from trim.probes.probe_c_cache import CachedSupervisionRow, TeacherTarget
    from trim.probes.probe_c_data import validate_state_row
    from trim.probes import probe_c_qwen35_hooks as hooks
    from tests.probe_c_fixtures import make_state_row

    pair = make_add_pair(["verify_tool", "token_budget_marker"], git_sha="g1")
    state = validate_state_row(
        make_state_row(
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
            original_action_parse_result={
                "structurally_valid": True,
                "action_text": 'to=search_corpus\n{"query": "capital of france"}\n',
            },
        ),
        pair=pair,
    )
    target = TeacherTarget(
        cache_key="k",
        state_id=state.state_id,
        query_id=state.query_id,
        component_id="msc",
        target_action={"name": "curate", "arguments": {"add_ids": ["d1"]}},
        target_action_ids=[10, 11],
        teacher_token_logprobs=[-0.1, -0.2],
        weights=[1.0, 1.0],
        student_prompt_ids_hash="ph",
    )
    row = CachedSupervisionRow(state=state, target=target)

    monkeypatch.setattr(hooks, "_encode_action", lambda cfg, action: [1, 2, 3])
    monkeypatch.setattr(hooks, "load_encoder", lambda cfg: object())
    import json as _json
    cfg_raw = _json.loads(
        (
            __import__("pathlib").Path("configs/probes/probe_c_qwen35_9b_coalitions.json")
        ).read_text(encoding="utf-8")
    )
    from trim.probes.probe_c_contract import load_config
    from pathlib import Path as _P
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _P(td) / "c.json"
        cfg_raw["model_revision"] = "rev-1"
        cfg_raw["git_sha"] = "g1"
        cfg_raw["model_path"] = "/mnt/songzijun/models/Qwen3.5-9B"
        p.write_text(_json.dumps(cfg_raw), encoding="utf-8")
        cfg = load_config(p)
        fn = hooks.make_candidate_set(cfg, candidate_limit=8)
        kset = fn(state, pair, [row])

    ids = [cid for cid, _ in kset]
    assert "proj" in ids  # TeacherTarget object path works
    assert "base" in ids  # parsed base action included
    names = {a["name"] for _, a in kset}
    assert "curate" in names
    assert all(a.get("token_ids") for _, a in kset)


def test_candidate_set_no_duplicate_ids_across_states(monkeypatch):
    """K(s) is per-state: other states' proj targets must not collide in ids."""
    from trim.probes.probe_c_cache import CachedSupervisionRow, TeacherTarget
    from trim.probes.probe_c_data import validate_state_row
    from trim.probes import probe_c_qwen35_hooks as hooks
    from tests.probe_c_fixtures import make_state_row

    pair = make_add_pair(["verify_tool", "token_budget_marker"], git_sha="g1")
    states = []
    rows = []
    for i in range(2):
        st = validate_state_row(
            make_state_row(
                query_id=f"q{i}",
                student_mask=pair.student_mask,
                teacher_mask=pair.teacher_mask,
            ),
            pair=pair,
        )
        states.append(st)
        tgt = TeacherTarget(
            cache_key=f"k{i}",
            state_id=st.state_id,
            query_id=st.query_id,
            component_id="msc",
            target_action={"name": "curate", "arguments": {"add_ids": [f"d{i}"]}},
            target_action_ids=[10 + i],
            teacher_token_logprobs=[-0.1],
            weights=[1.0],
            student_prompt_ids_hash="ph",
        )
        rows.append(CachedSupervisionRow(state=st, target=tgt))

    monkeypatch.setattr(hooks, "_encode_action", lambda cfg, action: [1, 2, 3])
    monkeypatch.setattr(hooks, "load_encoder", lambda cfg: object())
    import json as _json
    import tempfile

    from trim.probes.probe_c_contract import load_config

    cfg_raw = _json.loads(
        open("configs/probes/probe_c_qwen35_9b_coalitions.json", encoding="utf-8").read()
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        cfg_raw["model_revision"] = "rev-1"
        cfg_raw["git_sha"] = "g1"
        cfg_raw["model_path"] = "/mnt/songzijun/models/Qwen3.5-9B"
        p.write_text(_json.dumps(cfg_raw), encoding="utf-8")
        cfg = load_config(p)
        fn = hooks.make_candidate_set(cfg, candidate_limit=8)
        kset = fn(states[0], pair, rows)

    ids = [cid for cid, _ in kset]
    assert len(ids) == len(set(ids)), ids
    proj_args = [a["arguments"].get("add_ids") for cid, a in kset if cid.startswith("proj")]
    assert proj_args == [["d0"]]  # only this state's target
    from trim.eval.probe_c_metrics import score_candidates  # noqa: F401  # id contract holds
    score_candidates(
        state_id=states[0].state_id,
        query_id=states[0].query_id,
        candidate_ids=ids,
        z_teacher={i: -1.0 for i in ids},
        z_student={i: -1.0 for i in ids},
    )


def test_regenerate_fn_unwraps_student_backend(monkeypatch):
    """regenerate_fn may receive the ProbeCLoraStudentBackend wrapper."""
    import types

    from trim.probes.probe_c_data import validate_state_row
    from trim.probes import probe_c_qwen35_hooks as hooks
    from tests.probe_c_fixtures import make_state_row

    pair = make_add_pair(["verify_tool", "token_budget_marker"], git_sha="g1")
    state = validate_state_row(
        make_state_row(student_mask=pair.student_mask, teacher_mask=pair.teacher_mask),
        pair=pair,
    )

    class _Inner:
        _device = "cpu"

        class model:
            @staticmethod
            def generate(inp, **kw):
                import torch

                # prompt ids (5) + 3 completion tokens
                return torch.tensor([[1, 2, 3, 4, 5, 80603, 77, 99]])

        model = model()

    class _Wrapper:
        backend = _Inner()

    class _Enc:
        stop_token_ids = [99]

        @staticmethod
        def decode_tokens(ids):
            return 'to=search_corpus\n{"query": "q"}\n'

        @staticmethod
        def parse_tool_call(text, completion_ids=None):
            return types.SimpleNamespace(
                parsed=True, tool_name="search_corpus", arguments={"query": "q"}
            )

    monkeypatch.setattr(hooks, "load_encoder_from_backend", lambda b: _Enc())
    rec = hooks.regenerate_fn(_Wrapper(), state, pair, checkpoint="pre")
    assert rec.ok is True, rec.detail
    assert rec.action == {"name": "search_corpus", "arguments": {"query": "q"}}
