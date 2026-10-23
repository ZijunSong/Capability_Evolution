"""test_standalone_install / test_no_legacy_runtime_dependency.

Only the RSI tree is copied into an isolated directory with PYTHONPATH
cleared; sibling research directories are absent.  Any import of the legacy
TRIM / Harness-1 / SCAPE runtimes therefore fails loudly.  A toy rollout, a
real evolve-fixture materialization and one real tiny-model update must run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RSI_ROOT = Path(__file__).resolve().parents[1]
SRC = RSI_ROOT / "src"
LEGACY_TOPS = {"trim", "scape", "scope", "harness1", "easyopd"}
LEGACY_PATH_TOKENS = ("/TRIM/", "/SCAPE/", "/SCOPE/", "Harness-1", "harness1")

_ISOLATED_RUN = r'''
import json, sys
sys.path.insert(0, "src")
out = {"steps": []}
import rsi, rsi.contracts, rsi.runtime, rsi.harness, rsi.evolution, rsi.opd, rsi.training, rsi.campaign
out["steps"].append("imports")

# no legacy modules may be loaded
legacy = [m for m in sys.modules if m.split(".")[0] in {"trim","scape","scope"} or m.startswith("harness1")]
assert not legacy, legacy

from rsi.environments.toy import ToyFileWorld, build_toy_file_tasks
from rsi.harness.loader import load_empty_bank
from rsi.runtime.loop import MinimalLoop
from rsi.runtime.budgets import BudgetSpec
from rsi.providers.scripted import ScriptedPolicy
from rsi.contracts.action import ToolCall, Finish
import re
_OBS = re.compile(r"\[\d+\] observation: (.*)")
def plan(view, step):
    obs = _OBS.findall(view)[1:]
    n = len(obs)
    if n == 0: return ToolCall(name="read_file", arguments={"path": "a.txt"})
    if n == 1: return ToolCall(name="read_file", arguments={"path": "b.txt"})
    return Finish(answer=None)
task, expected = build_toy_file_tasks(1, 17)[0]
env = ToyFileWorld()
loop = MinimalLoop(env=env, policy=ScriptedPolicy(plan), bank=load_empty_bank(), budgets=BudgetSpec(max_actions=6))
res = loop.run_episode(task, seed=1)
assert res.trace.terminated_reason == "finish"
out["steps"].append("toy_rollout")

# evolve fixture: real code materialization + execution
from rsi.evolution.materializer import materialize_candidate
from rsi.evolution.validator import validate_materialized
from rsi.contracts.artifact import CandidateFile, CandidatePatch
from pathlib import Path
import sys
sys.path.insert(0, "tests")
from helpers import toy_finalizer_patch
empty = load_empty_bank()
patch = toy_finalizer_patch(empty.bank_digest)
mat = materialize_candidate(patch, parent_bank_root=empty.root, parent_bank_digest=empty.bank_digest, out_root=Path("cands"))
rep = validate_materialized(mat.bank_dir)
assert rep.ok, rep.problems
out["steps"].append("evolve_fixture")

# one real tiny-model optimizer update (RL+OPD joint step)
import torch
from rsi.providers.tiny_hf import build_tiny_model
from rsi.training.hf_reference import HFReferenceBackend
from rsi.training.joint import MethodSpec, joint_update
from rsi.opd.records import RLRow, OPDRow
rl = RLRow(row_id="r", episode_id="e", decision_index=0,
           prompt_token_ids=(256,65,66), action_token_ids=(67,68),
           old_logprobs=(-1.0,-1.0), action_mask=(1,1), advantage=0.5,
           reward_ref="t", behavior_policy_version="p")
opd = OPDRow(row_id="o", episode_id="e", decision_index=0,
             student_prompt_token_ids=(256,65,66), teacher_prompt_token_ids=(256,66,67),
             target_token_ids=(67,68), target_mask=(1,1), target_action=None,
             target_kind="native", alignment_route="native_anchor", source_event_ids=(),
             realizability_json={}, provenance_json={}, token_weight=(1.0,1.0),
             student_logprobs=(-1.0,-1.0), teacher_logprobs=(-0.2,-0.2))
model = build_tiny_model()
before = {k: v.clone() for k, v in model.state_dict().items()}
backend = HFReferenceBackend(model=model, lr=1e-2)
report = joint_update(backend, rl_rows=[rl], opd_rows=[opd], spec=MethodSpec.resolve("harness_opd_rl"), lambda_opd=0.1)
assert report.n_optimizer_steps == 1
changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
assert changed
out["steps"].append("tiny_model_update")
print(json.dumps(out))
'''


def _static_no_legacy_scan() -> list[str]:
    problems = []
    str_lit = re.compile(r"('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\")")
    docstring = re.compile(r"\"\"\".*?\"\"\"", re.DOTALL)
    for py in SRC.rglob("*.py"):
        text = py.read_text("utf-8")
        # Documentation/denylist string literals may name the legacy runtimes
        # ("documentation references are not runtime deps"); strip docstrings
        # and inline string literals, then scan the remaining code.
        text = docstring.sub("''", text)
        for i, line in enumerate(text.splitlines(), 1):
            m = re.match(r"\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)", line)
            if m and m.group(1).split(".")[0] in LEGACY_TOPS:
                problems.append(f"{py}:{i}: legacy import {m.group(1)}")
            if "sys.path" in line and "insert" in line:
                problems.append(f"{py}:{i}: sys.path manipulation in package code")
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            code_only = str_lit.sub("''", line)
            for tok in LEGACY_PATH_TOKENS:
                if tok in code_only:
                    problems.append(f"{py}:{i}: legacy path token {tok}")
    return problems


def test_no_legacy_runtime_dependency():
    problems = _static_no_legacy_scan()
    assert not problems, problems
    # dependency graph of the running process must not include legacy tops
    import rsi.campaign.controller  # noqa: F401
    import rsi.training.service  # noqa: F401

    loaded = [m for m in sys.modules if m.split(".")[0] in LEGACY_TOPS]
    assert not loaded


def test_standalone_install(tmp_path):
    """Copy ONLY the RSI tree, clear PYTHONPATH, run rollout + evolve + update."""
    dest = tmp_path / "RSI_copy"
    shutil.copytree(
        RSI_ROOT,
        dest,
        ignore=shutil.ignore_patterns(
            "runs", "*.egg-info", "__pycache__", ".pytest_cache", "cands", "data"
        ),
    )
    # Sibling research directories must not exist in the isolated view.
    assert not (tmp_path / "TRIM").exists()
    assert not (tmp_path / "SCAPE").exists()

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATED_RUN],
        cwd=str(dest),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["steps"] == ["imports", "toy_rollout", "evolve_fixture", "tiny_model_update"]
