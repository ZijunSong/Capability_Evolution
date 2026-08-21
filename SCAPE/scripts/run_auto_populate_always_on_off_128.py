#!/usr/bin/env python3
"""Run auto-populate Teacher-always-on vs Student-always-off paired fork."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path('/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/scripts')
sys.path.insert(0, str(SRC))
import run_h1002_verify_value_confirm as verify

verify.COMPONENT = 'auto_populate_first_search'
verify.STRATA = ('NATURAL_FIRST_SEARCH', 'AUTO_EFFECT_ACTIVE')
verify.SEEDS = (2230, 2231)
verify.HORIZONS = (4, 8)
base = verify.base
_original_run_branch = base.run_branch

def _always_on_run_branch(start, first_action, *, k, scorer, renderer, component, label, replay_jitter=''):
    """Teacher T stays full-view; Student S/N stays reduced-view."""
    st = start.clone(label)
    trace = []
    st.execute(first_action)
    trace.append({'branch': label, 'phase': 'forced_first', 'action': dict(first_action), 'metrics': st.metrics()})
    full = label == 'T'
    for i in range(k):
        action, dist, dual = base.policy_action(
            st, scorer, renderer, component=component, full=full,
            tie_jitter=f'{replay_jitter}:{i}' if replay_jitter else '')
        st.execute(action)
        trace.append({'branch': label, 'phase': f'continue_{i+1}', 'action': action,
                      'top_prob': max(dist['tool_name_probs'].values()),
                      'snapshot_hash': dual['snapshot_hash'], 'metrics': st.metrics(),
                      'view_policy': 'full' if full else 'reduced'})
    return st, trace

base.run_branch = _always_on_run_branch

# Reuse the verified state-manifest and cell machinery, then relabel the contract.
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--stratum', default='NATURAL_FIRST_SEARCH')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--horizon', '--K', type=int, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--browsecomp-root', type=Path, default=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus'))
    p.add_argument('--index-path', type=Path, default=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25'))
    p.add_argument('--model', default='/mnt/songzijun/models/pat-jj_harness-1-full/harness-1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--max-prompt-tokens', type=int, default=3072)
    p.add_argument('--n-states', type=int, default=128)
    args = p.parse_args()
    if args.stratum != 'NATURAL_FIRST_SEARCH':
        raise SystemExit('always-on/off formal run requires --stratum NATURAL_FIRST_SEARCH')
    # verify.run_cell uses k=--horizon as continuation count; adjust to total-horizon semantics.
    original_run_cell = verify.run_cell
    def run_total_horizon(ns):
        old = ns.horizon
        ns.horizon = max(0, int(old) - 1)
        try:
            return original_run_cell(ns)
        finally:
            ns.horizon = old
    rc = run_total_horizon(args)
    if rc != 0:
        return rc
    shard = args.out_dir / 'shards' / f'{args.stratum}_seed{args.seed}_K{args.horizon - 1}.jsonl'
    target_shard = args.out_dir / 'shards' / f'{args.stratum}_seed{args.seed}_K{args.horizon}.jsonl'
    rows = []
    for line in shard.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        row['experiment'] = 'AUTO_ALWAYS_ON_OFF_128'
        row['continuation_policy'] = 'teacher_full_always_on_student_reduced_always_off'
        row['always_on_semantics'] = True
        rows.append(row)
    target_shard.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
