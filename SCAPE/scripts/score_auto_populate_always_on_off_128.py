#!/usr/bin/env python3
"""Audit and score auto-populate always-on/off K4/K8 shards."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def load(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def mean(xs):
    return sum(xs) / len(xs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    args = ap.parse_args()
    summaries, all_rows = [], {}
    for k in (4, 8):
        path = args.input_dir / 'shards' / f'NATURAL_FIRST_SEARCH_seed2230_K{k}.jsonl'
        rows = load(path)
        all_rows[k] = rows
        if len(rows) != 128:
            raise RuntimeError(f'K{k}: expected 128 rows, got {len(rows)}')
        trace_lengths_ok = all(len(r['branch_T_trace']) == k and len(r['branch_S_trace']) == k for r in rows)
        teacher_views_ok = all(all(x.get('view_policy') == 'full' for x in r['branch_T_trace'][1:]) for r in rows)
        student_views_ok = all(all(x.get('view_policy') == 'reduced' for x in r['branch_S_trace'][1:]) for r in rows)
        first_disagree = [float(r['teacher_action'] != r['student_action']) for r in rows]
        cost_delta = [float(r['branch_T_metrics']['tool_search_cost']) - float(r['branch_S_metrics']['tool_search_cost']) for r in rows]
        utility_delta = [float(r['utility_T']) - float(r['utility_S']) for r in rows]
        summaries.append({
            'horizon': f'K{k}', 'n': len(rows),
            'first_action_disagreement_rate': mean(first_disagree),
            'tool_cost_delta': mean(cost_delta), 'utility_delta': mean(utility_delta),
            'trace_lengths_ok': trace_lengths_ok,
            'teacher_always_on_views_ok': teacher_views_ok,
            'student_always_off_views_ok': student_views_ok,
            'snapshot_hash_sha256': hashlib.sha256('\n'.join(r['snapshot_hash'] for r in rows).encode()).hexdigest(),
        })
    ordered_same = [r['snapshot_hash'] for r in all_rows[4]] == [r['snapshot_hash'] for r in all_rows[8]]
    result = {
        'experiment': 'auto_populate_first_search_teacher_always_on_vs_student_always_off',
        'cohort': 'NATURAL_FIRST_SEARCH_seed2230_frozen_128',
        'first_action_counts_toward_horizon': True,
        'teacher_minus_student': True,
        'ordered_snapshot_hash_k4_k8_identical': ordered_same,
        'summaries': summaries,
    }
    if not ordered_same or not all(s['trace_lengths_ok'] and s['teacher_always_on_views_ok'] and s['student_always_off_views_ok'] for s in summaries):
        raise RuntimeError('formal audit failed')
    out = args.input_dir / 'AUTO_ALWAYS_ON_OFF_METRICS_SUMMARY.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
