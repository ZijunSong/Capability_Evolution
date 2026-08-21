#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path('/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def token_budget_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usages = []
    markers = []
    budgets = []
    for row in rows:
        payload = row.get('event_payload_student_visible') or {}
        usages.append(int(payload.get('used_tokens_proxy') or 0))
        budgets.append(int(payload.get('budget_proxy') or 30720))
        markers.append(str(payload.get('token_budget_marker') or ''))
    bins = Counter()
    for u, b in zip(usages, budgets):
        pct = 100.0 * u / max(1, b)
        if pct >= 90:
            bins['critical_90_plus'] += 1
        elif pct >= 75:
            bins['warning_75_90'] += 1
        elif pct >= 60:
            bins['over_half_60_75'] += 1
        else:
            bins['low_under_60'] += 1
    actionable = bins['critical_90_plus'] + bins['warning_75_90'] + bins['over_half_60_75']
    n = len(rows)
    positive = actionable > 0
    return {
        'component': 'token_budget_marker',
        'status': 'TEACHER_COMPONENT_NO_POSITIVE_UTILITY' if not positive else 'TEACHER_DIAGNOSTIC_POSITIVE_UTILITY_PROXY',
        'paper_grade_reward': False,
        'metric_scope': 'teacher_before_diagnostic_gate',
        'n_states': n,
        'budget_proxy_values': sorted(set(budgets)),
        'used_tokens_proxy_min': min(usages) if usages else None,
        'used_tokens_proxy_max': max(usages) if usages else None,
        'teacher': {
            'overall_reward': 'N/A',
            'diagnostic_marker_present_rate': sum(bool(m) for m in markers) / max(1, n),
            'diagnostic_actionable_marker_rate': actionable / max(1, n),
            'termination_timing_signal': 'positive_proxy' if positive else 'no_actionable_budget_pressure_observed',
        },
        'student_before': {
            'overall_reward': 'N/A',
            'student_inference_privilege': False,
            'marker_available': False,
        },
        'usage_bins': dict(bins),
        'decision': 'TEACHER_COMPONENT_NO_POSITIVE_UTILITY' if not positive else 'TEACHER_POSITIVE_PROXY_REQUIRES_FORMAL_REWARD',
        'reason': 'All frozen event states have token usage below 60% of budget; marker is present but has no observed termination-pressure utility in this dataset.' if not positive else 'Some states contain budget-pressure markers; formal reward evaluation is still required before Phase E.',
    }


def verify_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatch = 0
    for row in rows:
        payload = row.get('event_payload_student_visible') or {}
        if payload.get('available_teacher_action') == 'verify(doc_ids, claim)' and payload.get('student_action_space_has_verify') is False:
            mismatch += 1
    n = len(rows)
    return {
        'component': 'verify_tool',
        'status': 'NON_REALIZABLE_ACTION_SPACE_MISMATCH',
        'paper_grade_reward': False,
        'metric_scope': 'teacher_before_diagnostic_gate',
        'n_states': n,
        'teacher': {
            'overall_reward': 'N/A',
            'verify_action_available_rate': mismatch / max(1, n),
        },
        'student_before': {
            'overall_reward': 'N/A',
            'student_has_verify_tool': False,
        },
        'decision': 'NON_REALIZABLE_ACTION_SPACE_MISMATCH',
        'reason': 'Teacher action space includes verify(doc_ids, claim), while Student action space does not; Student After OPD is N/A.',
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--token-component-dir', type=Path, default=None)
    parser.add_argument('--verify-component-dir', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    token_dir = args.token_component_dir or (root / 'token_budget_marker')
    verify_dir = args.verify_component_dir or (root / 'verify_tool')
    token_rows = read_jsonl(token_dir / 'TRAIN_STATES_5K.jsonl')
    verify_rows = read_jsonl(verify_dir / 'TRAIN_STATES_5K.jsonl')
    if len(token_rows) != 5000 or len(verify_rows) != 5000:
        raise SystemExit('DIAGNOSTIC_REQUIRES_5K_STATES_PER_COMPONENT')
    token = token_budget_gate(token_rows)
    verify = verify_gate(verify_rows)
    write_json(token_dir / 'TEACHER_BEFORE_DIAGNOSTIC.json', token)
    write_json(verify_dir / 'TEACHER_BEFORE_DIAGNOSTIC.json', verify)
    output = args.output or (root / 'H1004_TEACHER_BEFORE_DIAGNOSTICS.json')
    write_json(output, {'status': 'H1004_TEACHER_BEFORE_DIAGNOSTICS_READY', 'components': [token, verify]})
    print(json.dumps({'status': 'H1004_TEACHER_BEFORE_DIAGNOSTICS_READY', 'components': [token, verify]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
