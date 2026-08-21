#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')


def target_text(payload: dict[str, Any]) -> str:
    marker=str(payload.get('token_budget_marker') or '')
    used=int(payload.get('used_tokens_proxy') or 0)
    budget=int(payload.get('budget_proxy') or 30720)
    pct=used/max(1,budget)
    if pct >= 0.90:
        answer='Context budget is critical; stop expanding evidence now and answer from the curated/high-confidence evidence already visible.'
    elif pct >= 0.75:
        answer='Context budget is nearly exhausted; finish with the best supported answer and avoid any broad additional search.'
    else:
        answer='Context is over halfway through the budget; continue only if one targeted evidence step is essential, otherwise finish promptly.'
    return (
        f"{marker}\n"
        "to=end_search\n"
        "{\"answer\": " + json.dumps(answer, ensure_ascii=False) + "}\n"
        "</tool_call>"
    )


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--component-dir', type=Path, default=Path('/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker'))
    ap.add_argument('--split', type=int, default=4500)
    args=ap.parse_args()
    states=load_jsonl(args.component_dir/'TRAIN_STATES_5K.jsonl')
    if len(states)!=5000:
        raise SystemExit(f'expected 5000 states, got {len(states)}')
    rows=[]
    for idx,state in enumerate(states):
        payload=state.get('event_payload_student_visible') or {}
        prompt=(state.get('student_visible_prefix') or '')+'\n\nToken budget signal:\n'+str(payload.get('token_budget_marker') or '')+'\n'
        response=target_text(payload)
        rows.append({
            'row_id': f"token_budget_marker_{idx:05d}",
            'component':'token_budget_marker',
            'state_uid':state.get('state_uid'),
            'query_id':state.get('query_id'),
            'rollout_id':state.get('rollout_id'),
            'prompt':prompt,
            'prompt_full':prompt,
            'prompt_reduced':prompt,
            'response_text':response,
            'teacher_response':response,
            'loss_path':'full_response_kl',
            'collector_mode':state.get('collector_mode'),
            'synthetic_fallback':False,
            'token_budget_marker':payload.get('token_budget_marker'),
            'used_tokens_proxy':payload.get('used_tokens_proxy'),
            'budget_proxy':payload.get('budget_proxy'),
        })
    train=rows[:args.split]
    valid=rows[args.split:]
    write_jsonl(args.component_dir/'OPD_TRAIN_ROWS.jsonl', train)
    write_jsonl(args.component_dir/'OPD_VALID_ROWS.jsonl', valid)
    manifest={
        'status':'ready',
        'component':'token_budget_marker',
        'train_rows':len(train),
        'valid_rows':len(valid),
        'unique_state_uid':len({r['state_uid'] for r in rows}),
        'loss_path':'full_response_kl',
        'collector_mode':'real_harness1',
        'synthetic_fallback':False,
    }
    (args.component_dir/'OPD_ROWS_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
