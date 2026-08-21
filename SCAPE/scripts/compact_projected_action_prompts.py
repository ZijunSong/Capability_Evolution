#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0818_projected_action_auto')
TOOLS=['fan_out_search','search_corpus','grep_corpus','read_document','review_docs','curate','verify','end_search']

def compact_state(state, full=False):
    wm=state.get('working_memory') or {}
    docs=[{'id':str(d.get('id')),'text':str(d.get('text') or '')[:256]} for d in (wm.get('documents') or [])[:12]]
    payload={'task':'Choose exactly one next BrowseComp tool call as JSON.','query_id':str(state.get('query_id')),'step':int(state.get('step',0) or 0),'available_tools':TOOLS,'documents':docs,'curated_ids':[str(x) for x in (wm.get('curated_ids') or [])],'tool_history':(state.get('tool_history') or [])[-4:]}
    if full: payload['full_runtime_context']={'auto_side_effect':'already applied'}
    return json.dumps(payload,ensure_ascii=False,sort_keys=True)


def compact(row, full=False):
    view=(row.get('full_view') if full else row.get('reduced_view')) or {}
    docs=[]
    for d in (view.get('documents') or [])[:12]:
        docs.append({'id':str(d.get('id')),'text':str(d.get('text') or '')[:256]})
    payload={
        'task':'Choose exactly one next BrowseComp tool call as JSON.',
        'query_id':str(row.get('query_id')),
        'step':int(row.get('step',0) or 0),
        'available_tools':TOOLS,
        'documents':docs,
        'curated_ids':[str(x) for x in (view.get('curated_ids') or row.get('curated_ids_pre') or [])],
        'tool_history':(view.get('tool_history') or [])[-4:],
    }
    if full:
        payload['full_runtime_context']={'auto_seed_present':True,'auto_side_effect':'top-k search results already visible in this state'}
    return json.dumps(payload,ensure_ascii=False,sort_keys=True)

def process(src,dst):
    with src.open() as f, dst.open('w') as out:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            r['prompt_reduced']=compact(r,False)
            r['prompt_full']=compact(r,True)
            if r.get('next_state'):
                r['next_prompt_reduced']=compact_state(r['next_state'],False)
                r['next_prompt_full']=compact_state(r['next_state'],True)
            out.write(json.dumps(r,ensure_ascii=False)+'\n')

for name in ['PROJECTED_ACTION_TRAIN','PROJECTED_ACTION_VALID','PROJECTED_ACTION_TEST','SHUFFLED_PROJECTED_ACTION_TRAIN']:
    process(ROOT/(name+'.jsonl'),ROOT/(name+'_COMPACT.jsonl'))
print('done')
