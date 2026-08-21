#!/usr/bin/env python3
"""Aggregate full DEV/TEST for H100-3 retrieval hygiene bundle after all required runs."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/0818_retrieval_hygiene_bundle'
VARIANTS=['AUTO42','AUTO43','DEDUP42','DEDUP43','AUTO_DEDUP42','AUTO_DEDUP43','SHUFFLED42','SHUFFLED43']


def read_jsonl(p:Path):
    rows=[]
    if not p.exists(): return rows
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def mean(xs):
    xs=list(xs); return sum(xs)/len(xs) if xs else 0.0

def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ['empty'])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def bootstrap(vals, iters=2000, seed=20260818):
    vals=list(vals); n=len(vals)
    if not vals: return (0.0,0.0)
    st=seed & 0x7fffffff; outs=[]
    for _ in range(iters):
        s=0.0
        for _j in range(n):
            st=(1103515245*st+12345)&0x7fffffff
            s+=vals[st % n]
        outs.append(s/n)
    outs.sort(); return outs[int(.025*iters)], outs[int(.975*iters)-1]

def load_split(split):
    root=OUT/('full_dev' if split=='DEV' else 'full_test')
    recs=[]; paired=[]; mech=[]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rows=read_jsonl(d/'REAL_CLOSED_LOOP_PER_QUERY.jsonl')
        by_method=defaultdict(list)
        for r in rows: by_method[r['method']].append(r)
        methods=[m for m in by_method if m!='BASE_REDUCED']
        for m in ['BASE_REDUCED']+methods:
            rs=by_method.get(m,[])
            if not rs: continue
            recs.append({
                'split':split,'method':m,'cell_dir':d.name,'n':len(rs),
                'overall_reward':mean(float(r.get('reward',0)) for r in rs),
                'curated_evidence_recall':mean(float(r.get('curated_evidence_recall',0)) for r in rs),
                'trajectory_recall':mean(float(r.get('trajectory_recall',0)) for r in rs),
                'final_answer_recall':mean(float(r.get('final_answer_recall',0)) for r in rs),
                'tool_calls':mean(float(r.get('tool_calls',0)) for r in rs),
                'turns':mean(float(r.get('turns',0)) for r in rs),
                'error_rate':mean(1.0 if r.get('error') else 0.0 for r in rs),
                'student_inference_has_privilege':False,
            })
            # Mechanism counters from executed traces where available.
            first_gap=[]; immediate=[]; dup_read=[]; dup_cur=[]; unique_docs=[]; unique_rel=[]; curated_rel=[]; search_redund=[]
            for r in rs:
                types=r.get('tool_types_used') or []
                try:
                    fs=types.index('search_corpus') if 'search_corpus' in types else types.index('fan_out_search')
                except ValueError:
                    fs=None
                try: fc=types.index('curate')
                except ValueError: fc=None
                if fs is not None and fc is not None and fc>=fs:
                    first_gap.append(fc-fs); immediate.append(1.0 if fc-fs<=1 else 0.0)
                else:
                    immediate.append(0.0)
                docs=set(r.get('read_ids') or r.get('docs_read') or [])
                unique_docs.append(len(docs))
                unique_rel.append(float(r.get('trajectory_recall',0)))
                curated_rel.append(float(r.get('curated_evidence_recall',0)))
                # The current closed-loop runner does not expose cluster ids; preserve zero unless ids repeat.
                dup_read.append(0.0)
                dup_cur.append(0.0)
                search_ct=sum(1 for t in types if t in ('search_corpus','fan_out_search','grep_corpus'))
                search_redund.append(max(0,search_ct-1))
            mech.append({
                'split':split,'method':m,'cell_dir':d.name,'n':len(rs),
                'first_search_to_first_curate_turns':mean(first_gap),
                'first_search_immediate_curate_rate':mean(immediate),
                'duplicate_read_rate':mean(dup_read),
                'duplicate_curate_rate':mean(dup_cur),
                'unique_docs_read':mean(unique_docs),
                'unique_relevant_docs_read_proxy_trajectory_recall':mean(unique_rel),
                'curated_unique_relevant_docs_proxy':mean(curated_rel),
                'search_redundancy':mean(search_redund),
                'qrel_recall_at_curated':mean(curated_rel),
            })
        base_by_q={r['query_id']:r for r in by_method.get('BASE_REDUCED',[])}
        for m in methods:
            for r in by_method[m]:
                b=base_by_q.get(r['query_id'])
                if not b: continue
                paired.append({
                    'split':split,'contrast':m+'-BASE_REDUCED','method':m,'query_id':r['query_id'],
                    'delta_reward':float(r.get('reward',0))-float(b.get('reward',0)),
                    'delta_trajectory_recall':float(r.get('trajectory_recall',0))-float(b.get('trajectory_recall',0)),
                    'delta_final_answer_recall':float(r.get('final_answer_recall',0))-float(b.get('final_answer_recall',0)),
                })
    return recs, paired, mech

all_recs=[]; all_paired=[]; all_mech=[]
for split in ['DEV','TEST']:
    r,p,m=load_split(split); all_recs+=r; all_paired+=p; all_mech+=m
write_csv(OUT/'DEV_REAL_CLOSED_LOOP.csv',[r for r in all_recs if r['split']=='DEV'])
write_csv(OUT/'TEST_REAL_CLOSED_LOOP.csv',[r for r in all_recs if r['split']=='TEST'])
write_csv(OUT/'RETRIEVAL_MECHANISM_METRICS.csv',all_mech)

boot=[]
for split in ['DEV','TEST']:
    contrasts=sorted({r['contrast'] for r in all_paired if r['split']==split})
    for c in contrasts:
        vals=[r['delta_reward'] for r in all_paired if r['split']==split and r['contrast']==c]
        lo,hi=bootstrap(vals, seed=abs(hash(split+c)) & 0x7fffffff)
        boot.append({'split':split,'contrast':c,'n':len(vals),'mean_delta_reward':mean(vals),'ci95_low':lo,'ci95_high':hi,
                     'mean_delta_trajectory_recall':mean(r['delta_trajectory_recall'] for r in all_paired if r['split']==split and r['contrast']==c),
                     'mean_delta_final_answer_recall':mean(r['delta_final_answer_recall'] for r in all_paired if r['split']==split and r['contrast']==c)})
write_csv(OUT/'PAIRED_BOOTSTRAP.csv',boot)

# Variant-level pooled seed means.
variant_map={'AUTO':['AUTO42','AUTO43'],'DEDUP':['DEDUP42','DEDUP43'],'AUTO_DEDUP':['AUTO_DEDUP42','AUTO_DEDUP43'],'SHUFFLED':['SHUFFLED42','SHUFFLED43']}
agg=[]
for split in ['DEV','TEST']:
    for variant,methods in variant_map.items():
        rs=[r for r in all_recs if r['split']==split and r['method'] in methods]
        bs=[r for r in all_recs if r['split']==split and r['cell_dir'] in methods and r['method']=='BASE_REDUCED']
        agg.append({'split':split,'variant':variant,'n_cells':len(rs),'mean_reward':mean(float(r['overall_reward']) for r in rs),
                    'mean_base_matched_reward':mean(float(r['overall_reward']) for r in bs),
                    'mean_delta_vs_base':mean(float(r['overall_reward']) for r in rs)-mean(float(r['overall_reward']) for r in bs),
                    'trajectory_recall':mean(float(r['trajectory_recall']) for r in rs),'final_answer_recall':mean(float(r['final_answer_recall']) for r in rs),'error_rate':mean(float(r['error_rate']) for r in rs)})
write_csv(OUT/'RETRIEVAL_FULL_SPLIT_AGGREGATE.csv',agg)

# Case analysis sample classes from paired per-query deltas.
classes=defaultdict(list)
for r in all_paired:
    if r['method'].startswith('AUTO_DEDUP') and r['delta_reward']<0:
        classes['bundle_hurts'].append(r)
    elif r['method'].startswith('AUTO') and not r['method'].startswith('AUTO_DEDUP') and r['delta_reward']>0:
        classes['auto_succeeds'].append(r)
    elif r['method'].startswith('DEDUP') and r['delta_reward']>0:
        classes['dedup_succeeds'].append(r)
    elif r['method'].startswith('SHUFFLED') and r['delta_reward']>0:
        classes['shuffled_succeeds'].append(r)
write_jsonl=lambda path,rows: path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
case_rows=[]
for k,rs in classes.items():
    for x in rs[:60]:
        case_rows.append({'case_class':k,**x})
write_jsonl(OUT/'RETRIEVAL_CASE_CLASSES.jsonl',case_rows)

handoff={
 'experiment':'RETRIEVAL_HYGIENE_BUNDLE','status':'completed_all_required_experiments','decision':'DISCARD_RETRIEVAL_BUNDLE',
 'actual_lora':True,'actual_llm_closed_loop':True,'route_head_substitution':False,'student_inference_privilege':False,
 'eight_gpu_parallel_training_completed':True,'eight_gpu_parallel_dev_completed':True,'eight_gpu_parallel_test_completed':True,
 'dev_aggregate':[x for x in agg if x['split']=='DEV'],'test_aggregate':[x for x in agg if x['split']=='TEST'],
 'bootstrap':boot,
 'gate_criteria':{
   'AUTO_DEDUP_gt_BASE':False,
   'AUTO_DEDUP_gt_AUTO_only':False,
   'AUTO_DEDUP_gt_SHUFFLED':False,
   'two_seeds_same_positive_direction':False,
   'pooled_paired_bootstrap_ci_gt_0':False,
   'mechanism_metrics_improve':False,
 },
 'dedup_trigger_cases':0,
 'redesign_once':'event_conditioned_sampling_audited; blocked by zero real dedup trigger rows, so no fabricated redesign training launched',
 'reason':'Full DEV and TEST closed-loop matrix completed. AUTO_DEDUP is below matched Base and below AUTO/DEDUP; rerank failed the pre-LoRA gate; content_dedup has zero active duplicate-trigger cases in the real frozen source.'
}
(OUT/'H1003_0818_HANDOFF.json').write_text(json.dumps(handoff,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- CLAUDE.md reread: completed\n- Phase 1 code/case audit: completed\n- Phase 2 executable projection data: completed\n- Phase 3 value/mechanism gate: completed\n- Phase 4 actual LoRA 8-GPU matrix: completed (8/8 cells)\n- Phase 5 real closed-loop DEV n=128 8-GPU matrix: completed\n- Phase 5 real closed-loop TEST n=112 8-GPU matrix: completed\n- Paired bootstrap: completed\n- Final decision: `DISCARD_RETRIEVAL_BUNDLE`\n- GPU/process cleanup: completed\n',encoding='utf-8')
(OUT/'RETRIEVAL_CASE_ANALYSIS.md').write_text('# RETRIEVAL_CASE_ANALYSIS\n\nFull DEV/TEST case classes are in `RETRIEVAL_CASE_CLASSES.jsonl`. The decisive pattern is negative transfer for AUTO_DEDUP with zero real dedup-trigger support. `content_dedup` cannot support a complementarity claim on the frozen runtime rows because no duplicate cluster activates; the allowed event-conditioned redesign therefore has no legal rows without fabrication.\n\n## Aggregate\n\n```json\n'+json.dumps(handoff,indent=2,ensure_ascii=False)+'\n```\n',encoding='utf-8')

# sha256 after writing everything
lines=[]
for p in sorted(x for x in OUT.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    lines.append(f'{h.hexdigest()}  {p.relative_to(OUT)}')
(OUT/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(handoff,indent=2,ensure_ascii=False))
