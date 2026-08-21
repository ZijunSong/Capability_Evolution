#!/usr/bin/env python3
"""Formal adaptive-rerank evidence recall fork over frozen 4x32 cohorts."""
from __future__ import annotations
import argparse, json, hashlib, math, statistics, sys
from pathlib import Path
from copy import deepcopy
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
import run_adaptive_rerank_instruction_formal_fork as base
from scape.rendering.dual_view import DualViewRenderer

SEEDS=(2214,2215,2216,2217)
COMP='adaptive_rerank_instruction'

def norm(x): return {str(v).split('_',1)[0] for v in (x or [])}
def recall(ids,gold):
 g=norm(gold); return len(norm(ids)&g)/len(g) if g else 0.0
def precision(ids,gold):
 p=norm(ids); g=norm(gold); return len(p&g)/len(p) if p else 0.0

def frozen_rows(root,seed,K):
 p=root/f'seed{seed}'/'shards'/f'{COMP}_K{K}.jsonl'
 return [json.loads(x) for x in p.open() if x.strip()]

def collect_exact(args,seed,K,queries,qrels,searcher,scorer,renderer):
 manifest=json.loads((args.cohort_root/f'seed{seed}'/'manifests'/'UTILITY_LIVE256.json').read_text())
 frozen=frozen_rows(args.cohort_root,seed,K)
 if args.limit:
  frozen=frozen[:args.limit]
 target={(str(r['query_id']),int(r['turn_id'])):r for r in frozen}
 # Candidate collection is deterministic under frozen qids/model/corpus; collect until all target snapshots found.
 found={}
 for qid in manifest['query_ids']:
  st=base.LiveState(qid=str(qid),query=queries[str(qid)],gold=qrels[str(qid)],searcher=searcher,component=COMP,branch_seed=f'collect:{COMP}:{qid}')
  for _ in range(8):
   key=(str(qid),int(st.step))
   if key in target:
    a_s,d_s,_=base.policy_action(st,scorer,renderer,component=COMP,full=False)
    a_t,d_t,_=base.policy_action(st,scorer,renderer,component=COMP,full=True)
    snap=st.snapshot(); found[key]=(target[key],{'snapshot':snap.to_dict(),'snapshot_hash':snap.content_hash(),'a_S':a_s,'a_T':a_t,'P_tool_reduced':d_s['tool_name_probs'],'P_tool_full':d_t['tool_name_probs'],'divergence':base.action_distance(a_s,a_t),'divergence_type':'tool-name' if a_s.get('name')!=a_t.get('name') else 'args-only'})
   a_s,_,_=base.policy_action(st,scorer,renderer,component=COMP,full=False); st.execute(a_s)
   if len(found)==len(target): return [found[k] for k in target]
 raise RuntimeError(f'frozen state replay mismatch seed={seed} K={K}: {len(found)}/{len(target)}')

def execute(st,action):
 # Valid reads only become successful/context-retained observations.
 name=str(action.get('name') or 'end_search'); args=dict(action.get('arguments') or {})
 before=set(st.curated_ids); ok=True; read_success=[]
 visible={str(d['id']) for d in st.documents}
 if name=='read_document':
  did=str(args.get('doc_id') or ''); ok=bool(did and did in visible)
  if ok and did not in st.read_ids: st.read_ids.append(did); read_success.append(did)
  st.cost+=1
 elif name=='review_docs':
  ids=[str(x) for x in (args.get('doc_ids') or [])]; valid=[x for x in ids if x in visible]; ok=bool(valid)
  for did in valid:
   if did not in st.read_ids: st.read_ids.append(did); read_success.append(did)
  st.cost+=1
 else:
  # Delegate all non-read transitions to canonical implementation.
  base.LiveState.execute(st,action); return ok,read_success
 st.history.append({'step':st.step,'action':dict(action)})
 st.observations.append({'step':st.step+1,'ok':ok,'read_success_ids':read_success,'curated_delta':len(set(st.curated_ids)-before),'n_curated':len(st.curated_ids)})
 st.step+=1
 return ok,read_success

def branch(start,first,K,scorer,renderer,label):
 st=start.clone(label); actions=[]; obs=[]; attempts=[]; entered=[]
 for i in range(K):
  if i==0: action=dict(first)
  else: action,_,_=base.policy_action(st,scorer,renderer,component=COMP,full=False)
  if action.get('name') in ('read_document','review_docs'):
   vals=action.get('arguments',{}).get('doc_id') if action.get('name')=='read_document' else action.get('arguments',{}).get('doc_ids',[])
   attempts.extend([str(vals)] if isinstance(vals,str) else [str(x) for x in vals])
  before=len(st.read_ids); execute(st,action); entered.extend(st.read_ids[before:])
  actions.append(dict(action)); obs.append(deepcopy(st.observations[-1]))
 docs=[str(d['id']) for d in st.documents]; curated=list(dict.fromkeys(st.curated_ids)); retained=list(dict.fromkeys(st.read_ids))
 activated=list(dict.fromkeys(curated+retained))
 return st,{'initial_candidate_evidence_ids':[],'final_candidate_evidence_ids':docs,'initial_curated_ids':[],'final_curated_ids':curated,'read_attempt_ids_within_k':attempts,'successful_read_ids_within_k':retained,'read_ids_entered_context':list(dict.fromkeys(entered)),'read_ids_retained_at_endpoint':retained,'final_activated_evidence_ids':activated,'actions':actions,'observations':obs,'context_evidence_ids_by_step':[retained],'initial_state_hash':start.snapshot().content_hash(),'final_state_hash':st.snapshot().content_hash(),'tool_cost':float(st.cost),'duplicate_read_count':len(attempts)-len(set(attempts))}

def run(args):
 queries=base._load_queries(args.browsecomp_root/'topics-qrels'/'queries.tsv'); qrels=base._load_qrels(args.browsecomp_root/'topics-qrels'/'qrel_evidence.txt')
 searcher,_=base.build_searcher(args.index_path,args.corpus_path); scorer=base.HFContinuationScorer(args.model,device=args.device,dtype=args.dtype,max_prompt_tokens=args.max_prompt_tokens); renderer=DualViewRenderer()
 pairs=collect_exact(args,args.seed,args.K,queries,qrels,searcher,scorer,renderer); out=args.out_dir; (out/'shards').mkdir(parents=True,exist_ok=True)
 p=out/'shards'/f'{COMP}_seed{args.seed}_K{args.K}.jsonl'
 with p.open('w') as f:
  for i,(old,item) in enumerate(pairs):
   start=base.state_from_snapshot(item['snapshot'],queries[old['query_id']],qrels[old['query_id']],searcher,COMP); init_docs=[str(d['id']) for d in start.documents]; init_cur=list(start.curated_ids)
   sf,es=branch(start,item['a_S'],args.K,scorer,renderer,'S'); tf,et=branch(start,item['a_T'],args.K,scorer,renderer,'T')
   es['initial_candidate_evidence_ids']=init_docs; et['initial_candidate_evidence_ids']=init_docs; es['initial_curated_ids']=init_cur; et['initial_curated_ids']=init_cur
   row={'component':COMP,'seed':args.seed,'K':args.K,'state_id':old['state_id'],'query_id':old['query_id'],'turn_id':old['turn_id'],'snapshot_hash':old['snapshot_hash'],'a_S':item['a_S'],'a_T':item['a_T'],'component_mask_first_action':{'teacher':{COMP:True},'student':{COMP:False}},'continuation_policy':'reduced','gold_evidence_ids':sorted(qrels[old['query_id']]),'branch_S_endpoint':es,'branch_T_endpoint':et,'branch_S_metrics':sf.metrics(),'branch_T_metrics':tf.metrics(),'full_harness_takeover':False,'runner':'adaptive_rerank_recall_128'}
   f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps({'seed':args.seed,'K':args.K,'n':len(pairs),'path':str(p)}))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--K',type=int,choices=[4,8],required=True); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--cohort-root',type=Path,default=REPO/'outputs/0820_adaptive_rerank_instruction_128_cohorts'); ap.add_argument('--out-dir',type=Path,default=REPO/'outputs/0820_adaptive_rerank_instruction_recall_128'); ap.add_argument('--browsecomp-root',type=Path,default=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus')); ap.add_argument('--index-path',type=Path); ap.add_argument('--corpus-path',type=Path,default=REPO/'outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl'); ap.add_argument('--model',default='/mnt/songzijun/models/pat-jj_harness-1-full/harness-1'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--dtype',default='bfloat16'); ap.add_argument('--max-prompt-tokens',type=int,default=3072); args=ap.parse_args(); args.index_path=args.index_path or args.browsecomp_root/'indexes/bm25'; run(args)
if __name__=='__main__': main()
