#!/usr/bin/env python3
import hashlib, json, random, statistics
from collections import defaultdict
from pathlib import Path
from run_joint_importance_subtractive_preopd_fork import JointLiveState, _load_queries, _load_qrels, build_searcher

ROOT=Path('/mnt/songzijun/Capability_Evolution/SCAPE')
SRC=ROOT/'outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry'
OUT=ROOT/'outputs/0820_joint_importance_subtractive_recall_128_final'
BC=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus')

def norm(x): return str(x).split('_',1)[0]
def metrics(ids,gold):
 p={norm(x) for x in ids}; g={norm(x) for x in gold}
 return len(p&g)/len(g), len(p&g)/max(1,len(p)), len(p)
def boot(vals,seed=20260820,n=10000):
 r=random.Random(seed); m=len(vals); xs=sorted(sum(vals[r.randrange(m)] for _ in range(m))/m for _ in range(n))
 return [100*xs[int(.025*n)],100*xs[int(.975*n)]]

def endpoint(st,initial,gold):
 activated=sorted(set(st.curated_ids)|set(st.read_ids_retained_at_endpoint))
 cr,cp,cs=metrics([d['id'] for d in st.documents],gold); ar,ap,asz=metrics(activated,gold)
 return {'gold_evidence_ids':sorted(gold),'initial_candidate_evidence_ids':[d['id'] for d in initial.documents],
 'final_candidate_evidence_ids':[d['id'] for d in st.documents],'initial_curated_ids':list(initial.curated_ids),
 'final_curated_ids':list(st.curated_ids),'read_attempt_ids_within_k':list(st.read_ids),
 'successful_read_ids_within_k':list(st.successful_read_ids),'read_ids_entered_context':list(st.read_ids_entered_context),
 'read_ids_retained_at_endpoint':list(st.read_ids_retained_at_endpoint),'final_activated_evidence_ids':activated,
 'candidate_recall':cr,'candidate_precision':cp,'candidate_size':cs,'activated_recall':ar,'activated_precision':ap,'activated_size':asz}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 queries=_load_queries(BC/'topics-qrels/queries.tsv'); qrels=_load_qrels(BC/'topics-qrels/qrel_evidence.txt')
 searcher,backend=build_searcher(BC/'indexes/bm25',ROOT/'outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl')
 rows=[]; mismatches=[]
 for seed in (8423,8424):
  for k in (4,8):
   path=SRC/'shards'/f'importance_tagging_plus_subtractive_curation_K{k}_seed{seed}.jsonl'
   for line in path.open():
    old=json.loads(line); qid=str(old['query_id']); base=JointLiveState(qid=qid,query=queries[qid],gold=qrels[qid],searcher=searcher,branch_seed=f"collect:importance_tagging_plus_subtractive_curation:{qid}")
    # All frozen rows were collected at turn 0 in this run. Require exact hash.
    got=base.snapshot().content_hash()
    if got!=old['snapshot_hash']:
     mismatches.append({'state_id':old['state_id'],'expected':old['snapshot_hash'],'got':got}); continue
    initial=base.clone('initial'); branches={}
    for b in ('S','T'):
     st=base.clone(b)
     trace=old[f'branch_{b}_trace']
     for event in trace: st.execute(event['action'])
     branches[b]=endpoint(st,initial,qrels[qid])
    row={k:v for k,v in old.items() if k not in ('branch_S_trace','branch_T_trace')}
    row['branch_S_endpoint']=branches['S']; row['branch_T_endpoint']=branches['T']
    row['candidate_recall_delta']=branches['T']['candidate_recall']-branches['S']['candidate_recall']
    row['activated_recall_delta']=branches['T']['activated_recall']-branches['S']['activated_recall']
    rows.append(row)
 (OUT/'JOINT_RECALL_PER_STATE.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 result={'schema_version':'joint_candidate_activated_recall_v2','status':'PASS' if len(rows)==512 and not mismatches else 'FAIL','component':'importance_tagging+subtractive_curation','source_artifact':str(SRC),'replay_backend':backend,'normalization':'split_at_first_underscore_v1','n_rows':len(rows),'snapshot_mismatch_count':len(mismatches),'snapshot_mismatches':mismatches,'results_by_K':{}}
 for k in (4,8):
  rr=[r for r in rows if r['K']==k]; item={'n':len(rr),'per_seed':{}}
  for metric in ('candidate','activated'):
   ds=[r[f'{metric}_recall_delta'] for r in rr]
   item[metric]={'teacher_mean':statistics.mean(r[f'branch_T_endpoint'][f'{metric}_recall'] for r in rr),'student_mean':statistics.mean(r[f'branch_S_endpoint'][f'{metric}_recall'] for r in rr),'delta_pp':100*statistics.mean(ds),'bootstrap_ci95_pp':boot(ds,20260820+k+(0 if metric=='candidate' else 100)),'positive_negative_zero':[sum(x>0 for x in ds),sum(x<0 for x in ds),sum(x==0 for x in ds)],'teacher_precision':statistics.mean(r['branch_T_endpoint'][f'{metric}_precision'] for r in rr),'student_precision':statistics.mean(r['branch_S_endpoint'][f'{metric}_precision'] for r in rr),'teacher_size':statistics.mean(r['branch_T_endpoint'][f'{metric}_size'] for r in rr),'student_size':statistics.mean(r['branch_S_endpoint'][f'{metric}_size'] for r in rr)}
  for seed in (8423,8424):
   sr=[r for r in rr if r['seed']==seed]; item['per_seed'][str(seed)]={'n':len(sr),'candidate_delta_pp':100*statistics.mean(r['candidate_recall_delta'] for r in sr),'activated_delta_pp':100*statistics.mean(r['activated_recall_delta'] for r in sr)}
  for metric in ('candidate','activated'):
   vals=[item['per_seed'][str(s)][f'{metric}_delta_pp'] for s in (8423,8424)]; item[metric]['seed_mean_pp']=statistics.mean(vals); item[metric]['seed_sample_std_pp']=statistics.stdev(vals)
  result['results_by_K'][str(k)]=item
 (OUT/'JOINT_CANDIDATE_ACTIVATED_EVIDENCE_RECALL.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
