#!/usr/bin/env python3
import hashlib, json, math, random, statistics
from pathlib import Path

ROOT=Path('/mnt/songzijun/Capability_Evolution/SCAPE')
OUT=ROOT/'outputs/0820_joint_importance_subtractive_recall_fresh_128'
QREL=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt')

def norm(x): return str(x).split('_',1)[0]
def recall(ids,gold):
 p={norm(x) for x in ids}; g={norm(x) for x in gold}
 return len(p&g)/len(g) if g else None
def precision(ids,gold):
 p={norm(x) for x in ids}; g={norm(x) for x in gold}
 return len(p&g)/max(1,len(p))
def boot(vals,seed):
 r=random.Random(seed); n=len(vals); means=[]
 for _ in range(10000): means.append(sum(vals[r.randrange(n)] for _ in range(n))/n)
 means.sort(); return [100*means[250],100*means[9749]]
def qrels():
 d={}
 for line in QREL.read_text().splitlines():
  p=line.split()
  if len(p)>=3: d.setdefault(str(p[0]),set()).add(str(p[2]))
 return d

def main():
 gold=qrels(); rows=[]
 for seed in (8423,8424):
  for f in sorted((OUT/'shards').glob(f'fresh_seed{seed}_shard*.jsonl')):
   for line in f.read_text().splitlines():
    if line.strip(): rows.append(json.loads(line))
 assert len(rows)==256
 assert all(not r['full_harness_takeover'] for r in rows)
 assert len({r['state_id'] for r in rows})==256
 for r in rows:
  assert set(r['gold_evidence_ids'])==gold[r['query_id']]
  assert r['branch_S']['checkpoints']['4']['final_state_hash']
  assert r['branch_T']['checkpoints']['4']['final_state_hash']
  for b in ('branch_S','branch_T'):
   for k in ('4','8'):
    e=r[b]['checkpoints'][k]
    assert set(e['final_activated_evidence_ids']) == set(e['final_curated_ids']) | set(e['read_ids_retained_at_endpoint'])
    assert set(e['read_ids_retained_at_endpoint']) <= set(e['successful_read_ids_within_k'])
    assert set(e['successful_read_ids_within_k']) <= set(e['read_attempt_ids_within_k'])
 result={'schema_version':'joint_candidate_activated_recall_fresh_v1','status':'PASS','component':'importance_tagging+subtractive_curation','n_frozen_states':256,'n_paired_rows_by_K':{'4':256,'8':256},'seeds':[8423,8424],'horizons':[4,8],'qrel_path':str(QREL),'qrel_sha256':hashlib.sha256(QREL.read_bytes()).hexdigest(),'normalization':'split_at_first_underscore_v1','runner':'run_joint_importance_subtractive_recall_fresh.py','results_by_K':{}}
 for k in ('4','8'):
  summary={'n':len(rows),'per_seed':{}}
  for metric in ('candidate','activated'):
   t=[]; s=[]; d=[]
   for r in rows:
    te=r['branch_T']['checkpoints'][k]; se=r['branch_S']['checkpoints'][k]
    ids_t=te['final_candidate_evidence_ids'] if metric=='candidate' else te['final_activated_evidence_ids']
    ids_s=se['final_candidate_evidence_ids'] if metric=='candidate' else se['final_activated_evidence_ids']
    t.append(recall(ids_t,r['gold_evidence_ids'])); s.append(recall(ids_s,r['gold_evidence_ids'])); d.append(t[-1]-s[-1])
   summary[metric]={'teacher_mean':statistics.mean(t),'student_mean':statistics.mean(s),'delta_pp':100*statistics.mean(d),'bootstrap_ci95_pp':boot(d,20260820+int(k)+(0 if metric=='candidate' else 1000)),'positive_negative_zero':[sum(x>0 for x in d),sum(x<0 for x in d),sum(x==0 for x in d)],'teacher_precision':statistics.mean(precision(r['branch_T']['checkpoints'][k]['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_T']['checkpoints'][k]['final_activated_evidence_ids'],r['gold_evidence_ids']) for r in rows),'student_precision':statistics.mean(precision(r['branch_S']['checkpoints'][k]['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_S']['checkpoints'][k]['final_activated_evidence_ids'],r['gold_evidence_ids']) for r in rows),'teacher_size':statistics.mean(len(set(norm(x) for x in (r['branch_T']['checkpoints'][k]['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_T']['checkpoints'][k]['final_activated_evidence_ids']))) for r in rows),'student_size':statistics.mean(len(set(norm(x) for x in (r['branch_S']['checkpoints'][k]['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_S']['checkpoints'][k]['final_activated_evidence_ids']))) for r in rows)}
   for seed in (8423,8424):
    sr=[r for r in rows if r['seed']==seed]; ds=[]
    for r in sr:
     te=r['branch_T']['checkpoints'][k]; se=r['branch_S']['checkpoints'][k]
     it=te['final_candidate_evidence_ids'] if metric=='candidate' else te['final_activated_evidence_ids']; is_=se['final_candidate_evidence_ids'] if metric=='candidate' else se['final_activated_evidence_ids']
     ds.append(recall(it,r['gold_evidence_ids'])-recall(is_,r['gold_evidence_ids']))
    summary['per_seed'].setdefault(str(seed),{})[f'{metric}_delta_pp']=100*statistics.mean(ds)
  for metric in ('candidate','activated'):
   x=[summary['per_seed'][str(s)][f'{metric}_delta_pp'] for s in (8423,8424)]
   summary[metric]['seed_mean_pp']=statistics.mean(x); summary[metric]['seed_sample_std_pp']=statistics.stdev(x)
  result['results_by_K'][k]=summary
 result_path=OUT/'JOINT_CANDIDATE_ACTIVATED_EVIDENCE_RECALL_FRESH.json'
 result_path.write_text(json.dumps(result,indent=2)+'\n')
 (OUT/'JOINT_FRESH_VALUE_PER_STATE.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 print(json.dumps(result,indent=2))

if __name__=='__main__': main()
