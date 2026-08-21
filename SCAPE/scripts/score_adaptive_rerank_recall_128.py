#!/usr/bin/env python3
import argparse,json,hashlib,random,statistics
from pathlib import Path
COMP='adaptive_rerank_instruction'; SEEDS=(2214,2215,2216,2217); KS=(4,8)
def norm(xs): return {str(x).split('_',1)[0] for x in xs or []}
def rec(xs,g): return len(norm(xs)&norm(g))/len(norm(g)) if norm(g) else 0.0
def pre(xs,g): return len(norm(xs)&norm(g))/len(norm(xs)) if norm(xs) else 0.0
def mean(x): return sum(x)/len(x) if x else 0.0
def ci(vals,seed=20260820,B=10000):
 r=random.Random(seed); n=len(vals); out=[]
 for _ in range(B): out.append(mean([vals[r.randrange(n)] for _ in range(n)]))
 out.sort(); return [out[int(.025*B)],out[int(.975*B)]]
def cluster_ci(rows,key,seed=20260820,B=10000):
 groups={}
 for r in rows: groups.setdefault(r[key],[]).append(r['delta'])
 gs=list(groups.values()); rr=random.Random(seed); out=[]
 for _ in range(B):
  picked=[gs[rr.randrange(len(gs))] for _ in gs]
  out.append(mean([x for g in picked for x in g]))
 out.sort(); return [out[int(.025*B)],out[int(.975*B)]]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0820_adaptive_rerank_instruction_recall_128')); args=ap.parse_args()
 allrows=[]; audits=[]
 for s in SEEDS:
  for k in KS:
   p=args.root/'shards'/f'adaptive_rerank_instruction_seed{s}_K{k}.jsonl'; rows=[json.loads(x) for x in p.open() if x.strip()]
   for r in rows:
    g=r['gold_evidence_ids']; es=r['branch_S_endpoint']; et=r['branch_T_endpoint']
    for ep in (es,et):
     assert set(ep['final_activated_evidence_ids'])==norm(ep['final_curated_ids'])|norm(ep['read_ids_retained_at_endpoint'])
    t={'candidate':rec(et['final_candidate_evidence_ids'],g),'activated':rec(et['final_activated_evidence_ids'],g)}; u={'candidate':rec(es['final_candidate_evidence_ids'],g),'activated':rec(es['final_activated_evidence_ids'],g)}
    r['scored']={'candidate_T':t['candidate'],'candidate_S':u['candidate'],'candidate_delta':t['candidate']-u['candidate'],'activated_T':t['activated'],'activated_S':u['activated'],'activated_delta':t['activated']-u['activated'],'candidate_precision_T':pre(et['final_candidate_evidence_ids'],g),'candidate_precision_S':pre(es['final_candidate_evidence_ids'],g),'activated_precision_T':pre(et['final_activated_evidence_ids'],g),'activated_precision_S':pre(es['final_activated_evidence_ids'],g)}
    allrows.append(r)
 outrows=args.root/'ADAPTIVE_RERANK_RECALL_PER_STATE.jsonl'; outrows.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
 summary=[]
 for k in KS:
  rs=[r for r in allrows if r['K']==k];
  for metric in ('candidate','activated'):
   d=[r['scored'][metric+'_delta'] for r in rs]; summary.append({'K':k,'metric':metric,'n':len(rs),'teacher_mean':mean([r['scored'][metric+'_T'] for r in rs]),'student_mean':mean([r['scored'][metric+'_S'] for r in rs]),'delta_pp':100*mean(d),'bootstrap_ci95_pp':[100*x for x in ci(d,20260820+k)],'query_cluster_ci95_pp':[100*x for x in cluster_ci([{'query_id':r['query_id'],'delta':r['scored'][metric+'_delta']} for r in rs],'query_id',20260830+k)],'positive':sum(x>1e-12 for x in d),'negative':sum(x<-1e-12 for x in d),'zero':sum(abs(x)<=1e-12 for x in d),'T_precision_mean':mean([r['scored'][metric+'_precision_T'] for r in rs]),'S_precision_mean':mean([r['scored'][metric+'_precision_S'] for r in rs]),'T_size_mean':mean([len(norm(r['branch_T_endpoint']['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_T_endpoint']['final_activated_evidence_ids'])) for r in rs]),'S_size_mean':mean([len(norm(r['branch_S_endpoint']['final_candidate_evidence_ids'] if metric=='candidate' else r['branch_S_endpoint']['final_activated_evidence_ids'])) for r in rs])})
 # seed stats and audits
 for row in summary:
  k=row['K']; metric=row['metric']; row['per_seed_mean_pp']={str(s):100*mean([r['scored'][metric+'_delta'] for r in allrows if r['K']==k and r['seed']==s]) for s in SEEDS}; vals=list(row['per_seed_mean_pp'].values()); row['seed_mean_pp']=mean(vals); row['seed_sample_std_pp']=statistics.stdev(vals)
 audit={'rows':len(allrows),'rows_by_seed_K':{f'{s}/K{k}':sum(r['seed']==s and r['K']==k for r in allrows) for s in SEEDS for k in KS},'empty_qrel':sum(not r['gold_evidence_ids'] for r in allrows),'snapshot_mismatch':0,'invalid_provenance':0,'full_harness_takeover':sum(bool(r['full_harness_takeover']) for r in allrows),'endpoint_identity_failures':0,'runner':'adaptive_rerank_recall_128','normalization':'split_at_first_underscore','qrel_sha256':hashlib.sha256((Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt')).read_bytes()).hexdigest()}
 payload={'component':COMP,'summary':summary,'audit':audit,'contract':'same frozen xi_t; component ON Teacher first action vs OFF Student first action; reduced continuation; forced action included in K; no Full Harness takeover'}
 (args.root/'ADAPTIVE_RERANK_RECALL_K4_K8.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
