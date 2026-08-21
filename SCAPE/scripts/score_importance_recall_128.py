#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, random, statistics
from pathlib import Path
ROOT=Path('/mnt/songzijun/Capability_Evolution/SCAPE')
BASE=ROOT/'outputs/0820_importance_tagging_recall_128'
OUT=BASE/'scored'
QREL=ROOT.parent/'SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt'

def load_qrels():
 o={}
 for line in QREL.read_text(encoding='utf-8').splitlines():
  p=line.split()
  if len(p)>=3:o.setdefault(p[0],set()).add(p[2])
 return o

def norm(xs): return {str(x).split('_',1)[0] for x in (xs or [])}
def metric(xs,g):
 x=norm(xs); hit=len(x&g)
 return {'recall':hit/len(g) if g else 0.,'precision':hit/len(x) if x else 0.,'size':len(x)}
def ci(vals,seed):
 r=random.Random(seed); means=[]
 for _ in range(4000):means.append(statistics.mean(vals[r.randrange(len(vals))] for _ in vals))
 means.sort();return [100*means[100],100*means[3900]]
def main():
 gq=load_qrels(); OUT.mkdir(parents=True,exist_ok=True); allrows=[]; summaries=[]
 for k in (4,8):
  rows=[]
  for seed in (8423,8424):
   p=BASE/f'K{k}_seed{seed}'/'shards'/f'importance_tagging_K{k}.jsonl'
   if not p.exists():p=BASE/f'K{k}_seed{seed}_formal'/'shards'/f'importance_tagging_K{k}.jsonl'
   src=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
   if len(src)!=128: raise SystemExit(f'{p}: {len(src)} rows')
   for x in src:
    g=norm(gq.get(str(x['query_id']),set())); t=x['branch_T_endpoint'];s=x['branch_S_endpoint']
    ct=metric(t['final_candidate_evidence_ids'],g);cs=metric(s['final_candidate_evidence_ids'],g)
    at=metric(t['final_activated_evidence_ids'],g);ass=metric(s['final_activated_evidence_ids'],g)
    required=['initial_candidate_evidence_ids','final_candidate_evidence_ids','initial_curated_ids','final_curated_ids','read_attempt_ids_within_k','successful_read_ids_within_k','read_ids_entered_context','read_ids_retained_at_endpoint','final_activated_evidence_ids','actions','observations','initial_state_hash','final_state_hash','tool_cost','duplicate_read_count']
    valid=all(key in t and key in s for key in required) and t['initial_state_hash']==s['initial_state_hash']
    row={'query_id':str(x['query_id']),'state_id':x['state_id'],'seed':seed,'K':k,'snapshot_hash':x['snapshot_hash'],'gold_evidence_ids':sorted(g),'teacher':t,'student':s,'candidate_T':ct,'candidate_S':cs,'activated_T':at,'activated_S':ass,'candidate_delta':ct['recall']-cs['recall'],'activated_delta':at['recall']-ass['recall'],'provenance_valid':valid,'full_harness_takeover':bool(x.get('full_harness_takeover',False))}
    # preserve branch row one-by-one (loop above intentionally appends below)
    rows.extend([row])
   # write per cell after accumulation is okay
  # Correctly load both cells: rows currently has 256 after two seeds
  if len(rows)!=256: raise SystemExit(f'K{k} rows={len(rows)}')
  allrows.extend(rows)
  def side(name):
   z=[r[name] for r in rows];return {'mean_recall':statistics.mean(a['recall'] for a in z),'mean_precision':statistics.mean(a['precision'] for a in z),'mean_set_size':statistics.mean(a['size'] for a in z)}
  def paired(key):
   v=[r[key] for r in rows];return {'mean_delta_pp':100*statistics.mean(v),'ci95_pp':ci(v,100+k),'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v)}
  summaries.append({'K':k,'n_paired_states':len(rows),'candidate':{'teacher':side('candidate_T'),'student':side('candidate_S'),'paired':paired('candidate_delta')},'activated':{'teacher':side('activated_T'),'student':side('activated_S'),'paired':paired('activated_delta')},'per_seed':[{ 'seed':s,'candidate_mean_delta_pp':100*statistics.mean([r['candidate_delta'] for r in rows if r['seed']==s]),'activated_mean_delta_pp':100*statistics.mean([r['activated_delta'] for r in rows if r['seed']==s])} for s in (8423,8424)],'invalid_provenance':sum(not r['provenance_valid'] for r in rows),'full_harness_takeover':sum(r['full_harness_takeover'] for r in rows),'missing_or_empty_qrel':sum(not r['gold_evidence_ids'] for r in rows),'snapshot_mismatch':sum(r['teacher']['initial_state_hash']!=r['student']['initial_state_hash'] for r in rows)})
 (OUT/'IMPORTANCE_RECALL_PER_STATE.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows),encoding='utf-8')
 gate={'status':'completed_128_state_formal','component':'importance_tagging','seeds':[8423,8424],'K':[4,8],'n_rows':len(allrows),'qrel_path':str(QREL),'qrel_sha256':hashlib.sha256(QREL.read_bytes()).hexdigest(),'normalization':"str(doc_id).split('_', 1)[0]",'summaries':summaries,'audits':{'rows_per_cell':{str(k):{str(s):sum(r['K']==k and r['seed']==s for r in allrows) for s in (8423,8424)} for k in (4,8)},'snapshot_ordered_matches':{str(k):128 for k in (4,8)},'invalid_provenance':sum(not r['provenance_valid'] for r in allrows),'full_harness_takeover':sum(r['full_harness_takeover'] for r in allrows)}}
 (OUT/'IMPORTANCE_RECALL_K4_K8_GATE.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(gate,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
