#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
SRC=Path('/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/scripts');sys.path.insert(0,str(SRC))
import run_btp_h1002_auto_populate as btp
base=btp.base
COMPONENT='auto_populate_first_search'
def run_branch(start,first_action,*,k,scorer,renderer,component,label,replay_jitter=''):
 st=start.clone(label); trace=[]; st.execute(first_action); trace.append({'branch':label,'phase':'forced_first','action':dict(first_action),'metrics':st.metrics()}); full=label=='T'
 for i in range(k):
  action,dist,dual=base.policy_action(st,scorer,renderer,component=component,full=full,tie_jitter=f'{replay_jitter}:{i}' if replay_jitter else '')
  st.execute(action); trace.append({'branch':label,'phase':f'continue_{i+1}','action':action,'top_prob':max(dist['tool_name_probs'].values()),'snapshot_hash':dual['snapshot_hash'],'metrics':st.metrics(),'view_policy':'full' if full else 'reduced'})
 return st,trace
base.run_branch=run_branch
def main():
 p=argparse.ArgumentParser();p.add_argument('--stratum',choices=['NATURAL_FIRST_SEARCH','AUTO_EFFECT_ACTIVE'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--horizon',type=int,choices=[4,8],required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--dtype',default='bfloat16');p.add_argument('--max-prompt-tokens',type=int,default=3072);p.add_argument('--n-states',type=int,default=128);p.add_argument('--n-queries-pool',type=int,default=768);p.add_argument('--max-turns-per-query',type=int,default=4);p.add_argument('--append-existing',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--allow-active-fill',action='store_true');p.add_argument('--browsecomp-root',type=Path,default=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus'));p.add_argument('--index-path',type=Path,default=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25'));a=p.parse_args();btp._install_auto_globals()
 if btp.base.LuceneSearcher is None: btp.base.LuceneSearcher=lambda _: btp.base.JsonlSearcher(Path('/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl'))
 m=btp._manifest_path(a.out_dir,a.stratum,a.seed)
 if not m.exists():
  rc=btp.collect_manifest(a)
  if rc not in (0,2): return rc
 ns=argparse.Namespace(**vars(a));ns.horizon=a.horizon-1
 rc=btp.run_cell(ns)
 if rc!=0:return rc
 src=a.out_dir/'AUTO_VALUE_CONFIRM'/'shards'/f'{a.stratum}_seed{a.seed}_K{a.horizon-1}.jsonl';dst=a.out_dir/'AUTO_VALUE_CONFIRM'/'shards'/f'{a.stratum}_seed{a.seed}_K{a.horizon}.jsonl';rows=[]
 for line in src.read_text().splitlines():
  if line.strip():
   r=json.loads(line);r.update({'horizon':a.horizon,'K':a.horizon,'experiment':'GPT_OSS_AUTO_ALWAYS_ON_OFF_TWO_SETTINGS','model':str(a.model),'continuation_policy':'teacher_full_always_on_student_reduced_always_off','always_on_semantics':True});rows.append(r)
 dst.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));return 0 if len(rows)==a.n_states else 2
if __name__=='__main__':raise SystemExit(main())
