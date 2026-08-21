#!/usr/bin/env python3
"""Score strict legal-action rate and official-test Evidence Recall@5 only."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any

SETTINGS=("TEACHER","STUDENT_BEFORE_OPD","STUDENT_AFTER_PURE_OPD","STUDENT_AFTER_RL_PLUS_OPD")
INDEX=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25')
HARNESS=Path('/mnt/songzijun/Capability_Evolution/SCAPE/external/harness-1')
sys.path.insert(0,str(HARNESS))
from harness.ultra_core import FAN_OUT_MAX_QUERIES
from pyserini.search.lucene import LuceneSearcher

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def norm(x:str)->str: return str(x).split('_',1)[0]

def contract(r:dict[str,Any])->tuple[bool,list[str]]:
 n,p=r.get('tool_name'),r.get('params') or {}
 if n=='fan_out_search':
  qs=p.get('queries'); ok=isinstance(qs,list) and 1<=len(qs)<=FAN_OUT_MAX_QUERIES and all(isinstance(q,str) and q.strip() for q in qs)
  return ok,[q.strip() for q in qs] if ok else []
 if n=='search_corpus':
  q=p.get('query') or p.get('q'); ok=isinstance(q,str) and bool(q.strip()); return ok,[q.strip()] if ok else []
 if n=='grep_corpus':
  q=p.get('pattern'); ok=isinstance(q,str) and bool(q.strip()); return ok,[q.strip()] if ok else []
 return False,[]

def fused_top5(searcher,queries):
 runs=[[str(h.docid) for h in searcher.search(q,5)] for q in queries]
 out=[]; seen=set()
 for rank in range(5):
  for run in runs:
   if rank<len(run) and run[rank] not in seen:
    seen.add(run[rank]); out.append(run[rank])
    if len(out)==5:return out
 return out

def summarize(rows,split):
 xs=rows if split=='all_pool' else [r for r in rows if r['official_split']=='test']; n=len(xs)
 return {'split':split,'n_queries':n,'legal_action_rate':sum(r['legal'] for r in xs)/n,'evidence_recall_at_5':sum(r['evidence_recall_at_5'] for r in xs)/n}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--shards',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 mans=[a.shards/s/'384_QUERY_MANIFEST.json' for s in SETTINGS]; hs=[sha(p) for p in mans]
 if len(set(hs))!=1: raise RuntimeError('manifest mismatch')
 man=json.loads(mans[0].read_text()); qrels={r['query_id']:{norm(x) for x in r['evidence_docids']} for r in man['queries']}; (a.output_dir/'384_QUERY_MANIFEST.json').write_text(json.dumps(man,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
 searcher=LuceneSearcher(str(INDEX)); summaries=[]; order=None; source_hashes={}
 for s in SETTINGS:
  src=a.shards/s/s/'PER_QUERY.jsonl'; source_hashes[s]=sha(src); rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]; ids=[r['query_id'] for r in rows]
  if len(rows)!=384 or len(set(ids))!=384 or (order is not None and ids!=order): raise RuntimeError(f'{s} alignment failure')
  order=ids
  for r in rows:
   legal,qs=contract(r); docs=fused_top5(searcher,qs) if legal else []; rel=qrels[r['query_id']]
   r.update({'legal':legal,'executable':legal,'executed_queries':qs,'retrieval_backend':'pyserini_lucene','fan_out_fusion':'rankwise_round_robin','retrieved_docids_at_5':docs,'evidence_recall_at_5':len({norm(x) for x in docs}&rel)/max(1,len(rel))})
  d=a.output_dir/s; d.mkdir(parents=True,exist_ok=True)
  with (d/'PER_QUERY.jsonl').open('w') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  sm={'setting':s,'all_pool':summarize(rows,'all_pool'),'official_test':summarize(rows,'official_test')}; (d/'SUMMARY.json').write_text(json.dumps(sm,indent=2,sort_keys=True)+'\n'); summaries.append(sm)
 payload={'status':'AUTO_POPULATE_OPD_384_R5_COMPLETE','component':'auto_populate_first_search','query_count':384,'test_query_count':76,'manifest_sha256':sha(a.output_dir/'384_QUERY_MANIFEST.json'),'metrics':['legal_action_rate','evidence_recall_at_5'],'explicitly_not_computed':['recall_at_100','recall_at_1000'],'retrieval_backend':'pyserini_lucene','fan_out_fusion':'rankwise_round_robin_top5','settings':summaries,'provenance':{'source_generation_sha256':source_hashes,'runner_sha256':sha(Path('/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/eval_auto_populate_opd_384.py')),'scorer_sha256':sha(Path(__file__)),'java_home':os.environ.get('JAVA_HOME'),'fan_out_max_queries':FAN_OUT_MAX_QUERIES,'index_sha256':{p.name:sha(p) for p in sorted(INDEX.iterdir()) if p.is_file()}}}
 (a.output_dir/'SUMMARY.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); files=[p for p in a.output_dir.rglob('*') if p.is_file() and p.name!='SHA256SUMS']; (a.output_dir/'SHA256SUMS').write_text('\n'.join(f'{sha(p)}  {p.relative_to(a.output_dir)}' for p in sorted(files))+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
