#!/usr/bin/env python3
"""Construct qrel-compatible real-corpus token-pressure snapshots from a frozen query cohort."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from scape.state.snapshot import capture_snapshot

def load_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def norm(x): return str(x).split('_',1)[0]

def format_pressure_marker(used: int, budget: int) -> str:
 pct=int(100.0*used/max(1,budget)); flag=''
 if pct>=90: flag=' CRITICAL - end_search NOW'
 elif pct>=75: flag=' warning: finish up soon'
 elif pct>=60: flag=' over halfway'
 return f'[Context: {used}/{budget}{flag}]'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--source-cache',type=Path,required=True); ap.add_argument('--corpus',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--budget',type=int,default=30720); ap.add_argument('--n',type=int,default=128); args=ap.parse_args()
 tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
 corpus={}
 for line in args.corpus.open():
  r=json.loads(line); corpus[str(r.get('id') or r.get('docid'))]=str(r.get('text') or r.get('contents') or '')
 source=load_jsonl(args.source_cache); selected=[]
 bins=[('over_half',.60,.75),('warning',.75,.90),('critical',.90,.98)]
 # Each source state contributes at most one pressure state; vary real document text
 # retention rather than adding synthetic text or rewriting budget values.
 for idx,src in enumerate(source):
  wm=src['snapshot']['working_memory']; docs=[]
  for d in wm.get('documents',[]):
   did=str(d['id']); text=corpus.get(did)
   if not text: raise RuntimeError(f'missing real corpus document {did}')
   docs.append({'id':did,'text':text})
  chosen=None
  frac=(.62,.78,.92)[idx%3]
  label='over_half' if frac<.7 else 'warning' if frac<.9 else 'critical'
  hist=[{'step':j,'action':{'name':'search_corpus','arguments':{'query':wm['query']}}} for j in range(max(1,idx%8))]
  target=int(args.budget*frac)
  per_doc=max(64,(target-1200)//max(1,len(docs)))
  cand=[]
  for d in docs:
   ids=tok.encode(d['text'],add_special_tokens=False)[:per_doc]
   cand.append({'id':d['id'],'text':tok.decode(ids,skip_special_tokens=True)})
  view={'query_id':src['query_id'],'step':idx%8,'documents':cand,'curated_docs':cand[:2],'curated_ids':[d['id'] for d in cand[:2]],'tool_history':hist,'token_budget_marker':'','verify_available':False}
  prompt='You are Harness-1 choosing the next BrowseComp tool call. Choose exactly one tool and JSON arguments. STATE:\n'+json.dumps(view,ensure_ascii=False,sort_keys=True)
  used=len(tok.encode(prompt,add_special_tokens=False))
  # One proportional correction keeps the final tokenizer-measured prompt in-bin.
  if used:
   per_doc=max(64,int(per_doc*target/used))
   cand=[]
   for d in docs:
    ids=tok.encode(d['text'],add_special_tokens=False)[:per_doc]
    cand.append({'id':d['id'],'text':tok.decode(ids,skip_special_tokens=True)})
   view['documents']=cand; view['curated_docs']=cand[:2]; view['curated_ids']=[d['id'] for d in cand[:2]]
   prompt='You are Harness-1 choosing the next BrowseComp tool call. Choose exactly one tool and JSON arguments. STATE:\n'+json.dumps(view,ensure_ascii=False,sort_keys=True)
   used=len(tok.encode(prompt,add_special_tokens=False))
  if int(args.budget*frac)<=used<int(args.budget*(frac+.03)): chosen=(cand,hist,used,label)
  if not chosen: continue
  cand,hist,used,label=chosen; qid=str(src['query_id']); curated=[d['id'] for d in cand[:2]]
  snap=capture_snapshot(query_id=qid,step=idx%8,harness_mask={'token_budget_marker':False},working_memory={'query':wm['query'],'documents':cand,'curated_docs':cand[:2],'curated_ids':curated,'curated_importance':{},'evidence_graph':{},'token_budget_marker':format_pressure_marker(used,args.budget),'rerank_instruction':'prefer direct evidence and diverse corroborating sources','auto_populate_seed':[],'chunk_neighbors':[d['id'] for d in cand[2:6]],'verified_supported':[],'verified_unsupported':[],'pressure_metadata':{'used_tokens':used,'budget':args.budget,'threshold_60':int(args.budget*.60),'threshold_75':int(args.budget*.75),'threshold_90':int(args.budget*.90),'pressure_bin':label,'measurement':'Harness-1 local tokenizer on real-corpus context','real_documents':True}},tool_history=hist,observations=[],metadata={'backend':'real_corpus_tokenizer_pressure','pressure_bin':label})
  selected.append({'component':'token_budget_marker','query_id':qid,'turn_id':idx%8,'snapshot':snap.to_dict(),'snapshot_hash':snap.content_hash(),'pressure_metadata':snap.working_memory['pressure_metadata'],'gold_evidence_ids':sorted(src.get('gold_evidence_ids',[]))})
  if len(selected)>=args.n: break
 if len(selected)<args.n: raise SystemExit(f'only {len(selected)} pressure states available')
 args.out.parent.mkdir(parents=True,exist_ok=True)
 with args.out.open('w') as f:
  for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps({'n':len(selected),'bins':{b:sum(x['pressure_metadata']['pressure_bin']==b for x in selected) for b,_,_ in bins},'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__': main()
