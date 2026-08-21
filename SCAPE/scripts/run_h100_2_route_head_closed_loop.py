#!/usr/bin/env python3
"""Real closed-loop BM25 execution for H100-2 route-head students.

This is not a same-state route proxy: each trained route head chooses tools over
multiple steps, the BM25 environment mutates, and final reward is computed from
executed state.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, subprocess, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

REPO=Path(__file__).resolve().parents[1]
SCOPE=Path('/mnt/songzijun/Capability_Evolution/SCOPE')
BCP=SCOPE/'external/BrowseComp-Plus'
TOOLS=["fan_out_search","search_corpus","grep_corpus","read_document","review_docs","curate","verify","end_search"]

class RouteHead(nn.Module):
    def __init__(self,d:int):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Dropout(.05),nn.Linear(128,64),nn.GELU(),nn.Linear(64,len(TOOLS)))
    def forward(self,x): return self.net(x)

def stable_float(key:str)->float: return int(hashlib.sha256(key.encode()).hexdigest()[:13],16)/float(16**13-1)
def doc_text(raw:Any)->str:
    if raw is None: return ''
    if isinstance(raw,str):
        try:
            o=json.loads(raw); return str(o.get('contents') or o.get('text') or raw) if isinstance(o,dict) else raw
        except Exception: return raw
    return str(raw)
def load_queries(p:Path):
    out={}
    for line in p.read_text(encoding='utf-8').splitlines():
        parts=line.split('\t');
        if len(parts)>=2: out[str(parts[0])]=parts[1]
    return out
def load_qrels(p:Path):
    out={}
    for line in p.read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if len(parts)>=3: out.setdefault(str(parts[0]),set()).add(str(parts[2]))
    return out
def recall(ids:list[str],gold:set[str])->float:
    if not gold: return 0.0
    norm={str(x).split('_',1)[0] for x in ids}; return len(norm & gold)/len(gold)

class LiveState:
    def __init__(self,qid,query,gold,searcher,component='auto_populate_first_search'):
        self.qid=str(qid); self.query=query; self.gold=set(gold); self.searcher=searcher; self.component=component; self.step=0; self.documents=[]; self.curated_ids=[]; self.read_ids=[]; self.verified_supported=[]; self.verified_unsupported=[]; self.history=[]; self.cost=0; self._search(query,10); self.curated_ids=[d['id'] for d in self.documents[:2]]
    def _search(self,q,k=20):
        self.documents=[{'id':str(h.docid),'text':doc_text(getattr(h,'raw',None) or '')[:1800]} for h in self.searcher.search(q,k)]; self.cost+=1
    def execute(self,action:Mapping[str,Any]):
        name=str(action.get('name') or 'end_search'); args=dict(action.get('arguments') or {})
        if name in {'search_corpus','fan_out_search'}:
            q=str((args.get('queries') or [args.get('query') or self.query])[0]); self._search(q,20)
        elif name=='grep_corpus':
            pat=str(args.get('pattern') or '').lower(); filt=[d for d in self.documents if pat and pat in d.get('text','').lower()]
            if filt: self.documents=filt+[d for d in self.documents if d not in filt]
            self.cost+=1
        elif name=='read_document':
            did=str(args.get('doc_id') or (self.documents[0]['id'] if self.documents else ''))
            if did and did not in self.read_ids: self.read_ids.append(did)
            self.cost+=1
        elif name=='review_docs':
            for did in args.get('doc_ids') or [d['id'] for d in self.documents[:5]]:
                if str(did) not in self.read_ids: self.read_ids.append(str(did))
            self.cost+=1
        elif name=='curate':
            for did in args.get('remove_ids') or []: self.curated_ids=[x for x in self.curated_ids if x!=str(did)]
            for did in args.get('add_ids') or []:
                if str(did) and str(did) not in self.curated_ids: self.curated_ids.append(str(did))
            self.cost+=1
        elif name=='verify':
            for did in args.get('doc_ids') or [self.curated_ids[0] if self.curated_ids else '']:
                sid=str(did)
                if sid.split('_',1)[0] in self.gold:
                    if sid not in self.verified_supported: self.verified_supported.append(sid)
                elif sid:
                    if sid not in self.verified_unsupported: self.verified_unsupported.append(sid)
            self.cost+=1
        self.history.append({'step':self.step,'action':{'name':name,'arguments':args}}); self.step+=1
    def metrics(self):
        unique=list(dict.fromkeys(self.curated_ids)); useful=[i for i in unique if i.split('_',1)[0] in self.gold]; red=max(0,len(self.curated_ids)-len(unique))/max(1,len(self.curated_ids)); cov=recall(unique,self.gold); vs=len(set(self.verified_supported)); vu=len(set(self.verified_unsupported)); obj=.45*cov+.20*(len(useful)/max(1,len(self.gold)))+.20*(vs/max(1,len(self.gold)))-.05*red-.015*self.cost-.03*vu
        return {'curated_evidence_recall':cov,'trajectory_recall':recall([d['id'] for d in self.documents],self.gold),'final_answer_recall':recall(unique[:3],self.gold),'overall_reward':obj,'tool_calls':float(self.cost),'verified_supported':float(vs),'unsupported_claims':float(vu)}

def feature(st:LiveState):
    q=(int(st.qid) if st.qid.isdigit() else sum(map(ord,st.qid)))%997
    prior_search=sum(1 for h in st.history if h['action']['name'] in {'fan_out_search','search_corpus','grep_corpus'})
    return [q/997.0, st.step/16.0, len(st.history)/16.0, len(st.documents)/64.0, prior_search/16.0, stable_float('live:'+st.qid+':'+str(st.step))]
def action_for_tool(tool,st:LiveState):
    docs=st.documents; curated=st.curated_ids; first=docs[0]['id'] if docs else ''
    if tool=='fan_out_search': return {'name':tool,'arguments':{'queries':[st.query,st.query+' evidence',st.query+' source']}}
    if tool=='search_corpus': return {'name':tool,'arguments':{'query':st.query}}
    if tool=='grep_corpus': return {'name':tool,'arguments':{'pattern':next((w for w in st.query.split() if len(w)>5), st.query[:8])}}
    if tool=='read_document': return {'name':tool,'arguments':{'doc_id':first}}
    if tool=='review_docs': return {'name':tool,'arguments':{'doc_ids':[d['id'] for d in docs[:5]]}}
    if tool=='curate': return {'name':tool,'arguments':{'add_ids':[d['id'] for d in docs[:4] if d['id'] not in curated][:2], 'remove_ids':[]}}
    if tool=='verify': return {'name':tool,'arguments':{'doc_ids':curated[:4] or ([first] if first else []),'claim':st.query[:160]}}
    return {'name':'end_search','arguments':{}}
def choose(head,st,device):
    x=torch.tensor([feature(st)],dtype=torch.float32,device=device)
    with torch.no_grad(): probs=torch.softmax(head(x),dim=-1)[0].detach().cpu().numpy()
    idx=int(np.argmax(probs)); return action_for_tool(TOOLS[idx],st), {TOOLS[i]:float(probs[i]) for i in range(len(TOOLS))}
def load_head(path:Path,device):
    ck=torch.load(path,map_location=device); h=RouteHead(6).to(device); h.load_state_dict(ck['state_dict']); h.eval(); return h

def select_qids(args,queries,qrels):
    eligible=sorted(set(queries)&set(qrels), key=lambda q: hashlib.sha256(f'closed:{args.seed}:{q}'.encode()).hexdigest())
    return eligible[:args.n_queries]
def run_method(method,ckpts,args,queries,qrels,searcher,device):
    qids=select_qids(args,queries,qrels); rows=[]
    heads=[load_head(p,device) for p in ckpts] if ckpts else []
    for qid in qids:
        st=LiveState(qid,queries[qid],qrels[qid],searcher,args.component); trace=[]
        for t in range(args.max_steps):
            if method=='BASE_REDUCED':
                tool='end_search' if t>=2 else ('read_document' if t==0 else 'review_docs'); action=action_for_tool(tool,st); dist={tool:1.0}
            else:
                head=heads[t % len(heads)]; action,dist=choose(head,st,device)
            st.execute(action); trace.append({'step':t,'action':action,'top_tool':max(dist,key=dist.get),'top_prob':max(dist.values())})
            if action['name']=='end_search': break
        m=st.metrics(); rows.append({'method':method,'query_id':qid,**m,'trace':trace,'student_inference_has_privilege':False,'runner':'real_closed_loop_bm25_route_head'})
    return rows
def aggregate(rows):
    methods=sorted({r['method'] for r in rows}); out=[]
    for m in methods:
        rs=[r for r in rows if r['method']==m]; out.append({'method':m,'n':len(rs),'overall_reward':statistics.mean(float(r['overall_reward']) for r in rs),'curated_evidence_recall':statistics.mean(float(r['curated_evidence_recall']) for r in rs),'trajectory_recall':statistics.mean(float(r['trajectory_recall']) for r in rs),'final_answer_recall':statistics.mean(float(r['final_answer_recall']) for r in rs),'tool_calls':statistics.mean(float(r['tool_calls']) for r in rs),'student_inference_has_privilege':False})
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--auto-dir',type=Path,required=True); ap.add_argument('--importance-dir',type=Path,default=None); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--n-queries',type=int,default=128); ap.add_argument('--max-steps',type=int,default=6); ap.add_argument('--seed',type=int,default=8164); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--component',default='auto_populate_first_search'); args=ap.parse_args()
    from pyserini.search.lucene import LuceneSearcher
    args.out_dir.mkdir(parents=True,exist_ok=True); device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    queries=load_queries(BCP/'topics-qrels/queries.tsv'); qrels=load_qrels(BCP/'topics-qrels/qrel_evidence.txt'); searcher=LuceneSearcher(str(BCP/'indexes/bm25'))
    auto_h=json.loads((args.auto_dir/'H1002_STRUCTURED_PRIVILEGE_HANDOFF.json').read_text())
    text_ckpts=sorted((args.auto_dir/'cells').glob('AUTO_MATCHED_TEXT_seed*/route_head.pt'))
    best=auto_h.get('best_structured_variant','AUTO_STRUCT_TYPED'); best_ckpts=sorted((args.auto_dir/'cells').glob(f'{best}_seed*/route_head.pt'))
    methods={'BASE_REDUCED':[], 'AUTO_MATCHED_TEXT':text_ckpts, best:best_ckpts}
    all_rows=[]
    for method,ckpts in methods.items(): all_rows.extend(run_method(method,ckpts,args,queries,qrels,searcher,device))
    with (args.out_dir/'REAL_CLOSED_LOOP_PER_QUERY.jsonl').open('w',encoding='utf-8') as f:
        for r in all_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=aggregate(all_rows)
    with (args.out_dir/'REAL_CLOSED_LOOP_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    by={r['method']:r for r in summary}; delta=by[best]['overall_reward']-by['AUTO_MATCHED_TEXT']['overall_reward']
    hand={'status':'completed_real_closed_loop_bm25','best_structured_variant':best,'structured_vs_textual_delta':delta,'structured_vs_base_delta':by[best]['overall_reward']-by['BASE_REDUCED']['overall_reward'],'textual_vs_base_delta':by['AUTO_MATCHED_TEXT']['overall_reward']-by['BASE_REDUCED']['overall_reward'],'student_inference_has_privilege':False,'n_queries':args.n_queries,'max_steps':args.max_steps,'summary':summary}
    (args.out_dir/'REAL_CLOSED_LOOP_HANDOFF.json').write_text(json.dumps(hand,indent=2,ensure_ascii=False)+'\n')
    (args.out_dir/'REAL_CLOSED_LOOP.md').write_text(f"# REAL_CLOSED_LOOP\n\n- status: completed_real_closed_loop_bm25\n- best_structured_variant: `{best}`\n- structured_vs_textual_delta: {delta}\n- student_inference_has_privilege: false\n- runner: real BM25 state progression with route-head tool choices\n")
    subprocess.run(['bash','-lc',f'cd {args.out_dir} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS'],check=True)
    print(json.dumps(hand,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
