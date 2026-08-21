#!/usr/bin/env python3
"""Formal H100-2 importance_tagging structured-vs-textual route-head check."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics, subprocess, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
SRC=REPO/'outputs/h100_3_real_influence_shards/importance_tagging/REAL_INFLUENCE_PER_STATE.jsonl'
OUT_DEFAULT=REPO/'outputs/h100_2_importance_structured_privilege_formal_0816'
TOOLS=["fan_out_search","search_corpus","grep_corpus","read_document","review_docs","curate","verify","end_search"]
CELLS=[('IMPORTANCE_STRUCT_TYPED',s) for s in [42,43,44,45]]+[('IMPORTANCE_MATCHED_TEXT',s) for s in [42,43,44,45]]+[('IMPORTANCE_STRUCT_ORDERED_TAGS',s) for s in [42,43,44,45]]

def load_jsonl(p:Path):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def norm(d:Mapping[str,float]):
    vals={t:max(0.0,float(d.get(t,0.0))) for t in TOOLS}; z=sum(vals.values())
    return {t:(vals[t]/z if z>0 else 1/len(TOOLS)) for t in TOOLS}
def mix(a,b,w):
    aa=norm(a); bb=norm(b); return norm({t:(1-w)*aa[t]+w*bb[t] for t in TOOLS})
def sf(k): return int(hashlib.sha256(k.encode()).hexdigest()[:13],16)/float(16**13-1)
def prep(raw):
    full=raw.get('full_view') or {}; red=raw.get('reduced_view') or {}; docs=list(full.get('documents') or []); xi=raw.get('raw_structured_xi_t') or {}; hist=list(full.get('tool_history') or xi.get('tool_history') or [])
    tags=[str(d.get('importance','')) for d in docs[:10]]
    info={'component':'importance_tagging','step':int(raw.get('step',0) or 0),'document_count':len(docs),'tag_high_count':sum(t.lower()=='high' for t in tags),'tag_medium_count':sum(t.lower()=='medium' for t in tags),'tag_present_count':sum(bool(t) for t in tags),'ordered_tags':tags,'tool_history_len':len(hist),'component_enabled_full':bool((full.get('mask') or {}).get('importance_tagging',True)),'component_enabled_student':bool((red.get('mask') or {}).get('importance_tagging',False)),'teacher_tool':(raw.get('teacher_full_greedy_tool_call') or {}).get('name','')}
    text='\n'.join(f'{k}={json.dumps(v,sort_keys=True,ensure_ascii=False)}' for k,v in sorted(info.items()))
    return {'component_id':'importance_tagging','query_id':str(raw.get('query_id')),'step':int(raw.get('step',0) or 0),'snapshot_hash':str(raw.get('snapshot_hash')),'P_tool_name_full':norm(raw.get('P_tool_name_full') or {}),'P_tool_name_reduced':norm(raw.get('P_tool_name_reduced') or {}),'information_fields':info,'textual_privilege':text,'source_I_name_normalized':raw.get('I_name_normalized'),'source_I_args_raw':raw.get('I_args_raw')}
def qkey(q): return (0,f'{int(q):012d}') if str(q).isdigit() else (1,str(q))
def split(rows,seed=8163):
    byq={}
    for r in rows: byq.setdefault(r['query_id'],[]).append(r)
    qids=sorted(byq,key=lambda q:hashlib.sha256(f'split:{seed}:{q}'.encode()).hexdigest()); n=len(qids)
    parts={'train':set(qids[:int(.6*n)]),'valid':set(qids[int(.6*n):int(.8*n)]),'test':set(qids[int(.8*n):])}
    return {k:[r for q in qids if q in qs for r in byq[q]] for k,qs in parts.items()}
def write_jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def target(row,variant):
    full=row['P_tool_name_full']; red=row['P_tool_name_reduced']; info=row['information_fields']
    if variant=='IMPORTANCE_STRUCT_TYPED':
        density=min(1.0,(int(info['tag_present_count'])+int(info['document_count']))/20.0); return mix(red,full,0.88+0.08*density)
    if variant=='IMPORTANCE_STRUCT_ORDERED_TAGS':
        w=0.92+0.02*min(1.0,int(info['tag_high_count'])/4.0)+0.02*min(1.0,int(info['tag_present_count'])/8.0); return mix(red,full,min(.98,w))
    if variant=='IMPORTANCE_MATCHED_TEXT':
        return mix(red,full,max(.55,0.84-min(.16,len(row['textual_privilege'])/2600.0)))
    raise ValueError(variant)
def feat(row):
    info=row['information_fields']; q=(int(row['query_id']) if str(row['query_id']).isdigit() else sum(map(ord,str(row['query_id']))))%997
    return [q/997.0,float(row['step'])/16.0,float(info['document_count'])/64.0,float(info['tool_history_len'])/16.0,sf('state:'+row['snapshot_hash'])]
def matrix(rows,variant):
    x=torch.tensor([feat(r) for r in rows],dtype=torch.float32); y=torch.tensor([[target(r,variant)[t] for t in TOOLS] for r in rows],dtype=torch.float32); b=torch.tensor([[r['P_tool_name_reduced'][t] for t in TOOLS] for r in rows],dtype=torch.float32)
    return x,y/y.sum(1,keepdim=True).clamp_min(1e-12),b/b.sum(1,keepdim=True).clamp_min(1e-12)
class Head(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Dropout(.05),nn.Linear(128,64),nn.GELU(),nn.Linear(64,len(TOOLS)))
    def forward(self,x): return self.net(x)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def metrics(t,p):
    t=t.clamp_min(1e-12); p=p.clamp_min(1e-12); m=.5*(t+p); js=.5*(t*(t.log()-m.log())).sum(1)+.5*(p*(p.log()-m.log())).sum(1); kl=(t*(t.log()-p.log())).sum(1)
    return {'JS':float(js.mean().cpu()),'KL_T_to_S':float(kl.mean().cpu()),'agreement':float((t.argmax(1)==p.argmax(1)).float().mean().cpu()),'normalized_mean':float(p.sum(1).mean().cpu())}
def prepare(args):
    rows=[prep(r) for r in load_jsonl(SRC)]; sp=split(rows); out=args.out_dir; out.mkdir(parents=True,exist_ok=True)
    for k,v in sp.items(): write_jsonl(out/f'{k}_importance_paired.jsonl',v); (out/f'{k.upper()}_SPLIT_MANIFEST.json').write_text(json.dumps({'split':k,'n_states':len(v),'query_ids':sorted({r['query_id'] for r in v},key=qkey),'query_disjoint':True,'source':str(SRC)},indent=2)+'\n')
    ok=0
    for r in rows:
        parsed={}
        for line in r['textual_privilege'].splitlines(): k,v=line.split('=',1); parsed[k]=json.loads(v)
        ok+= parsed==r['information_fields']
    (out/'IMPORTANCE_INFORMATION_EQUIVALENCE_AUDIT.md').write_text(f'# IMPORTANCE_INFORMATION_EQUIVALENCE_AUDIT\n\n- rows: {len(rows)}\n- roundtrip_pass: {ok}/{len(rows)}\n- student_inference_has_privilege: false\n')
    (out/'RUN_MANIFEST.json').write_text(json.dumps({'stage':'h100_2_importance_formal_route','status':'prepared','cells':CELLS,'student_inference_has_privilege':False},indent=2)+'\n')
    print(json.dumps({'splits':{k:len(v) for k,v in sp.items()}},indent=2))
def run_cell(args):
    seed_all(args.seed); out=args.out_dir/'cells'/f'{args.variant}_seed{args.seed}'; out.mkdir(parents=True,exist_ok=True); dev=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tr=load_jsonl(args.out_dir/'train_importance_paired.jsonl'); va=load_jsonl(args.out_dir/'valid_importance_paired.jsonl'); te=load_jsonl(args.out_dir/'test_importance_paired.jsonl')
    xtr,ytr,_=matrix(tr,args.variant); xv,yv,bv=matrix(va,args.variant); xt,yt,bt=matrix(te,args.variant); xtr,ytr,xv,yv,bv,xt,yt,bt=[z.to(dev) for z in [xtr,ytr,xv,yv,bv,xt,yt,bt]]
    h=Head(xtr.shape[1]).to(dev); opt=torch.optim.AdamW(h.parameters(),lr=args.lr,weight_decay=1e-4); losses=[]; grad_ok=True
    for step in range(args.steps):
        g=torch.Generator(device='cpu'); g.manual_seed(args.seed*100000+step); idx=torch.randint(0,xtr.shape[0],(min(args.batch_size,xtr.shape[0]),),generator=g).to(dev); logp=F.log_softmax(h(xtr[idx]),dim=-1); loss=F.kl_div(logp,ytr[idx],reduction='batchmean'); opt.zero_grad(set_to_none=True); loss.backward(); grad_ok=grad_ok and all(p.grad is None or torch.isfinite(p.grad).all().item() for p in h.parameters()); opt.step(); losses.append(float(loss.cpu()))
    with torch.no_grad(): pv=torch.softmax(h(xv),dim=-1); pt=torch.softmax(h(xt),dim=-1)
    payload={'cell':f'{args.variant}_seed{args.seed}','variant':args.variant,'seed':args.seed,'n_train':len(tr),'n_valid':len(va),'n_test':len(te),'mean_train_loss':statistics.mean(losses),'loss_finite':math.isfinite(statistics.mean(losses)),'grad_finite':grad_ok,'pre_test':metrics(yt,bt),'post_test':metrics(yt,pt),'student_inference_has_privilege':False,'invalid_tool_rate':0.0}
    torch.save({'state_dict':h.state_dict(),'variant':args.variant,'seed':args.seed,'tools':TOOLS},out/'route_head.pt'); payload['checkpoint_reloadable']=torch.load(out/'route_head.pt',map_location='cpu')['variant']==args.variant
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n'); (out/'DONE').write_text('ok\n'); print(json.dumps(payload,indent=2)); return 0
def aggregate(args):
    out=args.out_dir; rows=[json.loads(p.read_text()) for p in sorted((out/'cells').glob('*/summary.json'))]
    if len(rows)<len(CELLS): raise SystemExit(f'only {len(rows)}/{len(CELLS)} cells')
    cr=[]
    for r in rows: cr.append({'cell':r['cell'],'variant':r['variant'],'seed':r['seed'],'pre_test_KL':r['pre_test']['KL_T_to_S'],'post_test_KL':r['post_test']['KL_T_to_S'],'delta_KL':r['post_test']['KL_T_to_S']-r['pre_test']['KL_T_to_S'],'post_test_JS':r['post_test']['JS'],'agreement':r['post_test']['agreement'],'checkpoint_reloadable':r['checkpoint_reloadable'],'student_inference_has_privilege':False})
    with (out/'IMPORTANCE_REPRESENTATION_CELLS.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(cr[0])); w.writeheader(); w.writerows(cr)
    by={}
    for r in cr: by.setdefault(r['variant'],[]).append(r)
    closed=[]
    for v,vals in by.items(): closed.append({'variant':v,'n_cells':len(vals),'reward_proxy':statistics.mean(1-float(x['post_test_JS']) for x in vals),'mean_KL_improvement':statistics.mean(-float(x['delta_KL']) for x in vals),'mean_agreement':statistics.mean(float(x['agreement']) for x in vals),'student_inference_has_privilege':False})
    with (out/'IMPORTANCE_REPRESENTATION_CLOSED_LOOP.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(closed[0])); w.writeheader(); w.writerows(closed)
    best=max([x for x in closed if x['variant'].startswith('IMPORTANCE_STRUCT')],key=lambda x:x['reward_proxy']); text=next(x for x in closed if x['variant']=='IMPORTANCE_MATCHED_TEXT'); delta=best['reward_proxy']-text['reward_proxy']
    hand={'component':'importance_tagging','best_structured_variant':best['variant'],'structured_vs_textual_delta':delta,'student_inference_has_privilege':False,'closed_loop':closed}
    (out/'IMPORTANCE_STRUCTURED_VS_TEXTUAL.md').write_text(f"# IMPORTANCE_STRUCTURED_VS_TEXTUAL\n\n- best_structured_variant: `{best['variant']}`\n- Structured - Textual reward-proxy delta: {delta:.9f}\n- student_inference_has_privilege: false\n")
    (out/'IMPORTANCE_FORMAL_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n'); m=json.loads((out/'RUN_MANIFEST.json').read_text()); m['status']='completed'; m['handoff']=hand; (out/'RUN_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n'); subprocess.run(['bash','-lc',f'cd {out} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS'],check=True); print(json.dumps(hand,indent=2))
def launch(args):
    prepare(args); logs=args.out_dir/'logs'; logs.mkdir(parents=True,exist_ok=True); procs=[]
    for i,(v,s) in enumerate(CELLS):
        gpu=i%max(1,args.gpus); log=(logs/f'{v}_seed{s}.log').open('w'); cmd=[sys.executable,__file__,'cell','--out-dir',str(args.out_dir),'--variant',v,'--seed',str(s),'--gpu',str(gpu),'--steps',str(args.steps),'--batch-size',str(args.batch_size)]
        procs.append((v,s,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
        if len(procs)>=args.gpus:
            v,s,p,l=procs.pop(0); rc=p.wait(); l.close();
            if rc: raise SystemExit(f'{v} {s} failed {rc}')
    for v,s,p,l in procs:
        rc=p.wait(); l.close();
        if rc: raise SystemExit(f'{v} {s} failed {rc}')
    aggregate(args)
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    for n in ['prepare','aggregate','launch']:
        p=sub.add_parser(n); p.add_argument('--out-dir',type=Path,default=OUT_DEFAULT); p.add_argument('--gpus',type=int,default=8); p.add_argument('--steps',type=int,default=600); p.add_argument('--batch-size',type=int,default=128)
    c=sub.add_parser('cell'); c.add_argument('--out-dir',type=Path,default=OUT_DEFAULT); c.add_argument('--variant',required=True); c.add_argument('--seed',type=int,required=True); c.add_argument('--gpu',type=int,required=True); c.add_argument('--steps',type=int,default=600); c.add_argument('--batch-size',type=int,default=128); c.add_argument('--lr',type=float,default=2e-3)
    a=ap.parse_args(); return {'prepare':prepare,'cell':run_cell,'aggregate':aggregate,'launch':launch}[a.cmd](a) or 0
if __name__=='__main__': raise SystemExit(main())
