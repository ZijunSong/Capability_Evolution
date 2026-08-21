#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_divergence

def load(p):
    with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def action_text(a):
    return f"to={a.get('name') or a.get('tool_name') or 'end_search'}\n{json.dumps(a.get('arguments') or a.get('parameters') or {},ensure_ascii=False,sort_keys=True)}\n</tool_call>"
def next_rows(rows):
    out=[]
    for r in rows:
        if not r.get('next_student_action') or not r.get('next_teacher_action'): continue
        out.append({'row_id':r['row_id']+'_next','query_id':r['query_id'],'snapshot_hash':r['snapshot_hash'],
          'prompt_reduced':r['next_prompt_reduced'],'prompt_full':r['next_prompt_full'],
          'response_text':action_text(r['next_teacher_action']), 'student_inference_privilege':False})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--model-path',required=True); ap.add_argument('--train-jsonl',type=Path,required=True); ap.add_argument('--valid-jsonl',type=Path,required=True); ap.add_argument('--test-jsonl',type=Path,required=True); ap.add_argument('--variant',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--batch-size',type=int,default=1); ap.add_argument('--lr',type=float,default=1e-5); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: loading\n',encoding='utf-8')
    try:
        train,valid,test=load(args.train_jsonl),load(args.valid_jsonl),load(args.test_jsonl)
        backend=ScapeHFToolOPD(model_path=args.model_path,device_map='cuda:0',learning_rate=args.lr,anchor_weight=0.05,use_lora=True,lora_r=8,lora_alpha=16)
        loss_path='tool_token_kl' if 'OLD_REVERSE_ROUTE_KL' in args.variant else 'action_ce'
        span=backend.audit_tool_spans([r['response_text'] for r in train[:32]])
        pre=mean_divergence(backend,valid,loss_path=loss_path)
        losses=[]; t=time.time()
        for _ in range(args.epochs):
            for i in range(0,len(train),args.batch_size):
                losses.append(backend.train_step(train[i:i+args.batch_size],loss_path=loss_path))
                if 'PLUS_NEXTTURN_KL' in args.variant:
                    nr=next_rows(train[i:i+args.batch_size])
                    if nr: losses.append(backend.train_step(nr,loss_path='tool_token_kl'))
        post=mean_divergence(backend,test,loss_path=loss_path)
        adapter=args.out/'lora_checkpoint'; backend.save_pretrained(str(adapter))
        summary={'status':'completed','variant':args.variant,'seed':args.seed,'gpu_requested':args.gpu,'actual_model_weights':True,'student_inference_privilege':False,'route_head_substitution':False,'train_rows':len(train),'valid_rows':len(valid),'test_rows':len(test),'next_turn_rows':len(next_rows(train)),'span_audit':span,'loss_path':loss_path,'pre_divergence':pre,'heldout_divergence':post,'mean_train_loss':sum(x['loss'] for x in losses)/max(1,len(losses)),'n_train_steps':len(losses),'adapter_path':str(adapter),'next_turn_kl_applied':'PLUS_NEXTTURN_KL' in args.variant,'train_seconds':time.time()-t}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: completed\n- actual LoRA: true\n- no privilege inference: true\n',encoding='utf-8'); (args.out/'DONE').write_text('ok\n'); print(json.dumps(summary,ensure_ascii=False),flush=True); del backend; gc.collect()
    except Exception as e:
        (args.out/'FAILED.json').write_text(json.dumps({'error':str(e)},indent=2)+'\n'); (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: failed\n- error: '+str(e)+'\n',encoding='utf-8'); raise
if __name__=='__main__': main()
