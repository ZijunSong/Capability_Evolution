#!/usr/bin/env python3
"""One actual LoRA cell for H100-3 retrieval hygiene bundle."""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_divergence, run_tool_opd_train
from scape.training.tool_mask import tool_loss_mask_from_response

def load(p):
 rows=[]
 with p.open(encoding="utf-8") as f:
  for l in f:
   if l.strip(): rows.append(json.loads(l))
 return rows

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--model-path',required=True); ap.add_argument('--train-jsonl',type=Path,required=True); ap.add_argument('--valid-jsonl',type=Path,required=True); ap.add_argument('--test-jsonl',type=Path,required=True); ap.add_argument('--component-id',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--batch-size',type=int,default=1); ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--loss-path',default='tool_token_kl'); a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 (a.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: running\n',encoding='utf-8')
 try:
  train,valid,test=load(a.train_jsonl),load(a.valid_jsonl),load(a.test_jsonl)
  backend=ScapeHFToolOPD(model_path=a.model_path,device_map='cuda:0',learning_rate=a.lr,anchor_weight=0.05,use_lora=True,lora_r=8,lora_alpha=16)
  span=backend.audit_tool_spans([r['response_text'] for r in train[:32]])
  pre=mean_divergence(backend,valid,loss_path=a.loss_path)
  t=time.time(); result=run_tool_opd_train(backend,train,valid,loss_path=a.loss_path,epochs=a.epochs,batch_size=a.batch_size); secs=time.time()-t
  post=mean_divergence(backend,test,loss_path=a.loss_path)
  adapter=a.out/'lora_checkpoint'; backend.save_pretrained(str(adapter))
  summary={'status':'completed','variant':a.component_id,'seed':a.seed,'gpu_requested':a.gpu,'cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'actual_model_weights':True,'student_inference_privilege':False,'route_head_substitution':False,'train_rows':len(train),'valid_rows':len(valid),'test_rows':len(test),'loss_path':a.loss_path,'span_audit':span,'training_result':result,'heldout_divergence':post,'pre_divergence':pre,'train_seconds':secs,'adapter_path':str(adapter),'native_action_args_supervision':True}
  (a.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
  (a.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: completed\n- actual LoRA: true\n- no privilege inference: true\n',encoding='utf-8'); (a.out/'DONE').write_text('ok\n')
  print(json.dumps(summary,ensure_ascii=False),flush=True)
  del backend; gc.collect()
 except Exception as e:
  (a.out/'FAILED.json').write_text(json.dumps({'error':str(e)},indent=2)+'\n'); (a.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: failed\n- error: '+str(e)+'\n'); raise
if __name__=='__main__': main()
