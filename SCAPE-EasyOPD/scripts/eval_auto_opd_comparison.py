#!/usr/bin/env python3
import argparse, json, os, re, statistics
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file

TOOLS={"fan_out_search","search_corpus","grep_corpus","read_document","review_docs","curate","verify","end_search"}
def remap(s):
    return {k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for k,v in s.items()}
def load_adapter(model, p):
    try: return PeftModel.from_pretrained(model,p), 'peft'
    except Exception:
        cfg=json.loads((Path(p)/'adapter_config.json').read_text())
        lc=LoraConfig(task_type=cfg.get('task_type','CAUSAL_LM'),r=int(cfg.get('r',8)),lora_alpha=int(cfg.get('lora_alpha',16)),lora_dropout=float(cfg.get('lora_dropout',.05)),target_modules=list(cfg.get('target_modules') or ['q_proj','k_proj','v_proj','o_proj']),bias=cfg.get('bias','none'))
        m=get_peft_model(model,lc); miss,unexp=m.load_state_dict(remap(load_file(str(Path(p)/'adapter_model.safetensors'))),strict=False)
        bad=[x for x in miss if 'lora_' in x]+[x for x in unexp if 'lora_' in x]
        if bad: raise RuntimeError(str(bad[:5]))
        return m,'manual_safetensors'
def action(text):
    name=None
    for t in TOOLS:
        if re.search(rf'\b{re.escape(t)}\b',text or ''): name=t; break
    obj={}; ms=re.findall(r'\{.*?\}',text or '',re.S)
    if ms:
        try: obj=json.loads(ms[-1])
        except Exception: pass
    if isinstance(obj,dict) and obj.get('tool'):
        raw_tool=str(obj.get('tool'))
        aliases={'search':'search_corpus','search_corpus':'search_corpus','fanout_search':'fan_out_search','finish':'end_search'}
        name=aliases.get(raw_tool,raw_tool)
        obj={k:v for k,v in obj.items() if k!='tool'}
    return name,obj
def norm(name,obj):
    if name=='curate': return {'name':name,'arguments':{'add_ids':sorted(obj.get('add_ids') or []),'remove_ids':sorted(obj.get('remove_ids') or [])}}
    return {'name':name,'arguments':obj}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rows',required=True); ap.add_argument('--model',required=True); ap.add_argument('--condition',choices=['teacher','before','after'],required=True); ap.add_argument('--adapter'); ap.add_argument('--out',required=True); ap.add_argument('--gpu',default='0'); ap.add_argument('--limit',type=int,default=500); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    rows=[]
    for line in open(args.rows):
        if line.strip(): rows.append(json.loads(line))
        if len(rows)>=args.limit: break
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto')
    reload_path='base'
    if args.condition=='after': model,reload_path=load_adapter(model,args.adapter)
    model.eval(); out=[]
    for i,r in enumerate(rows):
        prompt=r['prompt_full'] if args.condition=='teacher' else r['prompt_reduced']
        msgs=[{'role':'system','content':'You are a SCAPE research agent. Return exactly one legal Harness-1 tool call as JSON.'},{'role':'user','content':prompt}]
        rendered=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt')
        if isinstance(rendered,dict) or hasattr(rendered,'input_ids'):
            ids=rendered['input_ids'] if isinstance(rendered,dict) else rendered.input_ids
        else:
            ids=rendered
        if ids.ndim == 1: ids=ids.unsqueeze(0)
        ids=ids.to(model.device)
        with torch.inference_mode(): outids=model.generate(input_ids=ids,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(outids[0,ids.shape[-1]:],skip_special_tokens=False)
        name,obj=action(text); pred=norm(name,obj); target=r.get('projectable_target') or {}
        exact=pred==norm(target.get('name'),target.get('arguments') or {})
        legal=name in TOOLS
        out.append({'row':i,'state_uid':r.get('state_uid'),'condition':args.condition,'generated':text,'predicted':pred,'target':norm(target.get('name'),target.get('arguments') or {}),'legal':legal,'exact_projected_target':exact})
        if (i+1)%25==0: print(json.dumps({'condition':args.condition,'done':i+1,'n':len(rows)}),flush=True)
    summary={'status':'completed','condition':args.condition,'n':len(out),'legal_rate':sum(x['legal'] for x in out)/len(out),'exact_projected_target_rate':sum(x['exact_projected_target'] for x in out)/len(out),'reload_path':reload_path,'student_inference_privilege':False if args.condition!='teacher' else 'teacher_only_privilege','metric':'projected_action_exact_match_on_frozen_OPD_VALID_ROWS'}
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'summary':summary,'rows':out},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
