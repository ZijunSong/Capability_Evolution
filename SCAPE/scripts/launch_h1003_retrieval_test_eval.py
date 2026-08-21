#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument('--out',type=Path,required=True)
  ap.add_argument('--query-manifest',type=Path,required=True)
  ap.add_argument('--n-queries',type=int,default=112)
  ap.add_argument('--seed',type=int,default=819)
  ap.add_argument('--split',default='test')
  ap.add_argument('--max-steps',type=int,default=6)
  ap.add_argument('--python',default='/opt/scape-h1003-hf-scorer/bin/python')
  ap.add_argument('--poll',type=int,default=60)
  args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
  base='/mnt/songzijun/models/pat-jj_harness-1-full/harness-1'
  specs=[
    (0,'AUTO42','outputs/0818_retrieval_hygiene_bundle/cells/auto_seed42/lora_checkpoint'),
    (1,'AUTO43','outputs/0818_retrieval_hygiene_bundle/cells/auto_seed43/lora_checkpoint'),
    (2,'DEDUP42','outputs/0818_retrieval_hygiene_bundle/cells/dedup_seed42/lora_checkpoint'),
    (3,'DEDUP43','outputs/0818_retrieval_hygiene_bundle/cells/dedup_seed43/lora_checkpoint'),
    (4,'AUTO_DEDUP42','outputs/0818_retrieval_hygiene_bundle/cells/auto_dedup_seed42/lora_checkpoint'),
    (5,'AUTO_DEDUP43','outputs/0818_retrieval_hygiene_bundle/cells/auto_dedup_seed43/lora_checkpoint'),
    (6,'SHUFFLED42','outputs/0818_retrieval_hygiene_bundle/cells/shuffled_seed42/lora_checkpoint'),
    (7,'SHUFFLED43','outputs/0818_retrieval_hygiene_bundle/cells/shuffled_seed43/lora_checkpoint'),
  ]
  procs=[]; entries=[]
  for gpu,label,adapter in specs:
    out=args.out/label; out.mkdir(parents=True,exist_ok=True)
    cmd=[args.python,str(ROOT/'scripts/run_btp_auto_lora_real_closed_loop.py'),'--base-model',base,'--out-dir',str(out),'--query-manifest',str(args.query_manifest),'--n-queries',str(args.n_queries),'--split',args.split,'--seed',str(args.seed),'--max-steps',str(args.max_steps),'--device-map','cuda:0','--dtype','bfloat16','--adapter',f'{label}={adapter}']
    log=(out/'worker.log').open('w')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'SCAPE_FORCE_LOCAL_HARMONY':'1'}
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    procs.append((gpu,label,p,log,out)); entries.append({'gpu':gpu,'label':label,'pid':p.pid,'out':str(out),'adapter':adapter})
    print(json.dumps({'launched':entries[-1]}),flush=True)
  (args.out/'FULL_EVAL_LAUNCH_MANIFEST.json').write_text(json.dumps({'status':'running','n_queries':args.n_queries,'seed':args.seed,'split':args.split,'query_manifest':str(args.query_manifest),'jobs':entries},indent=2)+'\n')
  while procs:
    alive=[]
    for gpu,label,p,log,out in procs:
      rc=p.poll()
      if rc is None:
        alive.append((gpu,label,p,log,out))
      else:
        log.close(); print(json.dumps({'finished':label,'gpu':gpu,'returncode':rc,'summary':str(out/'REAL_CLOSED_LOOP_SUMMARY.csv')}),flush=True)
    procs=alive
    if procs: time.sleep(args.poll)
  (args.out/'FULL_EVAL_LAUNCH_MANIFEST.json').write_text(json.dumps({'status':'completed','n_queries':args.n_queries,'seed':args.seed,'split':args.split,'query_manifest':str(args.query_manifest),'jobs':entries},indent=2)+'\n')
if __name__=='__main__': main()
