#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, platform, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/0818_actual_baselines_novelty'; OUT.mkdir(parents=True, exist_ok=True)
BASE='/mnt/songzijun/models/pat-jj_harness-1-full/harness-1'
existing=ROOT/'outputs/btp_h100_4_baselines'
rows=[]
for method, seeds, gpu, status, reason in [
 ('OPSD_ACTION_PI','42,43','0,1','BLOCKED_ENVIRONMENT','No torch/Transformers/PEFT runtime exists in /opt or system Python; no actual training launched.'),
 ('OPHSD_FAITHFUL','42,43','2,3','BLOCKED_ENVIRONMENT','No torch/Transformers/PEFT runtime exists in /opt or system Python; old route_head cells are not accepted.'),
 ('MATCHED_TEXT_PRIVILEGE','42,43','4,5','BLOCKED_ENVIRONMENT','No torch/Transformers/PEFT runtime exists in /opt or system Python; no actual training launched.'),
 ('SEED_OR_OPID_FAITHFUL','42,43','6,7','BLOCKED_FAITHFUL_ADAPTATION','No portable Search skill analyzer/official adaptation contract is available; no simplified prompt substitute.'),
]: rows.append({'method':method,'seeds':seeds,'gpu_assignment':gpu,'status':status,'actual_model_weights':False,'student_inference_privilege':'NA','real_closed_loop':'not_started','reason':reason})
fields=list(rows[0])
with (OUT/'BASELINE_STATUS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
manifest={'run_id':'h1004_0818_actual_baselines_novelty','date':'2026-08-18','status':'blocked_before_gpu_launch','base_checkpoint':BASE,'output_dir':str(OUT),'visible_gpus':8,'gpu_memory_free_at_preflight':True,'runtime_preflight':{'system_python':'3.12.13','torch_import':'ModuleNotFoundError','opt_python_candidates':[],'opt_requirement':'all GPU environments must be under /opt'},'planned_cells':rows,'existing_reference_dir':str(existing),'no_route_head_substitution':True,'innovation_tracking':'not_primary_task'}
(OUT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
(OUT/'STATUS_LIVE.md').write_text('''# STATUS_LIVE\n\n- status: `BLOCKED_BEFORE_GPU_LAUNCH`\n- preflight: 8 GPUs visible and idle; no usable `/opt` torch/Transformers/PEFT runtime found\n- GPU cells: not started; no process cleanup required\n- no route-head or argmax bridge promoted\n- existing 0817 actual-model reference remains separate and is not a new 0818 run\n\n## Planned Eight-Card Allocation\n\n| GPU | method | seed | status |\n|---:|---|---:|---|\n| 0 | OPSD_ACTION_PI | 42 | BLOCKED_ENVIRONMENT |\n| 1 | OPSD_ACTION_PI | 43 | BLOCKED_ENVIRONMENT |\n| 2 | OPHSD_FAITHFUL | 42 | BLOCKED_ENVIRONMENT |\n| 3 | OPHSD_FAITHFUL | 43 | BLOCKED_ENVIRONMENT |\n| 4 | MATCHED_TEXT_PRIVILEGE | 42 | BLOCKED_ENVIRONMENT |\n| 5 | MATCHED_TEXT_PRIVILEGE | 43 | BLOCKED_ENVIRONMENT |\n| 6 | SEED_OR_OPID_FAITHFUL | 42 | BLOCKED_FAITHFUL_ADAPTATION |\n| 7 | SEED_OR_OPID_FAITHFUL | 43 | BLOCKED_FAITHFUL_ADAPTATION |\n\nNo experiment process was launched because doing so without the actual runtime would produce invalid artifacts.\n''')
(OUT/'OPSD_TRAINING_CELLS.csv').write_text('status,method,seed,gpu,n_train,n_valid,loss_path,note\nblocked,OPSD_ACTION_PI,42,0,NA,NA,action_ce,environment unavailable\nblocked,OPSD_ACTION_PI,43,1,NA,NA,action_ce,environment unavailable\n')
(OUT/'OPHSD_TRAINING_CELLS.csv').write_text('status,method,seed,gpu,n_train,n_valid,loss_path,note\nblocked,OPHSD_FAITHFUL,42,2,NA,NA,action_ce,environment unavailable\nblocked,OPHSD_FAITHFUL,43,3,NA,NA,action_ce,environment unavailable\n')
(OUT/'MATCHED_TEXT_TRAINING_CELLS.csv').write_text('status,method,seed,gpu,n_train,n_valid,loss_path,note\nblocked,MATCHED_TEXT_PRIVILEGE,42,4,NA,NA,action_ce,environment unavailable\nblocked,MATCHED_TEXT_PRIVILEGE,43,5,NA,NA,action_ce,environment unavailable\n')
for name in ['OPSD_REAL_CLOSED_LOOP.csv','OPHSD_REAL_CLOSED_LOOP.csv','MATCHED_TEXT_REAL_CLOSED_LOOP.csv']:
 (OUT/name).write_text('status,method,seed,n,reward,trajectory_recall,curated_evidence_recall,final_answer_recall,invalid_tool_rate,turns\nnot_started,NA,NA,NA,NA,NA,NA,NA,NA,NA\n')
(OUT/'SEED_OPID_STATUS.md').write_text('''# SEED_OPID_STATUS\n\n- status: `BLOCKED_FAITHFUL_ADAPTATION`\n- No official Search skill extraction/analyzer and no portable real closed-loop adaptation contract is available in this checkout.\n- Simplified prompt, route-head, or bridge substitutions are prohibited.\n- GPU6/7 were not repurposed because the required fallback would also need the unavailable actual runtime.\n''')
(OUT/'CLOSEST_RECENT_BASELINE_STATUS.md').write_text('''# CLOSEST_RECENT_BASELINE_STATUS\n\nThe closest completed local evidence is the 0817 H100-1 actual-LoRA closed-loop reference in `outputs/btp_h100_4_baselines/h1001_actual_lora_sources/`. It is not a new 0818 baseline run. The 0817 Matched Text and OPHSD artifacts are explicitly route-level/blocked and are not promoted.\n''')
# Required table files remain explicit NA rather than fabricated scores.
main=[{'method':'BASE_STUDENT_REFERENCE','status':'reference_existing_0817','actual_model_weights':True,'student_inference_privilege':False,'real_closed_loop':'available_in_existing_reference_only','reward':'0.3677561679','trajectory_recall':'0.1527314157','curated_evidence_recall':'NA','final_answer_recall':'0.1379154266','invalid_tool_rate':'0.0','turns':'8.46484375'},*[{"method":r['method'],"status":r['status'],"actual_model_weights":r['actual_model_weights'],"student_inference_privilege":r['student_inference_privilege'],"real_closed_loop":r['real_closed_loop'],"reward":'NA',"trajectory_recall":'NA',"curated_evidence_recall":'NA',"final_answer_recall":'NA',"invalid_tool_rate":'NA',"turns":'NA'} for r in rows]]
with (OUT/'MAIN_TABLE.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(main[0])); w.writeheader(); w.writerows(main)
(OUT/'MAIN_TABLE.md').write_text('# MAIN_TABLE\n\nNo new 0818 actual-model baseline row was run. Missing rows are `NA`, not zero. Existing 0817 Base Student is a reference only.\n')
for name in ['BASE_STUDENT_REAL_CLOSED_LOOP.csv','FULL_HARNESS_REFERENCE.csv','PAIRED_BOOTSTRAP.csv','COMPUTE_COST.csv','BASELINE_CASE_ANALYSIS.md']:
 p=OUT/name
 if name.endswith('.md'): p.write_text('# '+name[:-3]+'\n\nStatus: not available because actual-model runtime preflight failed before launch.\n')
 else: p.write_text('status,note\nnot_available,actual runtime unavailable before launch\n')
(OUT/'ACTUAL_BASELINE_PROTOCOL.md').write_text((OUT/'ACTUAL_BASELINE_PROTOCOL.md').read_text()+'\n## 2026-08-18 Preflight\n\nGPU visibility was present, but the required `/opt` runtime was absent. No GPU process was started.\n')
# SHA256
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(OUT/'SHA256SUMS').write_text('\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}' for p in files)+'\n')
(OUT/'H1004_0818_HANDOFF.json').write_text(json.dumps({'status':'blocked_before_gpu_launch','scientific_result':'DOES_OURS_BEAT_BASE_AND_STRONG_BASELINES? NOT_DETERMINED_FROM_0818; existing 0817 AUTO actual-LoRA did not beat Base.','novelty_result':'PENDING_LITERATURE_MATRIX','faithful_baselines_run':False,'environment_blocker':'No usable /opt torch/Transformers/PEFT runtime','generated_files':sorted(p.name for p in OUT.iterdir())},indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'written','out':str(OUT),'files':len(files)}))
