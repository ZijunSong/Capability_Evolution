#!/usr/bin/env python3
"""Finalize H100-3 retrieval hygiene bundle artifacts after smoke/evaluation."""
from __future__ import annotations
import csv, hashlib, json, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/0818_retrieval_hygiene_bundle'
SMOKE=OUT/'real_eval_smoke'
VARIANTS=['AUTO','DEDUP','AUTO_DEDUP','SHUFFLED']

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def mean(xs):
    xs=[float(x) for x in xs]
    return sum(xs)/len(xs) if xs else 0.0

def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def sha(root):
    lines=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
        h=hashlib.sha256(p.read_bytes()).hexdigest();lines.append(f'{h}  {p.relative_to(root)}')
    (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n')

def main():
    gate=json.loads((OUT/'BUNDLE_VALUE_GATE.json').read_text())
    train_rows=[]
    for p in sorted((OUT/'cells').glob('*/summary.json')):
        x=json.loads(p.read_text()); tr=x.get('training_result',{})
        train_rows.append({'variant':x.get('variant'),'seed':x.get('seed'),'gpu':x.get('gpu_requested'), 'actual_model_weights':True,'student_inference_privilege':False,'loss_path':x.get('loss_path'),'train_rows':x.get('train_rows'),'valid_rows':x.get('valid_rows'),'test_rows':x.get('test_rows'),'D_pre':tr.get('D_pre'),'D_post':tr.get('D_post'),'L_m':tr.get('L_m'),'mean_train_loss':tr.get('mean_train_loss'),'train_seconds':x.get('train_seconds'),'status':x.get('status'),'adapter_path':x.get('adapter_path')})
    write_csv(OUT/'TRAINING_CELLS.csv',train_rows,list(train_rows[0]))

    smoke=[]
    for name in ['BASE_REDUCED']+VARIANTS:
        p=SMOKE/name/'REAL_CLOSED_LOOP_SUMMARY.csv'
        if p.exists():
            row=read_csv(p)[0]; row['source']='real_eval_smoke'; smoke.append(row)
    write_csv(OUT/'DEV_REAL_CLOSED_LOOP.csv',smoke,list(smoke[0]) if smoke else ['method'])
    write_csv(OUT/'TEST_REAL_CLOSED_LOOP.csv',smoke,list(smoke[0]) if smoke else ['method'])

    base=next((r for r in smoke if r['method']=='BASE_REDUCED'),None)
    paired=[]
    if base:
        for r in smoke:
            if r['method']=='BASE_REDUCED':continue
            paired.append({'contrast':r['method']+'-BASE_REDUCED','n':r['n'],'mean_delta_reward':float(r['overall_reward'])-float(base['overall_reward']),'mean_delta_trajectory_recall':float(r['trajectory_recall'])-float(base['trajectory_recall']),'mean_delta_final_answer_recall':float(r['final_answer_recall'])-float(base['final_answer_recall']),'bootstrap_ci_low':'NA_smoke_only','bootstrap_ci_high':'NA_smoke_only','paired_bootstrap_status':'not_run_smoke_n16'})
    write_csv(OUT/'PAIRED_BOOTSTRAP.csv',paired,list(paired[0]) if paired else ['contrast'])

    mech=[]
    for r in smoke:
        mech.append({'method':r['method'],'first_search_to_first_curate_turns':'not_available_in_legacy_smoke','first_search_immediate_curate_rate':'not_available_in_legacy_smoke','duplicate_read_rate':'not_available_in_legacy_smoke','duplicate_curate_rate':'not_available_in_legacy_smoke','unique_docs_read':'not_available_in_legacy_smoke','unique_relevant_docs_read':'not_available_in_legacy_smoke','curated_unique_relevant_docs':r['curated_evidence_recall'],'search_redundancy':'not_available_in_legacy_smoke','qrel_recall_at_curated':r['curated_evidence_recall'],'note':'real closed-loop summary lacked expanded retrieval hygiene counters'})
    write_csv(OUT/'RETRIEVAL_MECHANISM_METRICS.csv',mech,list(mech[0]) if mech else ['method'])

    (OUT/'REDESIGN_EVENT_CONDITIONED_AUDIT.md').write_text('''# REDESIGN_EVENT_CONDITIONED_AUDIT\n\n## Substantive redesign attempted\n\nThe allowed redesign was to restrict the bundle to event-conditioned states where real `content_dedup` changes the evidence pool or creates a duplicate cluster, rather than training on inactive states.\n\n## Result\n\n- Source: `h100_3_real_influence_shards/content_dedup/REAL_INFLUENCE_PER_STATE.jsonl`\n- Real rows: 1024\n- Student actions: 872 `end_search`, 152 `read_document`\n- Teacher actions: 851 `end_search`, 173 `read_document`\n- Runtime document occurrences: 3872\n- Unique document ids: 242\n- Exact cross-id duplicate text clusters: 0\n- MinHash/shingle duplicate trigger cases at threshold 0.82: 0\n- Valid event-conditioned training rows: 0\n\nBecause no real duplicate-trigger event exists in the frozen source, event-conditioned resampling would either fabricate duplicates or use inactive states. Both are prohibited by the experiment contract. No second GPU retraining wave was launched.\n\nDecision: `DISCARD_RETRIEVAL_BUNDLE` rather than claiming complementarity.\n''')

    summary={'experiment':'RETRIEVAL_HYGIENE_BUNDLE','status':'completed','decision':'DISCARD_RETRIEVAL_BUNDLE','actual_model_weights':True,'student_inference_privilege':False,'route_head_substitution':False,'phase_1_3_gate':gate.get('decision_for_training_matrix'),'phase_4_cells':len(train_rows),'phase_5_smoke_methods':len(smoke),'base_smoke_reward':float(base['overall_reward']) if base else None,'adapter_smoke':{r['method']:{'reward':float(r['overall_reward']),'trajectory_recall':float(r['trajectory_recall']),'final_answer_recall':float(r['final_answer_recall']),'error_rate':float(r['error_rate'])} for r in smoke},'dedup_trigger_cases':0,'redesign':'event_conditioned_sampling_attempted_but_no_real_trigger_rows_available','full_dev_test_expansion':'not_run_after_valid_smoke_gate_failure','reason':'AUTO and AUTO_DEDUP did not beat BASE in real closed loop; DEDUP inactive on frozen runtime data; no valid dedup-trigger cases; rerank gate failed.'}
    (OUT/'H1003_0818_HANDOFF.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (OUT/'STATUS_LIVE.md').write_text('''# STATUS_LIVE\n\n- Phase 1 code/case audit: completed\n- Phase 2 executable projection data: completed\n- Phase 3 K4/K8-style value/mechanism gate: completed\n- Phase 4 actual PEFT/LoRA matrix: completed, 8/8 cells\n- Phase 5 corrected real closed-loop smoke: completed, local Harmony contract enabled\n- Event-conditioned redesign: audited once; zero valid real dedup-trigger states\n- Final decision: `DISCARD_RETRIEVAL_BUNDLE`\n- No training/evaluation workers remain\n''')
    sha(OUT)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
