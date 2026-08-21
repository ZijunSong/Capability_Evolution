#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any


def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    remapped = {}
    for key, value in raw_state.items():
        if key.endswith('.lora_A.weight'):
            remapped[key.replace('.lora_A.weight', '.lora_A.default.weight')] = value
        elif key.endswith('.lora_B.weight'):
            remapped[key.replace('.lora_B.weight', '.lora_B.default.weight')] = value
        else:
            remapped[key] = value
    return remapped


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows)>=limit:
                    break
    return rows


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--component', required=True)
    ap.add_argument('--method', choices=['PURE_OPD','RL_PLUS_OPD'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', required=True, help='CUDA device id or comma-separated ids visible to this cell')
    ap.add_argument('--model', default='/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507')
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--valid', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-limit', type=int, default=4500)
    ap.add_argument('--valid-limit', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--max-prompt-tokens', type=int, default=None, help='Pilot-only length filter; formal 5K rows remain unchanged')
    ap.add_argument('--loss-path', default='action_ce', choices=['action_ce','tool_token_kl','weighted_tool_token_kl','tool_name_only_kl','args_only_kl','full_response_kl','offpolicy_matched'])
    args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: starting\n', encoding='utf-8')
    os.environ.setdefault('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt')
    os.environ.setdefault('REQUESTS_CA_BUNDLE','/etc/ssl/certs/ca-certificates.crt')
    os.environ.setdefault('HF_HOME','/opt/hf-cache')
    random.seed(args.seed)
    try:
        from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_divergence
        rows=load_rows(args.train, args.train_limit)
        valid=load_rows(args.valid, args.valid_limit)
        n_rows_dropped_long_prompt = 0
        if args.max_prompt_tokens is not None:
            from transformers import AutoTokenizer
            filter_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
            def keep(row):
                return len(filter_tokenizer.encode(row['prompt_reduced'])) <= args.max_prompt_tokens
            before_train, before_valid = len(rows), len(valid)
            rows = [row for row in rows if keep(row)]
            valid = [row for row in valid if keep(row)]
            n_rows_dropped_long_prompt = (before_train - len(rows)) + (before_valid - len(valid))
        if not rows or not valid:
            raise RuntimeError('empty train/valid rows after pilot prompt-length filter')
        random.shuffle(rows)
        backend=ScapeHFToolOPD(model_path=args.model, device_map='auto', learning_rate=args.lr, anchor_weight=0.05, use_lora=True, lora_r=8, lora_alpha=16)
        span=backend.audit_tool_spans([r['response_text'] for r in rows[:64]])
        if not span.get('pass'):
            raise RuntimeError('tool span audit failed: '+json.dumps(span, ensure_ascii=False)[:1000])
        loss_path=args.loss_path
        pre=mean_divergence(backend, valid, loss_path=loss_path)
        losses=[]
        t0=time.time()
        for _ in range(args.epochs):
            for i in range(0, len(rows), args.batch_size):
                batch=rows[i:i+args.batch_size]
                losses.append(backend.train_step(batch, loss_path=loss_path))
                if args.method=='RL_PLUS_OPD':
                    # Minimal RL+OPD hook: same projected OPD plus a second anchor-like OPD step.
                    # Online GRPO rollout accounting remains separate and is not counted as 5K OPD states.
                    losses.append(backend.train_step(batch, loss_path='tool_token_kl'))
                if len(losses) % 100 == 0:
                    (args.out/'STATUS_LIVE.md').write_text(f'# STATUS_LIVE\n\n- status: training\n- steps: {len(losses)}\n- elapsed_sec: {time.time()-t0:.1f}\n', encoding='utf-8')
        post=mean_divergence(backend, valid, loss_path=loss_path)
        adapter=args.out/'lora_checkpoint'
        backend.save_pretrained(str(adapter))
        reload_error = 'peft_native_reload_skipped_for_qwen3_transformers_compatibility'
        reload_path = 'manual_safetensors_state_dict'
        del backend
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype='auto', trust_remote_code=True)
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
        reloaded = get_peft_model(base, cfg)
        raw_state = load_file(str(adapter / 'adapter_model.safetensors'))
        missing, unexpected = reloaded.load_state_dict(remap_lora_state_dict(raw_state), strict=False)
        bad_missing = [key for key in missing if 'lora_' in key]
        bad_unexpected = [key for key in unexpected if 'lora_' in key]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f'manual adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}')
        reloaded.eval()
        del base
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        adapter_reload_acceptance = {
            'adapter_reload_pass': True,
            'adapter_path': str(adapter),
            'reload_path': reload_path,
            'reload_error': reload_error,
        }
        (args.out/'ADAPTER_RELOAD_ACCEPTANCE.json').write_text(json.dumps(adapter_reload_acceptance, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
        summary={
            'status':'completed',
            'component':args.component,
            'method':args.method,
            'seed':args.seed,
            'gpu':args.gpu,
            'model':args.model,
            'canonical_student_base':'/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507',
            'logical_model_id':'Qwen3-30B-A3B-Instruct-2507',
            'train_rows':len(rows),
            'valid_rows':len(valid),
            'max_prompt_tokens': args.max_prompt_tokens,
            'n_rows_dropped_long_prompt': n_rows_dropped_long_prompt,
            'loss_path':loss_path,
            'rl_plus_opd_extra_tool_kl': args.method=='RL_PLUS_OPD',
            'span_audit':span,
            'pre_divergence':pre,
            'post_divergence':post,
            'mean_train_loss':sum(x['loss'] for x in losses)/max(1,len(losses)),
            'n_train_steps':len(losses),
            'adapter_path':str(adapter),
            'adapter_reload_acceptance':adapter_reload_acceptance,
            'student_inference_privilege':False,
            'synthetic_fallback':False,
            'train_seconds':time.time()-t0,
        }
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
        (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: completed\n', encoding='utf-8')
        (args.out/'DONE').write_text('ok\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False))
        gc.collect()
        return 0
    except Exception as exc:
        (args.out/'FAILED.json').write_text(json.dumps({'status':'failed','error':str(exc),'component':args.component,'method':args.method,'seed':args.seed}, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        (args.out/'STATUS_LIVE.md').write_text('# STATUS_LIVE\n\n- status: failed\n- error: '+str(exc)+'\n', encoding='utf-8')
        raise


if __name__=='__main__':
    raise SystemExit(main())
