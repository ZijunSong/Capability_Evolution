#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/Capability_Evolution/SCOPE
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH=$PWD

# Build targeted test split if needed
python - <<'PY'
import json
from pathlib import Path
from training.scope_round13.build_natural_stage2 import build_split
repo=Path('.')
out=repo/'artifacts/datasets/scope_round13/checkpoint_targeted/test.jsonl'
sdi=repo/'artifacts/datasets/scope_round13/operation_sdi/test.jsonl'
if sdi.exists() and (not out.exists() or out.stat().st_size==0):
    rows=[json.loads(l) for l in sdi.open() if l.strip()]
    test=build_split(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('wrote test', len(test))
else:
    print('test exists or sdi missing')
PY

eval_one() {
  local gpu="$1" variant="$2"
  local dir="outputs/scope_round13/stage2_targeted/training/${variant}"
  [[ -f "${dir}/DONE" ]] || return 0
  [[ -f "${dir}/eval_valid/METRICS.json" ]] && return 0
  echo "[$(date -Is)] eval ${variant} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round13/eval_stage2_pointer.py \
    --variant-dir "${dir}" --split valid --gpu cuda:0 \
    >> "outputs/scope_round13/logs/stage2_${variant}_eval.log" 2>&1
}

# Poll until all stage2 done, eval on free GPUs
while true; do
  n=0
  for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
    [[ -f outputs/scope_round13/stage2_targeted/training/$v/DONE ]] && n=$((n+1))
  done
  # Launch evals for done variants on free GPUs
  for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
    d=outputs/scope_round13/stage2_targeted/training/$v
    [[ -f $d/DONE ]] || continue
    [[ -f $d/eval_valid/METRICS.json ]] && continue
    if ps -eo args | grep -F "eval_stage2_pointer.py --variant-dir" | grep -F "$v" | grep -v grep >/dev/null; then
      continue
    fi
    # find free gpu
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
    [[ -n "${gpu}" ]] || break
    nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=$PWD /data/ppnm/miniconda3/envs/bishop/bin/python training/scope_round13/eval_stage2_pointer.py --variant-dir outputs/scope_round13/stage2_targeted/training/${v} --split valid --gpu cuda:0 >> outputs/scope_round13/logs/stage2_${v}_eval.log 2>&1" &
    echo $! > "outputs/scope_round13/pids/stage2_eval_${v}.pid"
    echo "[$(date -Is)] started eval ${v} gpu${gpu}"
    sleep 15
  done

  n_eval=0
  for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
    [[ -f outputs/scope_round13/stage2_targeted/training/$v/eval_valid/METRICS.json ]] && n_eval=$((n_eval+1))
  done
  if [[ "$n" -ge 3 ]] && [[ "$n_eval" -ge 3 ]]; then
    echo "[$(date -Is)] all stage2 trained+evaled"
    python training/scope_round13/stage2_gates.py >> outputs/scope_round13/logs/stage2_gates.log 2>&1 || true
    python training/scope_round13/write_round13_report.py >> outputs/scope_round13/logs/write_report.log 2>&1 || true
    break
  fi
  sleep 120
done
echo "[$(date -Is)] stage2 continuum finished"
