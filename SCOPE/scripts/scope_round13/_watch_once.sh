#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/Capability_Evolution/SCOPE
echo "==== $(date -Is) ===="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo '--- S1 ---'
for v in r13_onpolicy_querynorm_seed42 r13_onpolicy_querynorm_seed43 r13_onpolicy_querynorm_seed44 r13_onpolicy_querynorm_nohard_seed42 r13_onpolicy_eventuniform_seed42; do
  d=outputs/scope_round13/phase_b_stage1/training/$v
  echo -n "$v: "
  [[ -f $d/DONE ]] && echo -n 'DONE '
  [[ -f $d/FAILED ]] && echo -n 'FAILED '
  [[ -f $d/eval_valid/METRICS.json ]] && echo -n 'METRICS '
  [[ -f $d/merged/config.json ]] && echo -n 'merged '
  if ps -eo args | grep -F "run_stage1_train.py --variant $v" | grep -v grep >/dev/null; then echo -n 'TRAIN '; fi
  if ps -eo args | grep -F "eval_stage1_split.py --variant-dir" | grep -F "$v" | grep -v grep >/dev/null; then echo -n 'EVAL '; fi
  echo
done
echo '--- S2 ---'
for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
  d=outputs/scope_round13/stage2_targeted/training/$v
  n=$(grep -c stage2-train outputs/scope_round13/logs/stage2_${v}.log 2>/dev/null || echo 0)
  echo -n "$v steps=$n "
  [[ -f $d/DONE ]] && echo -n DONE
  [[ -f $d/FAILED ]] && echo -n FAILED
  echo
done
echo '--- markers/gates ---'
ls outputs/scope_round13/markers/ 2>/dev/null
ls outputs/scope_round13/phase_b_stage1/STAGE1_*.json 2>/dev/null || echo no_stage1_gates
ls outputs/scope_round13/ROUND13_REPORT.md 2>/dev/null || echo no_report
# restart dead stage1 if needed
for gpu in 0 1 2 3; do
  case $gpu in
    0) v=r13_onpolicy_querynorm_seed42 ;;
    1) v=r13_onpolicy_querynorm_seed43 ;;
    2) v=r13_onpolicy_querynorm_seed44 ;;
    3) v=r13_onpolicy_querynorm_nohard_seed42 ;;
  esac
  d=outputs/scope_round13/phase_b_stage1/training/$v
  if [[ -f $d/DONE ]] && [[ -f $d/eval_valid/METRICS.json ]]; then continue; fi
  if [[ -f $d/FAILED ]] || { [[ ! -f $d/DONE ]] && ! ps -eo args | grep -F "run_stage1_train.py --variant $v" | grep -v grep >/dev/null && ! ps -eo args | grep -F "run_stage1_gpu.sh $gpu $v" | grep -v grep >/dev/null; }; then
    echo "RESTART S1 $v on GPU$gpu"
    rm -f "$d/FAILED" "$d/DONE"
    nohup bash scripts/scope_round13/run_stage1_gpu.sh "$gpu" "$v" >> "outputs/scope_round13/logs/stage1_${v}_supervisor.log" 2>&1 &
    echo $! > "outputs/scope_round13/pids/stage1_gpu${gpu}.pid"
  fi
done
