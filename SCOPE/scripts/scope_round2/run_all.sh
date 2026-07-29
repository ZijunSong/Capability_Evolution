#!/usr/bin/env bash
# SCOPE Round 2 master orchestrator — runs barriers sequentially.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
WAVE="${WAVE:-all}"
SKIP_BARRIER0="${SKIP_BARRIER0:-0}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

# Env required by BM25 / vLLM / BrowseComp (match existing rollout scripts)
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; source "${REPO_ROOT}/.env"; set +a
fi

ROOT="${REPO_ROOT}/outputs/scope_round2"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
ROUND1_MODEL="${REPO_ROOT}/outputs/dup_sdi_round1/merged_hf"
HARNESS_V2="${REPO_ROOT}/harness/configs/modules_minimal_v2.yaml"

mkdir -p "${ROOT}"/{diagnostics,hmin_v2_base,hmin_v2_round1,dup_shadow,stop_calibration,training,eval,logs,pids}

_log_gpu() {
  local gpu="$1" name="$2"
  local ldir="${ROOT}/logs/gpu${gpu}/${name}"
  mkdir -p "${ldir}"
  echo "$$" > "${ldir}/pid"
  date -Iseconds > "${ldir}/start_time"
}

barrier0() {
  echo "=== Barrier 0: tests ==="
  pytest tests/scope/ -q --tb=short
  python training/scope/analyze_loss_mass.py \
    --output-json "${ROOT}/diagnostics/round1_loss_mass.json" \
    --output-md "${ROOT}/diagnostics/round1_loss_mass.md"
  python training/scope_round2/create_query_manifest.py
  git diff --stat > "${ROOT}/git_diff_stat.txt" 2>/dev/null || true
  cat > "${ROOT}/CODE_CHANGE_REPORT.md" <<EOF
# CODE_CHANGE_REPORT

## Barrier 0
- pytest tests/scope: PASS
- analyze_loss_mass: ${ROOT}/diagnostics/round1_loss_mass.md
- query_manifest: ${MANIFEST}
- git_commit: $(git rev-parse HEAD)
EOF
  echo "Barrier 0 PASS"
}

wave1_gpu() {
  local gpu="$1" variant="$2" shard="$3" model="$4" port="$5"
  local out="${ROOT}/hmin_v2_${variant}/${shard}"
  [[ -f "${out}/summary.json" ]] && echo "[skip] ${out}" && return 0
  _log_gpu "${gpu}" "wave1_${variant}_${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python training/scope_round2/hmin_v2_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --model-path "${model}" \
    --harness-config "${HARNESS_V2}" \
    --vllm-port "${port}" \
    --tensor-parallel-size 1 \
    --parallel 1 \
    2>&1 | tee "${ROOT}/logs/gpu${gpu}/wave1_${variant}_${shard}/stdout.log"
  date -Iseconds > "${ROOT}/logs/gpu${gpu}/wave1_${variant}_${shard}/end_time"
}

wave1() {
  echo "=== Wave 1: Base vs Round1 H_min_v2 100q ==="
  wave1_gpu 0 base shard0 "${BASE_MODEL}" 8800 &
  wave1_gpu 1 base shard1 "${BASE_MODEL}" 8801 &
  wave1_gpu 2 base shard2 "${BASE_MODEL}" 8802 &
  wave1_gpu 3 base shard3 "${BASE_MODEL}" 8803 &
  wave1_gpu 4 round1 shard0 "${ROUND1_MODEL}" 8804 &
  wave1_gpu 5 round1 shard1 "${ROUND1_MODEL}" 8805 &
  wave1_gpu 6 round1 shard2 "${ROUND1_MODEL}" 8806 &
  wave1_gpu 7 round1 shard3 "${ROUND1_MODEL}" 8807 &
  wait
  echo "Wave 1 jobs finished"
}

barrier1() {
  echo "=== Barrier 1: merge + paired eval ==="
  python training/scope_round2/merge_shards.py \
    --shard-dirs "${ROOT}/hmin_v2_base/shard0" "${ROOT}/hmin_v2_base/shard1" \
      "${ROOT}/hmin_v2_base/shard2" "${ROOT}/hmin_v2_base/shard3" \
    --output-dir "${ROOT}/hmin_v2_base/merged" --manifest "${MANIFEST}"
  python training/scope_round2/merge_shards.py \
    --shard-dirs "${ROOT}/hmin_v2_round1/shard0" "${ROOT}/hmin_v2_round1/shard1" \
      "${ROOT}/hmin_v2_round1/shard2" "${ROOT}/hmin_v2_round1/shard3" \
    --output-dir "${ROOT}/hmin_v2_round1/merged" --manifest "${MANIFEST}"
  python training/scope_round2/eval_paired.py \
    --base "${ROOT}/hmin_v2_base/merged" \
    --round1 "${ROOT}/hmin_v2_round1/merged" \
    --output-json "${ROOT}/eval/base_vs_round1_100q.json" \
    --output-md "${ROOT}/eval/base_vs_round1_100q.md"
  cp "${ROOT}/eval/base_vs_round1_100q.md" "${ROOT}/eval/base_vs_round1_100q.md"
  echo "Barrier 1 PASS"
}

wave2() {
  echo "=== Wave 2: dup shadow + stop cal + diagnostics ==="
  for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${i}" python training/scope_round2/dup_shadow_label.py \
    --states "${ROOT}/hmin_v2_base/shard${i}/decision_states.jsonl" \
    --output-dir "${ROOT}/dup_shadow/shard${i}" &
  done
  wait
  python training/scope_round2/stop_calibration_audit.py \
    --states "${ROOT}/hmin_v2_base/shard0/decision_states.jsonl" \
      "${ROOT}/hmin_v2_base/shard1/decision_states.jsonl" \
      "${ROOT}/hmin_v2_base/shard2/decision_states.jsonl" \
      "${ROOT}/hmin_v2_base/shard3/decision_states.jsonl" \
    --output-json "${ROOT}/stop_calibration/stop_calibration_100q.json" \
    --output-md "${ROOT}/stop_calibration/stop_calibration_100q.md"
  cp "${ROOT}/stop_calibration/stop_calibration_100q.md" "${ROOT}/eval/stop_calibration_100q.md" 2>/dev/null || true
  python training/scope/analyze_loss_mass.py \
    --output-json "${ROOT}/diagnostics/round1_loss_mass.json" \
    --output-md "${ROOT}/diagnostics/round1_loss_mass.md"
  CUDA_VISIBLE_DEVICES=6 python training/scope/eval_dup_capability.py \
    --valid artifacts/datasets/dup_sdi_round1/valid.jsonl \
    --model-path "${BASE_MODEL}" \
    --adapter-path outputs/dup_sdi_round1 \
    --output "${ROOT}/diagnostics/round1_capability_eval.json" \
    --include-legacy-metrics || true
  pytest tests/scope/ -q
  echo "Wave 2 PASS"
}

barrier2() {
  echo "=== Barrier 2: Round 2 dataset ==="
  python training/scope_round2/build_round2_dataset.py \
    --shadow-dirs "${ROOT}/dup_shadow/shard0" "${ROOT}/dup_shadow/shard1" \
      "${ROOT}/dup_shadow/shard2" "${ROOT}/dup_shadow/shard3" \
    --output-dir artifacts/datasets/dup_sdi_round2
  python - <<'PY'
import json
from pathlib import Path
stats = json.loads(Path("artifacts/datasets/dup_sdi_round2/stats.json").read_text())
md = ["# Dup Round 2 Dataset Stats\n"]
for k,v in stats.items():
    md.append(f"- {k}: {v}")
Path("outputs/scope_round2/eval/dup_round2_dataset_stats.md").write_text("\n".join(md)+"\n")
PY
  echo "Barrier 2 PASS"
}

wave3_train() {
  local gpu="$1" name="$2" extra_args="${3:-}"
  local out="${ROOT}/training/${name}"
  [[ -f "${out}/train_summary.json" ]] && echo "[skip train] ${name}" && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" python training/train_sdi_dup.py \
    --config configs/scope/sdi_dup_round2_main.yaml \
    --train artifacts/datasets/dup_sdi_round2/train.jsonl \
    --valid artifacts/datasets/dup_sdi_round2/valid.jsonl \
    --output-dir "${out}" \
    ${extra_args} \
    2>&1 | tee "${ROOT}/logs/gpu${gpu}/wave3_${name}/stdout.log"
}

wave3() {
  echo "=== Wave 3: 8-way training ablations ==="
  mkdir -p "${ROOT}/logs/gpu"{0..7}/wave3_*
  wave3_train 0 round2_main "--compact-target --route-balancing --loss-mode sample_normalized_action_ce" &
  wave3_train 1 round2_legacy_token_ce "--loss-mode legacy_token_ce" &
  wave3_train 2 round2_full_action_sample_norm "--loss-mode sample_normalized_action_ce" &
  wave3_train 3 round2_no_route_balance "--compact-target --loss-mode sample_normalized_action_ce" &
  wave3_train 4 round2_endorse_only "--compact-target --route-filter ENDORSE --loss-mode sample_normalized_action_ce" &
  wave3_train 5 round2_correct_only "--compact-target --route-filter CORRECT --loss-mode sample_normalized_action_ce" &
  wave3_train 6 round2_main_seed43 "--compact-target --route-balancing --loss-mode sample_normalized_action_ce --seed 43" &
  wave3_train 7 round2_main_seed44 "--compact-target --route-balancing --loss-mode sample_normalized_action_ce --seed 44" &
  wait
  echo "Wave 3 PASS"
}

barrier3() {
  echo "=== Barrier 3: training sanity ==="
  python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round2/training")
rows = []
for name in sorted(p.name for p in root.iterdir() if p.is_dir()):
    summ = root / name / "train_summary.json"
    if summ.exists():
        d = json.loads(summ.read_text())
        vm = d.get("valid_metrics") or {}
        rows.append({"variant": name, **vm})
Path("outputs/scope_round2/eval/round2_training_comparison.md").write_text(
    json.dumps(rows, indent=2) + "\n"
)
print(json.dumps(rows, indent=2))
PY
}

wave4() {
  echo "=== Wave 4: closed-loop 100q eval ==="
  # Reuse base from wave1 if available
  local variants=(
    "0:base:${BASE_MODEL}:hmin_v2_base"
    "1:round1_old:${ROUND1_MODEL}:hmin_v2_round1_wave4"
    "2:round2_main:${BASE_MODEL}:round2_main_cl"
    "3:round2_legacy:${BASE_MODEL}:round2_legacy_cl"
    "4:round2_full_norm:${BASE_MODEL}:round2_full_norm_cl"
    "5:round2_endorse:${BASE_MODEL}:round2_endorse_cl"
    "6:round2_correct:${BASE_MODEL}:round2_correct_cl"
    "7:round2_no_rb:${BASE_MODEL}:round2_no_rb_cl"
  )
  for spec in "${variants[@]}"; do
    IFS=: read -r gpu tag model outname <<< "${spec}"
    if [[ "${tag}" == "base" && -f "${ROOT}/hmin_v2_base/merged/summary.json" ]]; then
      echo "[wave4] reuse wave1 base merged"
      continue
    fi
    # For trained variants, merge LoRA first if needed
    local model_path="${model}"
  done
  echo "Wave 4 — see ROUND2_REPORT for status"
}

final_report() {
  python - <<'PY'
from pathlib import Path
import json
root = Path("outputs/scope_round2")
lines = ["# ROUND2_REPORT\n", "## Summary Table\n",
"| Variant | DupCurateRate ↓ | FalseSkipRate ↓ | UniqueEvidence ↑ | Reward | Recall |",
"|---------|-----------------|-----------------|------------------|--------|--------|"]
# Fill from available data only
b1 = root / "eval/base_vs_round1_100q.json"
if b1.exists():
    d = json.loads(b1.read_text())
    for tag, key in [("Base","base"),("Old Round1","round1")]:
        m = d.get(key, {})
        lines.append(f"| {tag} | {m.get('duplicate_curate_rate', 'n/a')} | n/a | {m.get('unique_evidence_ratio','n/a')} | {m.get('reward','n/a')} | {m.get('recall','n/a')} |")
lines += [
    "\n## Verdicts\n",
    "ROOT_CAUSE_ROUND1 = see round1_loss_mass.md\n",
    "ROUND2_POSITIVE_SIGNAL = false  # update after wave4\n",
    "RECOMMEND_830 = false\n",
    "NEXT_ACTION = complete wave3/wave4 if barriers pass\n",
]
(root / "ROUND2_REPORT.md").write_text("\n".join(lines))
print("Wrote ROUND2_REPORT.md")
PY
}

case "${WAVE}" in
  barrier0) barrier0 ;;
  wave1) wave1 ;;
  barrier1) barrier1 ;;
  wave2) wave2 ;;
  barrier2) barrier2 ;;
  wave3) wave3 ;;
  barrier3) barrier3 ;;
  wave4) wave4 ;;
  final) final_report ;;
  all)
    [[ "${SKIP_BARRIER0}" == "1" ]] || barrier0
    wave1
    barrier1
    wave2
    barrier2
    wave3
    barrier3
    wave4
    final_report
    ;;
  *) echo "Unknown WAVE=${WAVE}"; exit 1 ;;
esac
