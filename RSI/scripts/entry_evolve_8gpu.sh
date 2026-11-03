#!/usr/bin/env bash
# =============================================================================
# RSI 第一阶段（EVOLVE 演化）运行入口 —— 单节点八卡
#
# 流水线：
#   preflight(GPU/版本/安装) -> doctor -> data prepare/validate
#   -> evolve（候选生成 + 八卡并行 panel 评估）-> champion 产物审计
#
# 用法：
#   bash scripts/entry_evolve_8gpu.sh [--config CONFIG] [--workers N] [--bank REF]
#
#   --config   campaign 配置（默认 configs/pilot_multidomain.yaml；
#              本地验证可用 configs/toy_smoke.yaml）
#   --workers  panel 并行 worker 数 = 占用 GPU 数（默认 8，上限可见 GPU 数）
#   --bank     初始 bank 引用（默认空库 empty_each_round；warm_archive 消融用）
#
# 环境变量：
#   RSI_PANEL_WORKERS   覆盖 --workers
#   RSI_PROPOSER_DEVICE proposer 所在设备（默认 cpu，避免与 8 个 panel worker
#                       抢显存；如需为 proposer 留卡：用 --workers 7 并自行
#                       在配置中指定模型设备）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RSI_PROJECT_ROOT override: CloudML wrappers copy this entry to /tmp (JuiceFS
# heredoc lesson) and pin the project root explicitly.
PROJECT_ROOT="${RSI_PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$PROJECT_ROOT"

CONFIG="configs/pilot_multidomain.yaml"
WORKERS=""
BANK_REF=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  CONFIG="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --bank)    BANK_REF="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "[FATAL] unknown arg: $1" >&2; exit 2 ;;
  esac
done

log()  { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[$(date '+%F %T')] [FATAL] $*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
log "== RSI phase-1 EVOLVE entry (single-node 8-GPU) =="
log "PROJECT_ROOT=$PROJECT_ROOT"
log "CONFIG=$CONFIG"
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

python -c "import rsi" 2>/dev/null \
  || fail "package 'rsi' not importable. Install first: python -m pip install -e '.[dev]'"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found; this entry requires a GPU node"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU' || true)"
[[ "$GPU_COUNT" -ge 1 ]] || fail "no GPU detected"
log "GPU inventory ($GPU_COUNT):"
nvidia-smi -L | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

if [[ -n "${RSI_PANEL_WORKERS:-}" ]]; then
  WORKERS="$RSI_PANEL_WORKERS"
fi
if [[ -z "$WORKERS" ]]; then
  if [[ "$GPU_COUNT" -ge 8 ]]; then WORKERS=8; else WORKERS="$GPU_COUNT"; fi
fi
[[ "$WORKERS" -le "$GPU_COUNT" ]] || fail "workers=$WORKERS exceeds visible GPUs=$GPU_COUNT"
log "panel workers: $WORKERS (one model replica per GPU; proposer device=${RSI_PROPOSER_DEVICE:-cpu})"

python - <<'PY' || fail "torch/CUDA preflight failed — see [FATAL] above (driver vs torch build mismatch)"
import sys, torch, transformers
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"  torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={n}")
print(f"  transformers={transformers.__version__}")
if not torch.cuda.is_available() or n < 1:
    sys.exit(f"[FATAL] CUDA unusable (torch {torch.__version__}, devices={n}); "
             "this entry requires a driver-matched torch (e.g. torch==2.10.0 on CUDA 12.8 drivers)")
PY

# ---------------------------------------------------------------- doctor
log "== doctor =="
python -m rsi.cli doctor --config "$CONFIG" > /tmp/rsi_doctor_evolve.json \
  || fail "doctor failed (see /tmp/rsi_doctor_evolve.json)"
python - <<'PY' || fail "doctor reported missing capabilities (see above)"
import json
rep = json.load(open("/tmp/rsi_doctor_evolve.json"))
print(json.dumps({k: rep["checks"].get(k) for k in
                  ("python", "core_modules", "environments", "training_backend", "torch")},
                 indent=1, ensure_ascii=False))
assert rep["ok"], "doctor ok=false"
PY

# ---------------------------------------------------------------- data
log "== data prepare / validate =="
python -m rsi.cli data prepare  --config "$CONFIG" | python -c "import json,sys; d=json.load(sys.stdin); print('  prepare:', d.get('ok'), d.get('split_manifest'))"
python -m rsi.cli data validate --config "$CONFIG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

# ---------------------------------------------------------------- evolve
CAMPAIGN_ID="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['campaign_id'])")"
LOG_DIR="runs/$CAMPAIGN_ID/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evolve_8gpu_$(date +%Y%m%d_%H%M%S).log"

export RSI_PANEL_WORKERS="$WORKERS"
BANK_ARGS=()
if [[ -n "$BANK_REF" ]]; then
  [[ -e "$BANK_REF" ]] || fail "--bank ref not found: $BANK_REF"
  BANK_ARGS=(--bank "$BANK_REF")
  log "initial bank: $BANK_REF"
else
  log "initial bank: empty (empty_each_round)"
fi

log "== evolve ($WORKERS GPU panel workers) -> $LOG_FILE =="
set +e
python -m rsi.cli evolve --config "$CONFIG" "${BANK_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
EVOLVE_RC="${PIPESTATUS[0]}"
set -e

# ---------------------------------------------------------------- audit
RESULT="runs/$CAMPAIGN_ID/evolve/evolve_result.json"
CHAMPION="runs/$CAMPAIGN_ID/evolve/champion.json"
log "== evolve artifacts =="
if [[ -f "$RESULT" ]]; then
  python - "$RESULT" "$CHAMPION" <<'PY'
import json, sys, os
res = json.load(open(sys.argv[1]))
print(f"  stopped_reason     : {res['stopped_reason']}")
print(f"  generations        : {len(res['generations'])}")
print(f"  evaluated_candidates: {res['evaluated_candidates']}")
for g in res["generations"]:
    print(f"    gen{g['generation']}: candidates={g['candidate_ids']} "
          f"champion={str(g['champion_id'])[:12]} sel={g['selection']['reason'][:60]}")
champ = sys.argv[2]
if os.path.isfile(champ):
    c = json.load(open(champ))
    print(f"  champion           : {c['candidate_id'][:16]}")
    print(f"  champion bank_dir  : {c['bank_dir']}")
    print(f"  champion hypothesis: {c['hypothesis']}")
    print("  next: python -m rsi.cli probe-opd --config <CONFIG> --bank " + champ)
    print("        python -m rsi.cli train     --config <CONFIG> --bank " + champ)
else:
    print("  champion           : NONE (no admissible harness this phase)")
PY
else
  fail "evolve result missing: $RESULT (rc=$EVOLVE_RC)"
fi

if [[ "$EVOLVE_RC" -ne 0 ]]; then
  log "evolve exited rc=$EVOLVE_RC (no admissible candidate is an honest outcome; see $LOG_FILE)"
  exit "$EVOLVE_RC"
fi
log "== phase-1 EVOLVE complete =="
