#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/iclr_readiness/preflight"
mkdir -p "$OUT"

{
  echo "branch: $(git branch --show-current)"
  echo "HEAD: $(git rev-parse HEAD)"
  git status --short
  git remote -v
  git log --oneline --decorate -10
} | tee "$OUT/git_state.txt"

{
  echo "python: $(python --version 2>&1)"
  python - <<'PY'
import sys
print("executable", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("gpu_count", torch.cuda.device_count())
except Exception as e:
    print("torch_error", e)
try:
    import transformers; print("transformers", transformers.__version__)
except Exception as e:
    print("transformers_error", e)
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm_error", e)
PY
  nvidia-smi 2>&1 | head -40 || true
  df -h . | tail -1
} | tee "$OUT/environment.txt"

# Path checks
for p in \
  harness/configs/modules_minimal_v2.yaml \
  harness/configs/modules_full_v2.yaml \
  artifacts/datasets/round2_audit_100q/query_manifest.json \
  experiments/registry.yaml
do
  if [[ ! -e "$p" ]]; then
    echo "MISSING: $p" | tee -a "$OUT/environment.txt"
    exit 1
  fi
done

python - <<'PY'
from experiments.common.registry import ExperimentRegistry
reg = ExperimentRegistry()
errs = reg.validate()
print("registry_experiments", len(reg.ids()))
print("registry_errors", len(errs))
for e in errs[:20]:
    print("ERR", e)
if errs:
    raise SystemExit(1)
print("preflight_ok")
PY

echo "preflight complete -> $OUT"
