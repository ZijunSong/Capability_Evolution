#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m experiments.ablations.aggregate --root outputs/iclr_ablations --output outputs/iclr_readiness/ablation_aggregate.json || true
python -m experiments.baselines.aggregate --root outputs/iclr_baselines | tee outputs/iclr_readiness/baseline_aggregate.json
echo "aggregate done"
