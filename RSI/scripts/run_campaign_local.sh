#!/usr/bin/env bash
# Single-host campaign launcher.  Multi-host torchrun launcher uses the same
# package interfaces (see docs/runbook.md); no legacy scripts are invoked.
set -euo pipefail

CONFIG="${1:-configs/toy_smoke.yaml}"
RUNS_ROOT="${RSI_RUNS_ROOT:-runs}"

cd "$(dirname "$0")/.."

python -m rsi.cli doctor --config "$CONFIG"
python -m rsi.cli data prepare --config "$CONFIG"
python -m rsi.cli data validate --config "$CONFIG"
python -m rsi.cli eval --config "$CONFIG" --harness minimal --split harness_select
python -m rsi.cli campaign run --config "$CONFIG"
python -m rsi.cli campaign report --run-dir "$RUNS_ROOT/$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['campaign_id'])")"
