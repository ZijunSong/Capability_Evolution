#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EID="${1:?usage: resume.sh <experiment_id>}"
python -m experiments.common.launcher --experiment-id "$EID" --resume --smoke-query-limit "${2:-4}"
