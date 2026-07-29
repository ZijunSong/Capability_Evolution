#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE=bilateral bash "${ROOT}/scripts/scope_round3/run_all_8gpu.sh"
