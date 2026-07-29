#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE=train bash "${ROOT}/scripts/scope_round3/run_all_8gpu.sh"
