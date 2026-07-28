#!/usr/bin/env bash
# Backward-compatible wrapper — delegates to vLLM rollout + HF train smoke test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/smoke_opd_vllm_hf_4gpu.sh" "$@"
