#!/usr/bin/env bash
# Source-only SCAPE/EasyOPD environment contract for the 0819 component sweep.
#
# This script intentionally does not create or update a conda/venv environment.
# Formal runs must use an existing /opt runtime.  Missing dependencies fail closed
# with STOP_ENV_SETUP_FAILED.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script instead of executing it:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

set -euo pipefail

export CAP_ROOT="${CAP_ROOT:-/mnt/songzijun/Capability_Evolution}"
export SCAPE_ROOT="${SCAPE_ROOT:-${CAP_ROOT}/SCAPE}"
export EASYOPD_ROOT="${EASYOPD_ROOT:-${CAP_ROOT}/SCAPE-EasyOPD}"
export CANONICAL_STUDENT_BASE="${CANONICAL_STUDENT_BASE:-/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507}"
export SCAPE_STUDENT_LOGICAL_MODEL="${SCAPE_STUDENT_LOGICAL_MODEL:-Qwen3-30B-A3B-Instruct-2507}"
if [[ -z "${SCAPE_PYTHON_ENV:-}" ]]; then
  if [[ -x /opt/scape-projected-action/bin/python3 || -x /opt/scape-projected-action/bin/python ]]; then
    export SCAPE_PYTHON_ENV=/opt/scape-projected-action
  elif [[ -x /opt/scape-h1004/bin/python3 || -x /opt/scape-h1004/bin/python ]]; then
    export SCAPE_PYTHON_ENV=/opt/scape-h1004
  else
    export SCAPE_PYTHON_ENV=/opt/scape-easyopd-smoke7
  fi
fi

if [[ ! -x "${SCAPE_PYTHON_ENV}/bin/python3" && ! -x "${SCAPE_PYTHON_ENV}/bin/python" ]]; then
  echo "STOP_ENV_SETUP_FAILED: missing /opt Python runtime at ${SCAPE_PYTHON_ENV}" >&2
  return 1
fi
if [[ -x "${SCAPE_PYTHON_ENV}/bin/python3" ]]; then
  export PYTHON_BIN="${SCAPE_PYTHON_ENV}/bin/python3"
else
  export PYTHON_BIN="${SCAPE_PYTHON_ENV}/bin/python"
fi

# Do not require /opt/scape-easyopd-smoke7.  Keep the script name for backwards
# compatibility with older launchers, but the contract is the current /opt runtime.
export PATH="${SCAPE_PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${EASYOPD_ROOT}:${SCAPE_ROOT}:${SCAPE_ROOT}/external/harness-1:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export SCAPE_FORCE_LOCAL_HARMONY="${SCAPE_FORCE_LOCAL_HARMONY:-0}"
export HF_HOME="${HF_HOME:-/opt/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/opt/vllm-cache}"

_preflight_root="${EASYOPD_ROOT}/outputs/component_sweep_0818/preflight"
mkdir -p "${_preflight_root}"
_env_txt="${_preflight_root}/ENVIRONMENT.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CAP_ROOT=${CAP_ROOT}"
  echo "SCAPE_ROOT=${SCAPE_ROOT}"
  echo "EASYOPD_ROOT=${EASYOPD_ROOT}"
  echo "SCAPE_PYTHON_ENV=${SCAPE_PYTHON_ENV}"
  echo "CANONICAL_STUDENT_BASE=${CANONICAL_STUDENT_BASE}"
  echo "SCAPE_STUDENT_LOGICAL_MODEL=${SCAPE_STUDENT_LOGICAL_MODEL}"
  command -v python || true
  "${PYTHON_BIN}" -V
  "${PYTHON_BIN}" - <<'PY'
import importlib, sys
print('python_executable=' + sys.executable)
failed = []
for mod_name in ['torch','transformers','peft','vllm','ray','verl','easyopd','harness','scape']:
    try:
        mod = importlib.import_module(mod_name)
        version = getattr(mod, '__version__', 'OK')
        print(f'{mod_name}={version}')
        if mod_name == 'torch':
            print(f'torch_cuda={getattr(mod.version, "cuda", None)}')
    except Exception as exc:
        print(f'{mod_name}=FAILED {type(exc).__name__}: {str(exc)[:240]}')
        failed.append(mod_name)
if failed:
    raise SystemExit('STOP_ENV_SETUP_FAILED missing_or_broken=' + ','.join(failed))
PY
} > "${_env_txt}" 2>&1 || {
  cat "${_env_txt}" >&2
  return 1
}

echo "SCAPE EasyOPD environment ready: ${PYTHON_BIN}"
echo "Preflight written: ${_env_txt}"
