# Shared vLLM serve flags derived from trim.eval.model_profiles.
# Source from GPU eval launch scripts after TRIM_ROOT / PY are set.

vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" -m trim.eval.model_profiles vllm-extra "${slug}"
}

model_needs_smoke_test() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" -m trim.eval.model_profiles needs-smoke-test "${slug}"
}

model_tool_parser() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" -m trim.eval.model_profiles tool-parser "${slug}"
}
