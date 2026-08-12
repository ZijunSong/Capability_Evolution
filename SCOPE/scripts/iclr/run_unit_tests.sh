#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/iclr_readiness"
mkdir -p "$OUT"
set +e
pytest tests/scope/test_experiment_spec.py \
  tests/scope/test_registry.py \
  tests/scope/test_config_diff.py \
  tests/scope/test_run_manifest.py \
  tests/scope/test_metrics_sanity.py \
  tests/scope/test_paired_stats.py \
  tests/scope/test_supervision_source_identity.py \
  tests/scope/test_verification_ablation.py \
  tests/scope/test_contract_threshold_matrix.py \
  tests/scope/test_decision_state_field_ablation.py \
  tests/scope/test_fallback_router.py \
  tests/scope/test_rollback_hierarchy.py \
  tests/scope/test_baseline_adapter.py \
  tests/scope/test_external_baseline_lock.py \
  tests/scope/test_binary_operation_metrics.py \
  -q --tb=line 2>&1 | tee "$OUT/UNIT_TEST_ICLR.log"
ICLR_EC=${PIPESTATUS[0]}
pytest tests/scope/ tests/scope_round9/ -q --tb=line 2>&1 | tee "$OUT/UNIT_TEST_FULL_SCOPE.log"
FULL_EC=${PIPESTATUS[0]}
set -e
echo "iclr_unit_exit=$ICLR_EC full_scope_exit=$FULL_EC" | tee "$OUT/UNIT_TEST_EXITCODES.txt"
exit $ICLR_EC
