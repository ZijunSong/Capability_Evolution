# LEGACY_OPD_CODE_AUDIT

## Summary

The legacy SCAPE OPD code base already contains useful state/dual-view/tool-mask helpers, but the old trainer scripts are not a unified framework.

## High-risk findings

1. Tool-span / config loading depends on external packages that may not be present in the default environment.
2. Some legacy runners hard-code device placement or couple collection with training logic.
3. The new migration should avoid patching the old route-head style trainer and instead route future work through the new method-local framework.

## Questions from the todo

1. Rollout from current student: existing probes use reduced-state rollout helpers, but some old scripts mix collection and training responsibilities.
2. Teacher rescoring on student prefix: the new framework must do this explicitly; legacy scripts are not a reliable contract.
3. Token alignment: tool-span helpers exist, but the new contract should own exact token ids.
4. Forward/reverse KL: legacy code mixes exact and proxy metrics; the new loss module now defines exact KL paths.
5. Tool-call mask: legacy support exists, but the new framework must enforce it in tests.
6. JSON argument tokens: must be kept in the new projected-action/CE path.
7. LoRA wrapping order: legacy code suggests potential double-wrap risk; new framework should keep adapter handling isolated.
8. Query split / state fork / leakage: legacy helpers are partial; the new framework now codifies them as explicit tests and gates.
