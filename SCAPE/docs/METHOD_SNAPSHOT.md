# METHOD_SNAPSHOT

## Claim

SCAPE studies whether cognitive components of a **public search harness** (Harness-1) can be migrated into model parameters via **same-environment-state tool-call OPD**, then **retired or recomposed at runtime** with a measurable quality–cost tradeoff.

## Non-goals (stopped)

- SCOPE `duplicate_evidence` KEEP/SKIP taxonomy expansion
- rollback Stage1/Stage2 rescue
- `P_m` OFF/PROC/FULL distillability map as the main object
- Information-Safe Gate / O7 discriminative classifier
- old `ModuleRetirementGate` A/B/C logic

## Core objects

1. **Component mask** over Harness-1 `V8D_*` flags (env override; no upstream fork for toggles).
2. **Snapshot `xi_t`** collected under reduced harness student rollout `H_-m`.
3. **Dual view** `r_-m(xi_t)` vs `r_F(xi_t)` without teacher env stepping.
4. **Tool-token OPD** (name + args spans) + light anchor KL; EMA/lagged teacher locked in manifest.
5. **Gates**: Learnability (L) → Single-component four-grid (S) → Multi-component (M) → Pareto recomposition.

## Primary metrics

- Contribution Δ on curated/trajectory/final-answer recall + cost deltas
- Same-state `I_name` (JS) / `I_args` (teacher-forced token divergence)
- Learnability `L_m = 1 - D_post/(D_pre+eps)`
- Four-grid `CCR_m`, `N_m_post`, `HRR`
- Quality–cost Pareto frontier
