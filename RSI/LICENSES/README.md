# Licenses & migrated code provenance

- `src/rsi/opd/losses.py` contains pure functions migrated from the legacy
  research tree (`TRIM/trim/training/sr_opd_loss.py` and its
  `canonical_metrics.kl_from_logits` dependency), rewritten with local
  dependencies only.  Behavior kept numerically identical
  (see `tests/unit/test_loss_and_grad_parity.py`).  Source project license
  applies to those migrated sections; see the parent repository's LICENSE.
- `src/rsi/training/hf_reference.py` CISPO reduction follows the verified
  semantics of the legacy `hf_rl_opd_client` implementation (documented in
  `docs/method_migration.md`); the code is a new implementation.
- Everything else in `src/rsi/` is a new implementation for this project
  (Apache-2.0 as declared in `pyproject.toml`).
- External references (Harness-Zero, ADAS, DGM, AFlow, ScholarEvolve,
  AppWorld, SpreadsheetBench, SWE-bench) are method references only; no code
  from those repositories is bundled.  Their benchmark packages are optional
  dependencies with their own licenses.
