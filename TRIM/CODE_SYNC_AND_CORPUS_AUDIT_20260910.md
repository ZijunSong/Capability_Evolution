# Code Sync and Corpus Audit — 2026-09-10

## Classification

This is a diagnostic synchronization and data-installation record. It is not a model evaluation result or formal artifact.

## Code baseline

- Repository: `/mnt/songzijun/Capability_Evolution`
- Branch: `scope/round14-capability-portfolio`
- GitHub remote branch tip verified with `git ls-remote`: `41641ea7c886edef4705b10316d31f2a98153966`
- Final local `HEAD`: `41641ea7c886edef4705b10316d31f2a98153966`
- Remote-tracking ref after fetch: same commit
- Previous local divergent history preserved at `backup/pre-github-sync-20260910-0d077d8`
- Synchronization method: fetched named branch and moved the clean main worktree exactly to the verified tip; no merge commit was created
- Four linked `SCAPE-wt-h100-*` worktrees were not changed
- `/mnt/songzijun/EasyOPD-baseline` was not changed

The local main worktree had no tracked modifications. Existing untracked TRIM experiment/audit/configuration files were preserved; no `git clean` was run.

## Corpus source and installation

- Source archive: `/mnt/songzijun/browsecomp_plus_corpus_full_20260910.tar.gz`
- Archive SHA-256: `24f7962292a5753e0a73f24f55a3c1cf71248654725e186dd9d0f5ce72a0355e`
- Archive members: `browsecomp_plus_corpus_full.jsonl` and `browsecomp_plus_corpus_full.BUILD_REPORT.json` only
- Archive gzip integrity: passed
- Installed corpus: `SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_corpus_full.jsonl`
- Installed corpus size: `3298498064` bytes
- Installed corpus SHA-256: `e641e4cbd087ae7911461c100c8ff73fbbe3a5ca5869b32b40fb1f274ae4eebc`
- Installed BUILD_REPORT was preserved unchanged from the archive
- Extraction used a staging directory and final file moves; no pre-existing target was overwritten
- Corpus directory is ignored by repository rules and is not code provenance

## Corpus validation

Passed:

- 100195 JSONL rows
- 100195 unique document IDs
- zero empty text records
- required `id`, `source`, and `text` fields present for every row
- probe document IDs `59931`, `69324`, and `44797` present
- archive BUILD_REPORT agrees with 100195 indexed documents, zero missing text, and zero index/corpus delta
- existing BM25 index is present at `SCOPE/external/BrowseComp-Plus/indexes/bm25` (approximately 2.1G)

Not run:

- Runtime Lucene/Pyserini corpus-vs-index validation could not run because the active Python environment has no `pyserini` module (`ModuleNotFoundError`). This remains a diagnostic gap; no formal evaluation claim is made.

## Code checks

Passed:

- `TRIM/tests/test_bcplus_full_gpu4_eval_script.py`: 3 passed
- `TRIM/tests/test_corpus_validate_and_health.py`: 5 passed
- synchronized TRIM/SCAPE Python modules compiled successfully with `compileall`
- `run_bcplus_full_gpu4_eval.sh` passed `bash -n`
- final `HEAD` equals GitHub branch tip and tracked working-tree/index diffs are empty

No GPU evaluation was launched by this synchronization task.
