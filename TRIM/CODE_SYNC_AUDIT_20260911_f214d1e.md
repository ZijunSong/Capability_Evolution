# Code Sync Audit — 2026-09-11 (f214d1e)

## Classification

This is a diagnostic synchronization record. It is not a model evaluation result or formal artifact.

## Code baseline

- Repository: `/mnt/songzijun/Capability_Evolution`
- Branch: `scope/round14-capability-portfolio`
- Requested target: commit `f214d1e` (tip of `origin/scope/round14-capability-portfolio`)
- GitHub remote branch tip verified with `git ls-remote`: `f214d1ef89191ec9a10bf89cf6973ddca35a5817`
- Final local `HEAD`: `f214d1ef89191ec9a10bf89cf6973ddca35a5817`
- Remote-tracking ref after fetch: `refs/remotes/origin/scope/round14-capability-portfolio` -> same commit
- Previous local `HEAD` before sync: `41641ea7c886edef4705b10316d31f2a98153966`
- Synchronization method: `41641ea` is a direct ancestor of `f214d1e`, so a fast-forward (`git merge --ff-only f214d1e`) moved the branch forward by 2 commits; no merge commit was created.
- Pre-sync state preserved at backup branch `backup/pre-sync-f214d1e-20260911` -> `41641ea`.
- Local working copy of the re-patched file preserved at `/tmp/vllm_rollout_worker.local.20260911.py` and `/tmp/vllm_rollout_worker.local.patch`.

## New upstream commits pulled (41641ea..f214d1e)

1. `4126955` Add offline o200k Harmony tiktoken bundle for air-gapped eval.
2. `f214d1e` Fix TRIM/RL training path so on-policy rollouts stay on vLLM after adapter updates.

Notable new/changed tracked files: `TRIM/trim/training/policy_digest.py`, `TRIM/trim/training/train_checkpoint.py`, `TRIM/trim/training/train_query_sampler.py`, `TRIM/trim/training/opd_prompt_encoding.py`, `TRIM/trim/training/vllm_hybrid.py`, `TRIM/trim/training/four_cell_runtime.py`, `TRIM/trim/training/hf_rl_opd_client.py`, `TRIM/scripts/run_train.py`, `TRIM/trim/cli/launch.py`, and the offline `TRIM/share/tiktoken-bundle/*` set; new test `TRIM/tests/test_trim_rl_fixes.py`.

## Conflict resolution: `TRIM/trim/training/vllm_rollout_worker.py`

This was the only tracked file with local modifications before the sync, and upstream `f214d1e` also edited the same region of `main()`.

- Local change (runtime fix, per CLAUDE.md / project memory): environment-driven `disable_custom_all_reduce` default — force NCCL all-reduce on gm402 multi-card TP to avoid vLLM engine restart deadlock at step>=1, gated by env `DISABLE_CUSTOM_ALL_REDUCE` (default `"1"`). Also adds `import os` and logs `disable_custom_all_reduce` in the `loading_llm` event.
- Upstream `f214d1e` change: config-driven `disable_custom_all_reduce` (applied only when `cfg.get("disable_custom_all_reduce") is not None`), plus LoRA `max_lora_rank` auto-detection from `adapter_config.json` and a new `lora_load` log line.

Merge resolution (re-applied on top of upstream version, no merge commit):

- Kept upstream config-driven block verbatim (`disable_ar = cfg.get(...)`; apply when not None).
- Added an `else:` branch that applies the environment-driven default **only when the config does not set the value**, so upstream explicit-config behavior is unchanged and the gm402 runtime fix stays active by default.
- Kept `import os` and the `disable_custom_all_reduce` key in the `loading_llm` log line.
- Upstream LoRA-rank detection and `lora_load` log preserved unchanged.
- Verified with `python3 -m py_compile TRIM/trim/training/vllm_rollout_worker.py` -> OK.

Resulting diff vs `HEAD` is limited to this single file (the three additions above). All other tracked files match `f214d1e` exactly.

## Untracked local artifacts preserved

Existing untracked TRIM experiment/audit/YAML files and `TRIM/.tiktoken_cache/` were preserved in place; no `git clean` was run. No new tracked files were committed (sync is ff-only; local patch remains uncommitted as before).

## Fetch configuration fix

- Before sync, `remote.origin.fetch` was `+refs/heads/main:refs/remotes/origin/main` only, so a plain `git fetch origin` did not retrieve this branch. Fixed to `+refs/heads/*:refs/remotes/origin/*` so future fetches retrieve all remote heads.
- Authenticated fetch used a token from `/mnt/songzijun/mify_api.env` (`GITHUB_TOKEN`) via a one-shot URL; the token is not persisted to `.git/config` or reflog.
- VSCode `GIT_ASKPASS`/credential-helper env vars were unset and `GIT_TERMINAL_PROMPT=0` was set during sync to avoid interactive credential prompts hanging in this non-interactive shell.
- Note: `git stash` and full-tree `git status` time out on this network filesystem; the restore was done with `git checkout -- <file>` on the explicit path instead, and status checks were run with explicit file paths.

## Caveats / follow-up

- The single re-applied local patch remains an uncommitted working-tree modification (consistent with prior sync practice). It is not a formal artifact and must not be reported as upstream code.
- Subsequent experiment runs using this code must still be independently audited; this sync record only documents code provenance.
