# BCPlus Full GPU4 Run Audit — 2026-09-10

## Classification

This is a failed diagnostic attempt. It is not a formal evaluation artifact.

## Requested execution

The requested command was `TRIM/scripts/run_bcplus_full_gpu4_eval.sh`, which defines six sequential experiments:

1. GPT-OSS-20B / zero
2. GPT-OSS-20B / all
3. Harness-1 / zero
4. Harness-1 / all
5. Qwen3-4B-Instruct-2507 / zero
6. Qwen3-4B-Instruct-2507 / all

## Attempt

- Run ID: `20260910_1912_bcplus_full_gpu4`
- Environment: `/opt/TRIM`
- Python: 3.12.13
- vLLM: 0.19.1
- Torch: 2.10.0+cu128
- GPUs visible: 4 × NVIDIA M402 80GB
- Effective layout: zero GPU 3; all actors 0,1,2 and verifier 3
- Ports: zero 8040; all actors 8040,8042,8044; verifier 8050
- Code commit: `41641ea7c886edef4705b10316d31f2a98153966`
- Logs: `outputs/logs/20260910_1912_bcplus_full_gpu4/`

## Outcome

The first experiment did not reach `run_eval.py`. GPT-OSS-20B loaded successfully in vLLM, but API initialization failed while `openai_harmony` attempted to load the GPT-OSS Harmony vocabulary. The required local `o200k_harmony`/tiktoken vocabulary was absent, and the external download timed out. The API therefore never became available at `/v1/models`, so the script remained in `wait_api` and no query output was produced.

The following were verified absent from `/root/.cache`, `/opt/TRIM`, and the model roots searched:

- `o200k_harmony.tiktoken`
- `o200k_base.tiktoken`
- `cl100k_base.tiktoken`

No experiment output is counted as smoke, partial, or formal evaluation data. The attempt's original logs are retained. No scorer or result was modified to bypass the failure. No six-experiment claim is made.

## Preconditions passed

- `bash -n scripts/run_bcplus_full_gpu4_eval.sh`
- `python -m py_compile scripts/run_eval.py`
- `pytest -q tests/test_bcplus_full_gpu4_eval_script.py tests/test_corpus_validate_and_health.py` — 8 passed
- BM25 index and full corpus were present
- Three model roots were available; the script's missing aliases were mapped to existing local model roots with symlinks:
  - `/mnt/songzijun/models/gpt-oss-20b` → `/mnt/songzijun/models/openai/gpt-oss-20b`
  - `/mnt/songzijun/models/harness-1` → `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`

## Recovery attempt 1

- Run ID: `20260910_2010_bcplus_full_gpu4_recovery1`
- The supplied `/mnt/songzijun/tiktoken-bundle.zip` was unpacked to `/tmp/tiktoken-bundle`.
- Manifest and SHA-256 checks passed for `o200k_base.tiktoken` and `cl100k_base.tiktoken`.
- Explicit `TIKTOKEN_ENCODINGS_BASE`, `TIKTOKEN_RS_CACHE_DIR`, and `TIKTOKEN_CACHE_DIR` variables were set; `openai_harmony.load_harmony_encoding("HarmonyGptOss")` passed offline.
- GPT-OSS vLLM reached `Application startup complete` and the zero evaluation entered `run_eval.py`.
- All three upstream API workers then failed before querying because the approved `/opt/TRIM` environment lacks the `chz` Python module (`ModuleNotFoundError: No module named 'chz'`).
- Output: `outputs/eval_h1_bcplus_full_gpt-oss-20b_zero_gpu3_20260910_2010_bcplus_full_gpu4_recovery1/`
- Classification: failed diagnostic attempt; no query result or formal artifact.

## Required recovery

Provide `chz` in the approved `/opt/TRIM` environment (or an approved offline wheel/bundle), then run a fresh attempt with a new `RUN_ID` and output/log prefix. Both failed attempts must remain separate and must not be overwritten or merged.
