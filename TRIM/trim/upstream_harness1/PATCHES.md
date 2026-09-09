# Upstream Harness-1 interface patches

Pinned tree: `TRIM/external/harness-1`  
Upstream: https://github.com/pat-jj/harness-1/commit/8ac4012167858f6478fb2a8fd840e4550e2af161

Every local difference from that commit must be an I/O boundary. Component algorithms, state updates, retrieval ranking, budget policy, and reward definitions stay upstream.

## Allowed patches

| File | Change | Why |
| --- | --- | --- |
| `training/train_rl.py` | `SlidingWindowSearchEnv.step_action(action)` extracted from `step(tokens)` after Harmony parse | API eval and token training share the original step lifecycle without rewriting `_execute_tools` |
| `training/train_rl.py` | `selected_context_window()` returns the same RECENT_K / WM split `_render_next_context` uses | API adapter builds messages from original selected context, not a second history policy |
| `training/train_rl.py` | `SlidingWindowSearchEnv(..., openai_client=)` | Inject a local verifier; `_exec_verify` only falls back to `get_config()` when unset |
| `harness/config.py` | Lazy `chromadb` import; `HARNESS1_FORBID_CHROMA=1` refuses `get_chroma_client()` | Local BM25 process must not construct CloudClient |
| `harness/tools.py` | Chroma Search DSL and BM25 embedding function imported inside Chroma tool methods; `Reranker` is `TYPE_CHECKING` only | Tool schema/metadata importable without `chromadb` |

## Recorded upstream behavior (not fixed in this baseline)

- `V8D_*` flags are read at `ultra_core` import time; schema for `verify` / curate `importance` is frozen then. Isolation must be a subprocess with flags set before import.
- Paper ablation `all_harness_mechanisms_disabled` also sets `ABLATE_REVIEW_DOCS_UNAVAILABLE=1`. TRIM `--component zero` does **not** set that flag.
- `evaluate_harness1.py` uses a TokenCompleter, not chat/completions. Official TRIM eval adds a messages adapter in `trim/upstream_harness1/` rather than changing that script's algorithm.
- Adaptive rerank in RL env is constructed with `use_llm=False`. Kept as-is.
- `SearchCorpusTool.__call__` builds `rerank_instruction` override but does not pass it into the reranker. Local BM25 keeps this native no-op and records `adaptive_rerank_instruction_consumed_by_search=false`.
- Vendored `ultra_core.py` still contains a local Harmony fallback import path that is not in the pinned upstream commit. That is a dependency-isolation difference, not an algorithm change; it is not treated as a complete byte-for-byte upstream tree.
- `Trajectory.to_openai_format()` on the OPENAI provider path does not keep `reasoning_content`. API eval records reasoning from the raw chat response separately.
