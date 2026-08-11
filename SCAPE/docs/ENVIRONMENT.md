# ENVIRONMENT

Canonical host: **H20** (`/data/ppnm/Capability_Evolution/SCAPE`).

## Required

| Item | Target |
|---|---|
| Python | >= 3.11 (dev smoke may use 3.12) |
| Package manager | `uv` preferred; `pip` acceptable for bootstrap |
| CUDA / driver | visible via `nvidia-smi` |
| torch / transformers | train optional extra |
| vLLM | GPT-OSS capable for Harness-1 serving |
| Harness-1 | `external/harness-1` pinned commit |
| Model | `pat-jj/harness-1` (or local cache) |
| Retrieval | compatible Chroma backend (not SCOPE BM25 fallback) |

## Record into every RUN_MANIFEST

- upstream repo + commit
- model checkpoint revision
- dataset revision
- retrieval/index revision
- CUDA / driver / torch / vllm / transformers
- uv.lock hash (when lock present)

## Bootstrap command

```bash
python scripts/preflight_harness1.py
```
