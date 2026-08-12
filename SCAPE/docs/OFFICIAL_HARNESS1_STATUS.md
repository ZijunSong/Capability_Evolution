# OFFICIAL_HARNESS1_STATUS

## Current official path status

- Harness-1 upstream checkout: present at `external/harness-1`, pinned to `8ac4012167858f6478fb2a8fd840e4550e2af161`.
- Official Python packages: installed in conda env `scape` (`vllm`, `torch`, `transformers`, `chromadb`, `tinker`, `baseten-performance-client`, `anthropic`, `openai`).
- Hugging Face TLS: works when `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` and `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` are set.
- Released model weights: complete from local archive at `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`; `tokenizer.json` and `model-00001-of-00009.safetensors` ... `model-00009-of-00009.safetensors` are present. The incomplete Hugging Face cache remains at `/mnt/songzijun/models/pat-jj_harness-1` and should not be used for serving.
- SCAPE local qrel corpus backend: available at `outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl`.

## Remaining requirements for official Harness-1 Cloud/Chroma evaluation

These variables must be present in the execution environment or `.env.local`; this file intentionally does not record secret values:

- `OPENAI_API_KEY`
- `CHROMA_API_KEY`
- `CHROMA_DATABASE`
- `HUGGINGFACE_TOKEN` if the model repo or rate limits require auth
- Optional reranker: `BASETEN_API_KEY`, `BASETEN_MODEL_URL`
- BrowseComp+ paths:
  - `BROWSECOMPPLUS_QUERIES_PATH`
  - `BROWSECOMPPLUS_QRELS_GOLD_PATH`
  - `BROWSECOMPPLUS_QRELS_EVIDENCE_PATH`
  - `BROWSECOMPPLUS_ANSWERS_PATH`

## Security/permission note

Starting official local vLLM requires `--trust-remote-code` for the `pat-jj/harness-1` model code. The user authorized localhost-only vLLM with `--trust-remote-code` for `pat-jj/harness-1`, and authorized running `external/harness-1/inference/evaluate_harness1_vllm.py`.
