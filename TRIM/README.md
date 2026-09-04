# TRIM

Harness-1 / Harness-G / BrowseComp+ **train** and **eval** line.

This is the successor of the SCAPE `run_train.py` / `run_eval.py` launchers.
SCAPE remains in the umbrella repo as the historical experiment tree; new
closed-loop runs should start here.

`--harness` selects the search runtime. Default is `Harness-1` (v8d tools).
Pass `--harness Harness-G` for the graph-menu runtime (`init` / `select` /
`lookup` / `answer`). In both cases `--component` lists **advanced** extras
to internalize, not the always-on basic tools:

- **Harness-1 basic:** `search_corpus`, `read_document`, `curate`, `end_search` + working memory.
- **Harness-1 advanced (v8d):** `evidence_graph`, `verify_tool`, `sentence_compress`, `auto_populate_first_search`, …
- **Harness-G basic:** `init`, `select`, `lookup`, `answer` + graph menu / working memory.
- **Harness-G advanced:** `answer_with`, `bridge_entities`, `entity_synonyms`, `sentence_neighbors`, `hybrid_init_retrieve`, `snc_frontier`, …

Student rollouts keep advanced flags **off**. Teacher DualView turns them on
as a privileged side branch; OPD projects those events onto the basic tools
(`answer_with` → `select`, unreachable bridge LOOKUP skipped, SNC previews
skipped then ALIGN to `select` / `lookup`).

```text
TRIM/
├── scripts/run_train.py    # training entry
├── scripts/run_eval.py     # closed-loop eval entry
├── scripts/run_sft.py      # Harness-1 Tinker SFT entry
├── trim/                   # package (CLI, adapters, training, eval)
├── manifests/browsecomp_plus_830/
├── external/harness-1/     # pinned Harness-1 runtime
└── tests/
```

## Quickstart

From `Capability_Evolution/`:

```bash
PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_train.py \
  --harness Harness-1 --benchmark BC+ \
  --model_name /mnt/songzijun/models/pat-jj_harness-1-full/harness-1 \
  --train_method trim --component all

PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_eval.py \
  --harness Harness-1 --benchmark bcplus_test_166 \
  --model_name /mnt/songzijun/models/pat-jj_harness-1-full/harness-1 \
  --component all

PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_train.py \
  --harness Harness-G --benchmark BC+ \
  --model_name /path/to/checkpoint \
  --train_method trim --component all
```

`--benchmark`:

- `bcplus_test_166` — official 166-query test split
- `bcplus_full` — 830-query pool (664 train + 166 test)
- `BC+` — dataset family; eval still defaults to the 830 pool

`--model_name`: path to the checkpoint being trained or evaluated.
`harness-1` remains a shorthand for the default local Harness-1 checkpoint.

`--train_method`: `opd` | `rl` | `rl+opd` | `scape+rl` | `trim`

`trim` is CISPO + projected teacher actions + SEED-scale OPD (the method
formerly launched as `scape+seed`; that flag remains an alias).

## SFT (Harness-1 Tinker)

SFT is a separate line from `run_train.py`. It uses the pinned Harness-1
`training/train_sft.py` + Tinker cookbook, with the public 899 GPT-5.4 v8d
trajectories from [`pat-jj/harness-1-train-data`](https://huggingface.co/datasets/pat-jj/harness-1-train-data)
(`stage=sft` only). Defaults match `external/harness-1/training/launch_sft_training.sh`.

```bash
PYTHONPATH=TRIM python TRIM/scripts/run_sft.py
PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --model-name openai/gpt-oss-20b
PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --pack-only
PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --smoke --dry-run
```

- **Model:** `openai/gpt-oss-20b` (Tinker id; also accepts `gpt-oss-20b`)
- **Data pack:** `TRIM/data/harness-1-sft-data.tar.gz` (in-repo 899 trajectories; override `--sft-data` / `TRIM_SFT_DATA`)
- **Recipe:** 3 epochs, batch 128, lr `5e-6`, LoRA rank 32, `max_length=32768`, `min_recall=0.1`, save/eval every 50
- **v8d flags:** same as Harness-1 SFT generation / RL (`VERIFY_TOOL`, `EVIDENCE_GRAPH`, …)
- Requires `TINKER_API_KEY` (in `external/harness-1/.env.local` or the environment) except for `--dry-run` / `--pack-only`
- Interpreter: `TRIM_SFT_PYTHON` / `--python`, else a local env that can `import tinker`, else `uv run --project external/harness-1`

`--train-data`:

- `sec` (default) — Harness-1 SEC RL pool (~3453 queries)
- `bcplus_train_664` — official BC+ train split

Local retrieval (no Chroma):

- **Train (SEC):** Lucene BM25 at `/data/ppnm/harness-1-sec-corpus/indexes/bm25`, built from the parquet corpus. Rebuild with `PYTHONPATH=TRIM python TRIM/scripts/build_sec_bm25_index.py`. Non-smoke training fails closed if this index is missing.
- **Eval (BC+):** Lucene BM25 at `SCOPE/external/BrowseComp-Plus/indexes/bm25`
- **Query packs:** `/data/ppnm/harness-1-rl-data.tar.gz` (SEC train) and BrowseComp-Plus `topics-qrels/` (eval)

`--n-queries` optionally caps the selected pool. SEC runs default the eval
score split to `bcplus_830`; `bcplus_train_664` defaults to `bcplus_test_166`.

`--tp` (eval only): data-parallel replica count. `--tp 8` starts 8 independent
model servers (one GPU each unless `--tensor-parallel-size` is set), round-robin
shards the eval set, then merges `PER_QUERY.jsonl` back into original order.

```bash
PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_eval.py \
  --harness Harness-1 --benchmark bcplus_test_166 \
  --model_name /mnt/songzijun/models/pat-jj_harness-1-full/harness-1 \
  --component all --tp 8
```

Other eval throughput knobs:

- `--eval-gpus 0,1,2,3,4,5,6,7` — GPU pool for replicas
- `--tensor-parallel-size` — GPUs *inside* each replica (default 1 when `--tp > 1`)
- `--max-num-seqs 256` — vLLM concurrent sequences per replica
- `--eval-chunk-size 32` — optional micro-batches inside a replica
- `--eval-stagger-s 2` — delay between replica launches
- `--gpu-memory-utilization 0.90`

Eval still needs `SCAPE-EasyOPD` on `PYTHONPATH` for skip-to-anchor projection,
and BrowseComp-Plus under `SCOPE/external/BrowseComp-Plus`.
