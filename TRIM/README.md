# TRIM

Harness-1 / BrowseComp+ **train** and **eval** line.

This is the successor of the SCAPE `run_train.py` / `run_eval.py` launchers.
SCAPE remains in the umbrella repo as the historical experiment tree; new
closed-loop runs should start here.

```text
TRIM/
├── scripts/run_train.py    # training entry
├── scripts/run_eval.py     # closed-loop eval entry
├── trim/                   # package (CLI, adapters, training, eval)
├── manifests/browsecomp_plus_830/
├── external/harness-1/     # pinned Harness-1 runtime
└── tests/
```

## Quickstart

From `Capability_Evolution/`:

```bash
PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_train.py \
  --harness Harness-1 --benchmark BC+ --model_name harness-1 \
  --train_method rl --component all

PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_eval.py \
  --harness Harness-1 --benchmark bcplus_test_166 \
  --model_name /mnt/songzijun/models/pat-jj_harness-1-full/harness-1 \
  --component all
```

`--benchmark`:

- `bcplus_test_166` — official 166-query test split
- `bcplus_full` — 830-query pool (664 train + 166 test)
- `BC+` — dataset family; eval still defaults to the 830 pool

`--train_method`: `opd` | `rl` | `rl+opd` | `scape+rl` | `trim`

`trim` is CISPO + projected teacher actions + SEED-scale OPD (the method
formerly launched as `scape+seed`; that flag remains an alias).

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
