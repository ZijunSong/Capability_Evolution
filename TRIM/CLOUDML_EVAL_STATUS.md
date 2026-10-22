## 2026-09-09 — EasyOPD SDPO Qwen3-4B Recovery22

### `t-20260909121252-h5um7` — stopped partial

- CloudML state: `stopped`; job `easyopd-sdpo-qwen3-4b-8gpu-recovery22-20260909`; created `12:12:52`, Pod started `13:40:59`, stopped `17:14:08`, reported runtime `12781s` (about 3h33m). Pod `tj-3050749-t-20260909121252-h5um7-master-0` was on an 8 x `gm402-80g` node. The Pod has been deleted, so current `cml custom_train logs` returns 404; preserve historical events and output evidence.
- Submitted configuration: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`; JuiceFS volume `songzijun`, endpoint/cluster `bj9-cloudml`, mounted `/mnt/songzijun`; `OPSD_EPOCHS=500`; `SEEDS=43,44`; `SKIP_EVAL=0`; isolated output root `/mnt/songzijun/EasyOPD-baseline/outputs/opsd_suite_recovery22_20260909`; 8-GPU resource `cloudml.ng1gm402-8-8.20-224`.
- Dependency command pinned Torch `2.6.0`, Transformers `4.51.1`, vLLM `0.8.5.post1`, PEFT `0.18.1` and related EasyOPD packages. The image command explicitly runs an `EasyOPD.list_methods()` preflight and asserts `sdpo` is registered; however, because the Pod was removed, the final preflight/log text is no longer retrievable through CloudML logs. No claim of successful preflight is made beyond the submitted command and resulting output.
- Output audit: seed43 exists with `run_metadata.json`, `verl_data/sdpo_train.parquet`, and `rl_samples.jsonl` containing 5 records at `global_step` 76–80. Seed44 has no output directory. No completed checkpoint, `eval_metrics.json`, or complete three-seed artifact exists in this output root.
- Events show successful scheduling, image pull, container start, and later `Stopping container pytorch`, `PyTorchJobSuspended`, and PodGroup deletion. No event indicates OOM or image failure. The output indicates seed43 made partial progress, but training stopped before seed44.
- Classification: **stopped partial**, not a complete SDPO result and not formal. Since the task is stopped, remaining-time ETA is unavailable; a hypothetical fresh recovery cannot be estimated from this single partial seed. Do not restart, stop, delete, or overwrite this task; any fix/retry must use a new job and output prefix.

## 2026-09-09 — EasyOPD OPHSD Qwen3-4B three-seed suite

### Recovery3 `t-20260909141312-8a0ej` — stopped partial

- CloudML state: `stopped`; created `2026-09-09 14:13:13`, Pod started `14:27:14`, stopped/completed `17:14:11`, reported runtime `10007s` (about 2h47m). Pod was `tj-3050749-t-20260909141312-8a0ej-master-0` on an 8 x `gm402-80g` node. Do not treat `stopped` as success.
- Events show successful scheduling, image pull and container start. At stop time CloudML emitted `Stopping container pytorch`, `PyTorchJobSuspended`, and a transient `FailedKillPod` cleanup warning. The Pod no longer exists, so `cml custom_train logs` returns 404; preserve output and event evidence rather than attempting to overwrite or resume in place.
- Submitted configuration audit: `CUDA_VISIBLE_DEVICES=0..7`, `OPSD_SAVE_FREQ=100`, `OPSD_EPOCHS=500`, `SKIP_EVAL=1`, queue `2334`, one node with 8 GPUs, JuiceFS `songzijun` mounted at `/mnt/songzijun` with endpoint/cluster `bj9-cloudml`. The image command pinned Torch `2.6.0`, vLLM `0.8.5.post1`, Transformers `4.51.1`, Ray `2.47.1`, and related EasyOPD dependencies, then ran the three seeds sequentially through `run_method.sh`.
- Output root: `/mnt/songzijun/EasyOPD-baseline/outputs/opsd_suite/ophsd_recovery3_20260909`. Only `seed42` has materialized data: `run_metadata.json`, `rl_samples.jsonl` (5 records; last `global_step=50`), and `verl_data/ophsd_train.parquet`. Seeds `43` and `44` have no output directory. There are no completed canonical `eval_metrics.json` artifacts, and `SKIP_EVAL=1` means evaluation was intentionally not run during training.
- Classification: **stopped partial**, not a complete three-seed result and not a formal artifact. Query/step progress cannot be extrapolated to completion because the attempt stopped before seed43/44. ETA: unavailable; a new recovery would need a fresh job/output prefix and explicit user authorization to restart.

### Prior-attempt preservation check

- `t-20260909094229-ojats`: `failed`, completed `2026-09-09 11:23:43`, runtime `48s`; preserved.
- `t-20260909114009-dqbs0`: `failed`, completed `2026-09-09 13:11:29`, runtime `863s`; preserved.
- `t-20260909131431-2ttgf`: `succeed`, completed `2026-09-09 14:20:25`, runtime `1977s`; remains preserved as its own attempt/output. It must not be conflated with Recovery3 or promoted as a complete Recovery3 three-seed result.

## 2026-09-08 — Harness-1 BC+ full, Qwen3-4B adapter, zero component

### Job `t-20260908205248-wmyuw` — completed

- Classification: `pipeline completed / diagnostic-partial`; not a formal full-vs-zero artifact.
- CloudML: `succeed`; Pod `tj-3050749-t-20260908205248-wmyuw-master-0` reached `Succeeded`, exit code `0`; runtime `1879` seconds; queue `2334`, priority `8`, one node with 8 x `gm402-80g`.
- Output: `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_run_eval_harness1_bcplus_full_qwen3_4b_adapter_8gpu_20260908_memqa_v5`.
- Startup evidence: `PREFLIGHT` passed with Torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, Pyserini `2.0.0`, ChromaDB `1.5.9`, structlog `26.1.0`, 8 GPUs, Java/PyJNI; `IMPORT_AND_LUCENE_PREFLIGHT` passed with a nonempty Lucene index and probe hit.
- Launch evidence: `LAUNCH.json` records Harness-1, `bcplus_830`, `component=zero`, adapter mode, `eval_replicas=8`, `tensor_parallel_size=1`, and max turns `40`; `SHARD_PLAN_eval.json` partitions 830 queries as `104,104,104,104,104,104,103,103` across ranks 0–7.
- Replica/lifecycle audit: all 8 shards contain `READY`, `RESULT`, `SHUTDOWN`, `DONE.json`, `SUMMARY.json`, and shard `PER_QUERY.jsonl`; every `DONE.json` has `ok=true` and `leak_count=0`.
- Query audit: merged `eval/PER_QUERY.jsonl` has exactly `830` rows; shard counts sum to `830` and each shard has unique IDs. Query-level error metrics report `72/830` errors (`error_rate=0.08674698795`), which are evaluation outcomes, not runtime failures. No fatal runtime markers were found in the final shard logs.
- Final summary: `FOUR_CELL_OFFICIAL_SUMMARY.json` reports `status=SR_OPD_CISPO_FOUR_CELL_EVAL`, `protocol_complete_rl_opd=true`, `legacy_tool_token_kl_hook_used=false`, `teacher_leak_rate=0`, and `claim_status=RUNTIME_WIRING_OK`. `CLAIM.json` explicitly says `claim_usable_for_full_vs_zero=false`; a paired full-mask run is required before any full-vs-zero claim.
- Representative all-pool metrics: legal action rate `0.9898368194`, Evidence Recall@5 `0.0118435647`, Evidence Recall@100 `0.0465973551`, mean tool calls `14.85180723`, recall `0.0697007644`, reward `0.6906534397`, mean E2E `1071.8796s`.
- No recovery is required for this attempt. Preserve the completed output and do not promote it to a formal full-vs-zero artifact.

## 2026-09-09 — Harness-1 BC+ full, GPT-OSS-20B SCAPE adapter, zero component

### Attempt 0 — failed startup

- Classification: `failed` (startup preflight only; not a formal artifact).
- CloudML job: `t-20260909092055-fs4v4` (`trim-run-eval-harness1-bcplus-full-gpt-oss-20b-zero-adapter-8gpu-20260909-memqa-v5`).
- Workspace/context: workspace `10063`, context `songzijun-cto`; queue `2334`, priority `8`, one `cloudml.ng1gm402-8-8.20-224` node with 8 x `gm402-80g`, non-preemptible.
- Image/storage: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`; JuiceFS `songzijun`, endpoint/cluster `bj9-cloudml`, mounted read-write at `/mnt/songzijun` through direct CLI JSON.
- Intended setting: GPT-OSS-20B, frozen `bcplus_full` pool, `component=zero`, eight single-GPU eval replicas, adapter `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_8gpu_20260908_recovery3/adapters/scape_seed`.
- Startup evidence: Pod scheduled and 8/8 M402 GPUs were visible; OpenJDK 21 installation and `libjvm.so`/PyJNI checks passed; pinned runtime installation completed with Torch 2.10.0, vLLM 0.19.1, Transformers 4.57.6, Pyserini 2.0.0, ChromaDB 1.5.9 and structlog 26.1.0.
- Failure: the standalone `LuceneSearcher` preflight imported Pyserini's OpenAI encoder before `run_eval.py` initialized local offline credentials and exited with `openai.OpenAIError: Missing credentials` at 2026-09-09 09:27 CST. No model replica or query evaluation started; metrics remain `N/A`.
- Recovery policy: preserve this attempt/output unchanged. Recovery1 uses a new job name/output prefix and initializes the repository's offline credentials before the Lucene import.

### Recovery1 — failed startup

- Classification: `failed` (startup preflight only; not a formal artifact).
- CloudML job: `t-20260909093223-isjti`; fresh output suffix `..._recovery1`.
- Recovery succeeded through 8-GPU visibility, Java 21, `libjvm.so`, PyJNI and `LuceneSearcher` (`lucene_import_ok`). It then failed at the explicit Harness import-chain preflight with `ModuleNotFoundError: No module named 'harness'` because `PYTHONPATH` omitted `/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1`.
- No vLLM replica or query evaluation started; metrics remain `N/A`. Attempt/output are preserved unchanged.

### Recovery2 — enqueued partial

- Classification: `partial` while waiting/starting; not a formal artifact.
- CloudML job: `t-20260909094800-optar`; priority `8`; fresh output `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_run_eval_harness1_bcplus_full_gpt_oss_20b_zero_adapter_8gpu_20260909_memqa_v5_recovery2`.
- Fix: added `/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1` to `PYTHONPATH`, retaining the offline credential initialization and all prior runtime pins/preflights.
- Current state at approximately 09:49 CST: `enqueued`; scheduler reports `fragment detected` for the single-node 8-GPU gang request. No runtime failure has occurred.

### Recovery2 startup update — 2026-09-09 approximately 10:00 CST

- Recovery2 transitioned to `running`; Pod `tj-3050749-t-20260909094800-optar-master-0` is running on `hd1-cloudml-g8gm402-k8s-slave81-20241218.kscn.idc.xiaomi.com` with all 8 M402 GPUs assigned.
- Startup preflights passed: Java 21 `libjvm.so` with `ldd`, PyJNI `java.lang.String`, offline-credential-safe `LuceneSearcher`, Harness `ultra_core`/`tools`, Torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, Pyserini `2.0.0`, ChromaDB `1.5.9`, structlog `26.1.0`, and `torch.cuda.device_count()==8`.
- `LAUNCH.json` and `SHARD_PLAN_eval.json` exist. The launch records `eval_replicas=8`, `tensor_parallel_size=1`, `component_ids=[]`, score split `bcplus_830`, and the fresh recovery2 output prefix. All eight shard configs and query partitions exist.
- At the latest audit, all eight vLLM worker logs had begun loading GPT-OSS-20B with `tensor_parallel_size=1`; no `READY`, first generation result, `DONE.json`, or merged `PER_QUERY.jsonl` marker was present yet. No Traceback, OOM, OutOfMemory, EngineDeadError, or fatal worker marker was observed. The job remains `partial/running`, metrics `N/A`.
- Monitoring remains every 5 minutes until all eight replicas report READY and sustained generation. ETA is not yet reliable; after stable generation, use the historical approximately 5,800-second wall-clock baseline and observed generation-call progress to estimate completion.

### Recovery2 — failed during first LoRA generation

- Classification: `failed`/diagnostic partial; not a formal artifact. CloudML job `t-20260909094800-optar` completed with Pod exit code 1 at approximately 10:05 CST, runtime 853 seconds.
- Startup gate passed completely: 8 GPUs, pinned Torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, Pyserini `2.0.0`, ChromaDB `1.5.9`, structlog `26.1.0`, Java/PyJNI/Lucene, Harness imports, `LAUNCH.json`, and `SHARD_PLAN_eval.json`.
- All 8 shards wrote `READY`, `RESULT`, `SHUTDOWN`, and `DONE.json`, but each `DONE.json` reports `ok=false`; no successful query output or merged `PER_QUERY.jsonl` exists. Each shard attempted its first 104/103-query batch and failed.
- Root cause is identical across all eight vLLM replicas: vLLM 0.19.1 LoRA Triton path `vllm/lora/ops/triton_ops/lora_shrink_op.py` asserts `inputs.is_contiguous()` during GPT-OSS LoRA execution. The assertion raises `EngineDeadError`; this is not an OOM or NCCL failure. The full traceback and shard evidence remain preserved under the recovery2 output prefix.
- Recovery3 is required with a fresh job/output prefix and the repository's existing minimal runtime patch, replacing the assertion with `inputs = inputs.contiguous()` after vLLM installation. No result from recovery2 is eligible for scoring or formal aggregation.

### Recovery3 — stopped before Pod startup due to stale output prefix

- CloudML job: `t-20260909171309-epdku`; state: `stopped` by explicit cleanup before Pod creation.
- Audit finding: the submitted command retained the Recovery2 output path in `--out` despite the Recovery3 job name. Because this could mix or overwrite Recovery2 evidence, it was stopped while still `enqueued`; no Pod or runtime output was produced.
- Recovery2 remains preserved. A corrected Recovery4 submission must use a fresh output suffix and the same vLLM contiguous-input patch.

## 2026-09-06 — Harness-1 BC+ full, GPT-OSS-20B, 8-GPU data parallel

### Attempt 0 — failed

- Classification: `failed` (startup only; not a formal artifact).
- CloudML job: `t-20260906084648-ytyqt` (`trim-run-eval-harness1-bcplus-full-gpt-oss-20b-8gpu-20260906`).
- Workspace/context: workspace `10063`, context `songzijun-cto`.
- Queue/resource: queue `2334`, one `cloudml.ng1gm402-8-8.20-224` node, 8 x `gm402-80g`, `GUARANTEED_PUBLIC`, non-preemptible.
- Image: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`.
- Storage: JuiceFS volume `songzijun`, cluster/endpoint `bj9-cloudml`, subpath `/`, mounted read-write at `/mnt/songzijun`. Submitted through direct CLI `--juicefs_mount_configs`, not `--filename`.
- Runtime evidence: all eight M402 GPUs were visible and the launcher loaded the frozen `bcplus_830` pool.
- Failure: `ModuleNotFoundError: No module named 'torch'` while importing `trim.training.hf_rl_opd_client`; task exited code 1 at 2026-09-06 08:48:21 CST.
- Partial artifact: the default output contains only `LAUNCH.json`; no query traces or metrics. It must not be included in formal summaries.

### Recovery 1 — running partial

- Classification: `partial` while running; not yet a formal artifact.
- CloudML job: `t-20260906085047-alsqh` (`trim-run-eval-h1-bcplus-gptoss20b-8gpu-r1-20260906`).
- Submitted: 2026-09-06 08:50:48 CST; Pod started at 08:51:50 CST.
- Queue/resource/image/storage: identical to attempt 0.
- Fresh output: `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_h1_bcplus_full_gpt_oss_20b_8gpu_20260906_recovery1`.
- Dependency recovery: internal Xiaomi PyPI; `torch 2.10.0+cu128`, `vllm 0.19.1`, `transformers 4.57.6`, `pyserini 2.0.0`; OpenJDK available. Preflight confirmed `torch.cuda.device_count() == 8`.
- Launch setting: `run_eval.py --harness Harness-1 --benchmark bcplus_full --model_name /mnt/songzijun/models/openai/gpt-oss-20b --component all --tp 8`.
- Frozen evaluation identity: score split `bcplus_830`, 830 total query IDs, all ten Harness-1 components enabled, max turns 40, max new tokens 2048, temperature 1.0, search k 10, max model length 32768.
- Parallelism audit: `eval_replicas=8`, per-replica `tensor_parallel_size=1`; ranks 0–5 each have 104 queries, ranks 6–7 each have 103. GPU assignment is rank 0→GPU 0 through rank 7→GPU 7.
- Startup audit: all eight vLLM 0.19.1 EngineCore processes loaded GPT-OSS-20B independently, each reported 56.33 GiB available KV cache and wrote `READY`; all eight `job.json` and `result.json` files continue updating. No traceback, OOM, EngineDeadError, or worker exit is present.
- Stable-run evidence at approximately 09:39 CST: eight replicas were actively processing generation batches after about 35 minutes of evaluation. Current batches contained roughly 99–104 requests and showed approximately 0.7–1.1k aggregate output tokens/s per GPU during active batch completion.
- ETA: prior same-setting completed artifact on 2026-09-05 took about 3h02m from launch to final outputs. Using that empirical baseline, recovery1 is expected to finish around 12:00–12:30 CST on 2026-09-06, subject to query-length tail variance. This is an estimate, not a formal result.
- Formal gate: do not promote until CloudML reaches `succeed`, all eight `DONE.json` files are `ok`, merged `PER_QUERY.jsonl` contains exactly 830 unique query IDs, and final summary/audit/provenance files exist with no invalid or truncated outputs.

#### Monitoring update — 2026-09-06 approximately 10:01 CST

- CloudML remains `running`; runtime was 4,174 seconds. Pod `tj-3050749-t-20260906085047-alsqh-master-0` remains `Running` with no new adverse event after startup.
- All 8/8 shard `READY` files remain present. All eight `vllm_worker.log` files are active and were updated within approximately 2–59 seconds of the audit.
- All ranks have entered generation batch 29. Latest observed within-batch completion was rank0 81%, rank1 87%, rank2 94%, rank3 88%, rank4 100%, rank5 transitioning at 0%, rank6 52%, and rank7 82%. These percentages are transient per-turn batch progress, not completed-query percentages.
- The latest IPC result files on every rank report `ok: true`, with approximately 93–103 outputs in the current/most recently completed batch. No shard has emitted `DONE.json` or `PER_QUERY.jsonl` yet, which is expected before its complete 40-turn closed-loop run finishes.
- Error scan remains clean: zero `Traceback`, `EngineDeadError`, CUDA OOM/out-of-memory, `Killed`, or fatal signals across all eight vLLM logs; parent logs also show no worker failure.
- Updated ETA: 29 of at most 40 closed-loop generation rounds have started after about 57 minutes of active model evaluation. At the observed roughly two minutes per round plus merge/audit time, likely completion is around 10:25–10:45 CST on 2026-09-06. Long-tail tool/search behavior can extend this; previous same-setting wall-clock remains a conservative upper reference of approximately 12:00–12:30 CST.
- Classification remains `partial`/running, not a formal artifact.

## Final audit — 2026-09-06 10:29 CST

- CloudML job `t-20260906085047-alsqh` reached `succeed`; Pod exited code 0 at 2026-09-06 10:29:03 CST and the CloudML task completed at 10:29:04 CST. Total runtime reported by CloudML: 5,819 seconds.
- All 8/8 `READY` markers are present. All 8/8 shard `DONE.json` files report `ok: true` and `leak_count: 0`.
- Per-shard final query counts are rank0–5: 104 each and rank6–7: 103 each; total 830. Per-shard `PER_QUERY.jsonl` files contain exactly those counts, with no duplicate IDs within a shard. The merged artifact is `harness/PER_QUERY.jsonl`, containing 830 rows and the official 830-query pool contract.
- Cross-shard audit found exactly 830 unique query IDs and zero duplicates. `SHARD_PLAN_harness.json` accounts for all eight GPU assignments and the full query pool.
- Final artifacts present: `LAUNCH.json`, `SHARD_PLAN_harness.json`, `harness/PER_QUERY.jsonl`, eight shard `DONE.json`/`SUMMARY.json` files, `FOUR_CELL_OFFICIAL_SUMMARY.json`, and `ADAPTER_RELOAD_AUDIT.json`. `LAUNCH.json` records `eval_replicas: 8`, `tensor_parallel_size: 1`, `max_turns: 40`, `max_new_tokens: 2048`, score split `bcplus_830`, and the GPT-OSS-20B model identity.
- Runtime error scan across all eight vLLM logs found zero traceback, EngineDeadError, OOM, killed, or fatal signals. The summary's `errors: 702` / `error_rate: 0.8458` is an evaluation metric counting query-level outcome errors, not a CloudML/runtime failure; it is retained as reported and not rewritten.
- Official summary/protocol audit: `status=SR_OPD_CISPO_FOUR_CELL_EVAL`, `legacy_tool_token_kl_hook_used=false`, `protocol_complete_rl_opd=true`, `teacher_leak_count=0`, retrieval `pyserini_lucene`, pool SHA256 `f8253234db2bc28eab62063d829bd1a9330f5a97d9ebccc90c5223cdb22ed9a3`, and `adapter_reload.all_reload_ready=true` for the harness cell with no adapter directory.
- Formal artifact gate passed. This recovery1 result is now the sole formal result for this setting; failed attempt0 remains preserved and excluded.
- Final all-pool metrics from `FOUR_CELL_OFFICIAL_SUMMARY.json`: legal action rate 0.8569634171, Evidence Recall@5 0.0118435647, Evidence Recall@100 0.0465973551, recall 0.0991565964, precision 0.1399213951, F1 0.1033496816, reward 0.5527743840, mean turns/tool calls 38.59036145, and mean E2E time 4774.3714 seconds.

### Separate user-requested zero-component attempt — final

- Attempt0 Job `t-20260906085228-3iydv`: `failed` startup, preserved separately. All eight GPUs and the `bcplus_830` pool were visible, but the image lacked Torch and exited with `ModuleNotFoundError: No module named 'torch'`. No metrics were used.
- Recovery1 Job `t-20260906085624-ogokb`: CloudML `succeed`; master Pod `Succeeded`; completed 2026-09-06 10:33:41 CST after 5800 seconds. Fresh output: `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_h1_bcplus_full_gpt_oss_20b_zero_8gpu_20260906_recovery1`.
- Exact setting: GPT-OSS-20B, `component=zero`, `component_ids=[]`, 830-query `bcplus_full`, `--tp 8` data-parallel replicas, one GPU per replica (`tensor_parallel_size=1`), max turns 40, max new tokens 2048, max model len 32768; runtime Torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, Pyserini `2.0.0`.
- Completeness audit: 8/8 shard `DONE.json` are `ok=true` with leak count 0; shard counts `104,104,104,104,104,104,103,103` sum to 830; merged `harness/PER_QUERY.jsonl` has 830 rows and 830 unique query IDs; 8/8 `READY`, `RESULT`, and `SHUTDOWN` lifecycle markers exist; final summary and adapter audit exist. Independent audit `INDEPENDENT_AUDIT.json` is `pass`; `PROVENANCE.json` and `SHA256SUMS` were generated for this attempt.
- Runtime error audit: all eight worker logs and CloudML events show normal exit code 0 with no Traceback, OOM, EngineDeadError, NCCL fatal marker, or worker exit. Query-level evaluation outcome remains error-heavy and must not be confused with process failure.
- Final all-pool metrics: legal action rate=`0.8541243789`; Evidence Recall@5=`0.0118435647`; Evidence Recall@100=`0.0465973551`; mean tool calls=`38.7337349398`; search cost=`29.3939759036`; recall=`0.0965265825`; precision=`0.1419013515`; F1=`0.1027701554`; reward=`0.5486752649`; mean E2E=`4785.3072433s`; p50/p95 E2E=`4810.8180351/4832.5447055s`; errors=`695/830`; error rate=`0.8373493976`; no-error=`0.1626506024`; max-turns reached=`0.9216867470`.
- Final official-test metrics: legal action rate=`0.8660838812`; Evidence Recall@5=`0.0112115127`; Evidence Recall@100=`0.0371246892`; mean tool calls=`38.6265060241`; recall=`0.1139940715`; precision=`0.1761755355`; F1=`0.1245804520`; reward=`0.5639068072`; errors=`143/166`; error rate=`0.8614457831`; no-error=`0.1385542169`.
- Classification: **pipeline completed / diagnostic-partial, not formal artifact**. Raw coverage and provenance completeness pass, but the 695 query-level errors (including 143/166 official-test errors) fail the no-error formal leaderboard gate. Attempt0 and Recovery1 remain separate; no recovery job was created. The 30-minute session monitor was cancelled after completion.

### 2026-09-06 10:17 CST monitoring checkpoint

- CloudML Job `t-20260906085624-ogokb` and its master Pod remain `running`; runtime was 4,774 seconds. No new adverse scheduler or container event appeared after startup.
- All rank0–7 workers remain active: every rank still has `READY`, no rank has `SHUTDOWN`, and worker-log mtimes advanced through 10:16–10:17 CST. Exact generation-call counts inferred from prompt-render starts were rank0–7=`34,35,34,34,35,34,35,35`; current batches had shrunk to roughly 94–102 active queries as some trajectories ended.
- Fatal-log gate remains clear across all eight worker logs: no Traceback, CUDA OOM/OutOfMemoryError, EngineDeadError, NCCL fatal marker, or worker exit. The raw grep `Processed prompts: 100%` counter is not used as a turn count because tqdm emits duplicate 100% records; prompt-render starts provide the auditable generation-call count.
- Completeness remains partial: all eight `READY` markers are present and live `JOB`/`RESULT` exchange is ongoing, but there are no shard `DONE.json`, `SHUTDOWN`, merged `PER_QUERY.jsonl`, or final summary files yet. Metrics therefore remain `N/A`.
- Updated ETA: approximately **10:35–10:55 CST on 2026-09-06**. This uses 34–35 completed/started generation calls out of the 40-turn ceiling and the observed roughly two minutes per generation call, plus merge/finalization time; variable trajectory termination and long-tail prompts can shift it.

## 2026-09-06 — independent GPT-OSS-20B seed repeats

- First seed43/44 attempts (`t-20260906171822-wkk8j`, `t-20260906171854-rore0`) were explicitly stopped by the user at approximately generation call 25. They remain `stopped partial`; their output prefixes are preserved and contain no final `DONE.json`, merged `PER_QUERY.jsonl`, or official summary. They must not be scored or mixed with later attempts.
- Recovery2 seed43 Job `t-20260906195454-hdsvr`, fresh output `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_h1_bcplus_full_gpt_oss_20b_zero_8gpu_seed43_r2_20260906`; Recovery2 seed44 Job `t-20260906195515-napgd`, fresh output `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_h1_bcplus_full_gpt_oss_20b_zero_8gpu_seed44_r2_20260906`.
- Both recovery2 jobs use the same model, frozen `bcplus_830` pool, component zero, max turns/tokens/context, runtime versions, queue, image, JuiceFS, and eight single-GPU replicas as Recovery1. The explicit sampling seeds are 43 and 44.

## 2026-09-08 — Harness-1 BC+ full, Qwen3-4B, zero component, 8-GPU data parallel

- Attempt0 Job `t-20260908095436-txf8l`: **failed startup**, preserved separately. JuiceFS and eight GPUs were assigned, but the image lacked Torch (`ModuleNotFoundError: No module named 'torch'`). No evaluation ran.
- Recovery1 Job `t-20260908105746-0cald`: **failed startup**, preserved separately. Dependency resolution failed because `pyserini==2.4.0` requires Transformers 5 while the attempt pinned `transformers==4.57.6`. No evaluation ran.
- Recovery2 Job `t-20260908110920-h9odx`: **failed/diagnostic**, preserved separately. Dependency and 8-GPU preflight passed and all eight single-GPU vLLM workers loaded the model, but Pyserini 2.4.0 attempted to fetch prebuilt metadata from the public network during import. All eight shards exited before query evaluation; completed query count is 0 and the attempt is not a formal artifact.
- Recovery3 Job `t-20260908114305-bcdau`: **running partial**, priority 8, fresh output `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_harness1_bcplus_full_qwen3_4b_zero_8gpu_20260908_recovery3`. Queue `2334`, one node with 8 x `gm402-80g`, image `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`, JuiceFS `songzijun` mounted read-write at `/mnt/songzijun` with endpoint/cluster `bj9-cloudml` via direct CLI JSON.
- Recovery3 startup gate passed: Pyserini offline metadata shim applied; `libjvm.so` passed `ldd`; `autoclass("java.lang.String")` succeeded; preflight reported Torch `2.10.0+cu128`, Transformers `5.14.1`, vLLM `0.19.1`, Pyserini `2.4.0`, and 8 visible GPUs. Runtime wiring audit passed.
- Actual parallelism is `eval_replicas=8`, one GPU per replica, `tensor_parallel_size=1`; this is not one TP=8 engine. The frozen `bcplus_830` pool is partitioned as 104 queries on ranks 0–5 and 103 on ranks 6–7.
- Stable-run evidence at 2026-09-08 12:03 CST: 8/8 vLLM workers wrote `READY`/`llm_ready`; all eight workers completed multiple generation batches and continued updating IPC jobs/results. Approximate generation-call progress from logs was rank0–7=`16,17,22,17,18,21,19,19` out of a maximum 40 turns. No shard had final `DONE.json`/`PER_QUERY.jsonl` yet, so metrics remain `N/A` and classification remains running partial.
- ETA: approximately **12:50–13:15 CST on 2026-09-08**, based on the slowest shard at about 16/40 generation calls after sustained generation began, roughly two minutes per remaining call, and final merge/audit time. Variable early termination and long-tail prompts may shift this estimate.
- Monitoring transitioned from 5-minute startup checks to every 30 minutes after stable generation began.
- Final audit at 2026-09-08 12:08 CST: CloudML reached `succeed`, the master Pod reached `Succeeded` and exited code 0, and CloudML reported runtime 1,448 seconds. All 8/8 shard `DONE.json` files report `ok=true` and leak count 0; shard query counts are `104,104,104,104,104,104,103,103`, summing to 830. The merged `harness/PER_QUERY.jsonl` contains exactly 830 rows, 830 unique query IDs, and zero duplicates. All eight vLLM workers wrote `READY`; fatal-log scan found zero Traceback, EngineDeadError, CUDA OOM/OutOfMemoryError, Killed, or fatal NCCL markers.
- Final result quality: the official summary reports 74/830 query-level errors (`error_rate=0.0891566265`, `no_error=0.9108433735`) and 20/166 errors on the official-test subset. No explicit truthy `invalid` or `truncated` marker was found in the merged traces. All-pool metrics include legal action rate `0.9902107018`, Evidence Recall@5 `0.0118435647`, Evidence Recall@100 `0.0465973551`, recall `0.0709122337`, precision `0.0366939667`, F1 `0.0406430967`, reward `0.6890038781`, and mean tool calls `14.7722891566`. Official-test recall is `0.0541262452` and reward is `0.6788301275`.
- Classification: **pipeline completed / diagnostic-partial, not formal artifact**. Runtime and 830-query completeness gates pass, but query-level errors are nonzero; additionally `PROVENANCE.json`, `SHA256SUMS`, and `INDEPENDENT_AUDIT.json` are absent, and `CLAIM.json` states `claim_usable_for_full_vs_zero=false` because a paired full-mask run is required. The recovery3 metrics must not be promoted as a formal artifact or a full-vs-zero claim. The 30-minute monitor was cancelled after completion.
- 20:49 CST checkpoint: both CloudML Jobs and Pods are `running`; all 16 rank workers have `READY` and current logs. Seed43 rank0–7 are at generation calls `20,20,20,21,21,20,21,21`; seed44 ranks are all at call 19. Fatal scans across all 16 logs found no Traceback, OOM/OutOfMemoryError, EngineDeadError, NCCL fatal marker, Killed, or worker exit. No final shard `DONE.json`, merged `PER_QUERY.jsonl`, or summary exists yet, so both remain `running partial` with metrics `N/A`.
- ETA at this checkpoint: approximately 21:25–21:50 CST, subject to long-tail query and merge/finalization time.
- Final recovery2 audit: seed43 Job `t-20260906195454-hdsvr` completed `succeed`/Pod `Succeeded` at 21:32:03 CST, runtime 5750s; seed44 Job `t-20260906195515-napgd` completed at 21:36:06 CST, runtime 5992s. Each run has 8/8 `DONE.json` with `ok=true`, 830 merged rows, 830 unique query IDs, 8/8 `READY`/`RESULT`/`SHUTDOWN`, zero fatal runtime markers, `INDEPENDENT_AUDIT.json=status=pass`, `PROVENANCE.json`, and `SHA256SUMS`.
- Three-seed aggregate (completed seed42/43/44 only): all-pool recall mean/sample-SD=`0.0906361955760/0.0067489902921`, reward=`0.541781854277/0.007745281949`, error rate=`0.820481927711/0.0146572591092`; official-test recall=`0.0978421623000/0.0139892384802`. Full per-run metrics and mean/SD/min/max tables were appended to `TRIM/eval.md` without replacing prior content. All three remain `pipeline completed / diagnostic-partial`, not formal artifacts, because query-level errors remain high.


## 2026-09-08 — Priority-8 recovery5 completion

- Classification: `succeed`; runtime completed, but result quality requires audit and is not usable for a full-vs-zero claim.
- CloudML job: `t-20260908122426-oyj3a` (`trim-harness1-bcplus-full-eval-8gpu-recovery5`).
- Workspace/context: workspace `10063`, context `songzijun-cto`; queue `2334`; one node with 8 x `gm402-80g`; priority `8`; non-preemptible.
- Image: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`.
- Storage: JuiceFS volume `songzijun`, explicit CLI endpoint/cluster `bj9-cloudml`, mounted read-write at `/mnt/songzijun`.
- Startup recovery: installed OpenJDK 17 and Python dependencies from the approved Xiaomi PyPI mirror; preflight passed with torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, 8 visible GPUs, and `java_ok`.
- Completion: CloudML succeeded at `2026-09-08 14:41:00`; all 8 shard `DONE.json` files report `ok: true`, `n_queries` totals 830, and merged `harness/PER_QUERY.jsonl` contains 830 unique query IDs.
- Audit: `FOUR_CELL_OFFICIAL_SUMMARY.json` reports `errors=723`, `error_rate=0.8710843373493976`, `no_error=0.12891566265060242`; `CLAIM.json` says `claim_usable_for_full_vs_zero=false` and status `RUNTIME_WIRING_OK`. Keep as a completed but quality-flagged artifact; do not use it as a clean formal metric or full-vs-zero claim without resolving the per-query errors and passing the paired comparison gate.

## 2026-09-08 — pat-jj Harness-1 full evaluation plan (not submitted)

- Classification: `planned / not submitted`; no CloudML job ID and no result metrics.
- Target: Harness-1 `bcplus_full`, model `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`, component `all`, `MAX_TURNS=40`, and `run_eval.py --tp 8`. In this runner, `--tp 8` means eight single-GPU evaluation replicas with per-replica tensor parallel size 1.
- Planned workspace/resource: context `songzijun-cto`, workspace `10063`, queue `2334`, priority `8`, one `cloudml.ng1gm402-8-8.20-224` node with 8 x `gm402-80g`, non-preemptible. The 2026-09-08 16:32:38 CST queue snapshot reported 30 free of 368 GPUs, `AVAILABLESPEC` including 8, with 1 pending and 46 running tasks; single-node 8-GPU gang placement may still queue because free capacity can be fragmented.
- Planned image/storage/tag: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`; task tag `MemQA-General-V5`; JuiceFS volume `songzijun`, endpoint `bj9-cloudml`, subpath `/`, mounted read-write at `/mnt/songzijun`. The YAML intentionally omits storage; submission must use direct CLI `--juicefs_mount_configs` with explicit `"endpoint":"bj9-cloudml"`, not `--filename`.
- Runtime gate follows the successful locked environment: Torch `2.10.0+cu128`, vLLM `0.19.1`, Transformers `4.57.6`, Pyserini `2.0.0`, plus ChromaDB and structlog. Startup checks eight visible GPUs, model/query/index paths, dynamically discovers and validates `libjvm.so`, runs PyJNI Java and Harness import checks, probes the Lucene index, and compiles `run_eval.py` before evaluation.
- Fresh planned output: `/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_run_eval_harness1_bcplus_full_pat_jj_harness1_full_8gpu_20260908`.
- Prepared artifacts: `cloudml_eval_pat_jj_harness1_full_8gpu_20260908_not_submitted.yaml` and `submit_cloudml_eval_pat_jj_harness1_full_8gpu_20260908_not_submitted.sh`. Neither has been submitted or executed.

## 2026-09-09 — Harness-1 BC+ full, Qwen3-4B default component, 8-GPU data parallel

- Classification: `pipeline completed / diagnostic-partial`; not a formal artifact and not valid for a full-vs-zero claim.
- CloudML Job: `t-20260908230448-q3zfh`; workspace `10063`, context `songzijun-cto`, queue `2334`, priority `8`, non-preemptible, one `cloudml.ng1gm402-8-8.20-224` node with 8 x `gm402-80g`.
- Image: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`. Pod `tj-3050749-t-20260908230448-q3zfh-master-0` ran on `hd1-cloudml-g8gm402-k8s-slave82-20241218.kscn.idc.xiaomi.com`; CloudML reached `succeed` at `2026-09-09 00:07:37 CST`, exit code 0, reported runtime 1818 seconds.
- Storage: JuiceFS volume `songzijun`, explicit CLI endpoint/cluster `bj9-cloudml`, mounted read-write at `/mnt/songzijun`; the YAML intentionally omitted the mount and submission used direct `--juicefs_mount_configs`.
- Startup preflight passed: 8 visible M402 GPUs; Torch `2.10.0+cu128`; vLLM `0.19.1`; Transformers `4.57.6`; Pyserini `2.0.0`; JVM `/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so`; `ldd` and PyJNI `java_ok` passed.
- Evaluation identity: Harness-1, `bcplus_full`, `component=default` (8 enabled components), score split `bcplus_830`, model `/mnt/songzijun/models/Qwen3-4B-Instruct-2507`, max turns 40, max new tokens 2048, temperature 1.0, search k 10, max model length 32768. `--tp 8` was audited as 8 single-GPU replicas (`eval_replicas=8`, `tensor_parallel_size=1`, GPU IDs 0–7), not one TP=8 engine.
- Completeness audit: `SHARD_PLAN_harness.json` covers all 830 IDs; all 8 shard `DONE.json` files are `ok=true`; ranks 0–5 have 104 queries and ranks 6–7 have 103; merged `harness/PER_QUERY.jsonl` has 830 rows, 830 unique IDs, and zero duplicates; all 8 `READY` and `SHUTDOWN` markers are present; fatal scan found no Traceback, EngineDeadError, CUDA OOM/OutOfMemoryError, Killed, or fatal NCCL marker.
- Result quality audit: `FOUR_CELL_OFFICIAL_SUMMARY.json` reports all-pool `errors=67/830`, `error_rate=0.0807228916`, `no_error=0.9192771084`, legal action rate `0.9905627117`, Evidence Recall@5 `0.0118435647`, Evidence Recall@100 `0.0465973551`, recall `0.0657190801`, precision `0.0268566175`, F1 `0.0328543603`, reward `0.6900605904`, and mean E2E time `1059.1642s`; official-test errors are `11/166` (`error_rate=0.0662650602`). No truthy `invalid` or `truncated` marker was found, but the 67 query-level errors fail the clean formal-result gate.
- Provenance/formal gate: `RUNTIME_EFFECT_AUDIT.json` passes wiring and reports `claim_status=RUNTIME_WIRING_OK`, while `CLAIM.json` reports `claim_usable_for_full_vs_zero=false`. `PROVENANCE.json`, `SHA256SUMS`, and `INDEPENDENT_AUDIT.json` are absent; retain this as a completed diagnostic-partial attempt and exclude it from formal leaderboard/full-vs-zero summaries. The duplicate 2-minute and 30-minute session monitors were cancelled after completion.
