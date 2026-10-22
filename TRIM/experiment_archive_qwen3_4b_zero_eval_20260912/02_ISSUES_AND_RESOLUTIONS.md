# 问题与解决方式（Qwen3-4B zero eval, job t-20260911223628-dhoxl）

按时间顺序记录本次实验遇到的全部问题、根因、解决方式。分为两类：
- **A. 提交前设计阶段规避的问题**（通过 preflight/entry 设计一次性预防，未实际触发）
- **B. 运行中实际发生的问题**（需要监控发现并处理）

---

## A. 提交前规避的问题（preflight 一次通过，无失败 attempt）

本次相对 gpt-oss-20b eval 先例（[[trim-gptoss20b-eval-all-20260911]]，attempt0/1 各失败一次）的最大改进：**preflight 加 `load_upstream_modules()` 自愈断言**，在 64 worker spawn 前（vLLM actor 起来前的 ~7min 之前）就抓到所有依赖缺失，**一次过，无失败 attempt**。

### A1. chz / tinker / json_repair 缺失（import-chain 硬依赖）
- **根因**：upstream_api worker 调 `load_upstream_modules()`→`training.train_rl`→顶层 `from tinker_cookbook.rl.types import ...`→`import chz`+`import tinker`+`from tinker_cookbook.completers import ...`；`harness/agent.py` 顶层 `import json_repair`。基础镜像 `3050749` 干净，这些包不在 vllm/transformers/pyserini 传递依赖里。
- **解决**：entry 里 `pip install --no-deps json_repair langchain-text-splitters wikipedia-api fastembed baseten-performance-client trl wandb modal pysqlite3-binary openai-harmony pydantic-settings` + `pip install --no-deps chz tinker`。`--no-deps` 关键：tinker metadata 声明 unconstrained `transformers`，普通装会把 transformers 升到 5.x 破坏四包锁版。
- **验证**：preflight 打印 chz 0.4.0 / tinker 0.28.1 / json_repair 0.63.4，`load_upstream_modules()` 返回 18 模块，`UPSTREAM_MODULES_OK`。
- **证据**：gpt-oss 先例 attempt0（`t-20260911101823-9wg6w`）正是 `ModuleNotFoundError: No module named 'chz'` 失败，attempt1 缺 json_repair 失败，attempt2 才成功。本次 preflight 前置断言避免了这 2 次浪费。

### A2. pyserini 2.0.0 版本漂移
- **根因**：`pip install pyserini`（不锁版本）会装到 2.4.0，致 preflight 断言 `pyserini==2.0.0` 失败。
- **解决**：显式 `pyserini==2.0.0`。

### A3. ANSERINI_CLASSPATH 指向错误
- **根因**：pyserini 的 `pyserini.pyclass` 把 `configure_classpath` 包在 bare `try/except` 里静默吞错，anserini fatjar 可能落不到 JVM classpath → `autoclass('io.anserini.*')` 报 `NoClassDefFoundError`。`jnius_config.add_classpath` 只在 JVM 启动前生效。
- **解决**：(a) entry 显式 `export ANSERINI_CLASSPATH=$(find <pyserini>/resources/jars -name '*.jar' | paste -sd:)` 指向 fatjar；(b) PYTHONPATH 最前加 `sitecustomize_dir`，让 `sitecustomize.py` 在解释器启动时（JVM 前）自动 `add_classpath`。双保险。
- **验证**：preflight `from jnius import autoclass; autoclass("java.lang.String")` + `LuceneSearcher(index).num_docs` 返回 100195。

### A4. JVM 路径硬编码风险
- **根因**：不能假设 `JAVA_HOME` 已设或路径固定；非空 `JAVA_HOME` 不代表 JVM 就绪。
- **解决**：`find /usr /opt -name libjvm.so` 探测 → `test -f` + `ldd` + `autoclass("java.lang.String")` 三层 preflight；探测不到才 apt 装 openjdk-21。镜像自带 `/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so`，`JAVA_HOME` 推算为 `/usr/lib/jvm/java-21-openjdk-amd64`（libjvm.so 上三层）。

### A5. heredoc JuiceFS 回读失败
- **根因**：wrapper 含 `python - <<PY`（smoke_test_tool_call）和 `cat <<EOF`（MANIFEST）。直接 source JuiceFS 上的脚本，bash 回读脚本文件时遇 JuiceFS IO 抖动报 `error reading input file`（[[cloudml-trim-launch-heredoc-juicefs-reread]]）。
- **解决**：`cp wrapper /tmp/run.sh && source /tmp/run.sh`，脚本本体走本地副本，模型/数据/tiktoken 仍走 JuiceFS 挂载。

### A6. source /tmp 副本导致 TRIM_ROOT 误算（本次新发现，最关键的设计修复）
- **根因**：wrapper line 27 `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`。source `/tmp/run.sh` 时 `BASH_SOURCE[0]=/tmp/run.sh`→`SCRIPT_DIR=/tmp`→`TRIM_ROOT=$(cd /tmp/.. && pwd)=/`（错！）。所有 `REL_MODELS=../../models` 等相对路径解析到 `/` 下，`run_eval` 的 `--out`/`--index-path`/`--corpus-path` 全错。
- **解决**：source 后立即 `export TRIM_ROOT=/mnt/songzijun/Capability_Evolution/TRIM; cd $TRIM_ROOT` 并**重算所有 REL_*/BCPLUS_*/MASTER_LOG**（wrapper 在 source 时已用错误 TRIM_ROOT 算过这些变量，必须覆盖）：
  ```bash
  export TRIM_ROOT="/mnt/songzijun/Capability_Evolution/TRIM"
  cd "${TRIM_ROOT}"
  REL_MODELS="../../models"; REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
  REL_OUT="outputs"; REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
  REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"
  BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
  BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"
  VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"
  mkdir -p "${REL_LOGS}"; MASTER_LOG="${REL_LOGS}/master.log"
  ```
- **验证**：日志确认 `TRIM_ROOT=/mnt/songzijun/Capability_Evolution/TRIM`、`cd` 到正确目录、`REL_*` 正确、`run_eval --out outputs/eval_h1_...` 正确。

### A7. wrapper 覆盖 PYTHONPATH
- **根因**：wrapper line 81 `export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"` 在 source 时执行，**覆盖** envConfigs 设的 PYTHONPATH（含 sitecustomize_dir/harness-1/tinker-cookbook）。worker 子进程经 `subprocess_env_for_mask→dict(os.environ)` 继承父 PYTHONPATH，故 sitecustomize 不会在 worker 自动加载。
- **解决**：source 后（在 A6 重设之后）re-prepend 完整 PYTHONPATH：
  ```bash
  export PYTHONPATH="${TRIM_ROOT}/sitecustomize_dir:${TRIM_ROOT}/external/harness-1:${TRIM_ROOT}/external/harness-1/tinker-cookbook:${TRIM_ROOT}:${TRIM_ROOT%/*}/SCAPE-EasyOPD:${PYTHONPATH}"
  ```

### A8. 不依赖 Chroma 服务
- **设计**：wrapper 已 `export HARNESS1_FORBID_CHROMA=1`（line 84）+ `--retrieval-backend local_bm25 --reranker none --offline`。worker `harness1_api_worker.py` 再设一次 `HARNESS1_FORBID_CHROMA=1`；`env_bridge.build_eval_toolset` 走 `build_local_toolset`（`chroma_initialized=False`）；`offline_credentials.py` 填 `CHROMA_API_KEY/CHROMA_DATABASE=local-offline`。envConfigs 兜底 `SCAPE_CHROMA_PATH=/tmp/empty_chroma`+`SCAPE_LOCAL_OPENAI_EMBEDDINGS=1`。代码库无 `CHROMA_HOST/PORT/URL/HttpClient`。
- **结果**：全程未起 Chroma 服务，local_bm25（PyseriniBackend Lucene + jsonl corpus）正常工作。

---

## B. 运行中实际发生的问题

### B1. rank24/rank32 grep_corpus 静默 hang（无超时）—— 最终导致 partial
- **现象**：
  - rank32：qid=822 turn=8 `grep_corpus` 后 worker.log 停更 11.4min，TURNS.jsonl 最后一条 `stage=attempt_start` 无 `attempt_result`，无 Traceback。后**自行恢复**（qid=822 最终完成）。
  - rank24：qid=1004 turn=4 `grep_corpus` 后 worker.log 停更 6.75min+，同型 hang，**未恢复**，最终任务停止时只完成 10/13 query，无 DONE.json。
- **根因**：`grep_corpus`（Lucene 正则/grep 搜索）对某些 query 会无限卡死，worker **无单 query / 单工具调用超时**。这与 [[trim-upstream-api-eval-chz-tinker-import-chain]] 记录的 gpt-oss-20b **harmony parser format_retry 死循环是不同问题**——Qwen3-4B 用 hermes parser 无 format_retry，但暴露了另一个 hang 源。
- **诊断方法**：
  - worker.log mtime 与当前时间差 >5min + 无 Traceback → 静默 hang
  - TURNS.jsonl 最后一条 `stage=attempt_start`（无 `attempt_result`）→ 卡在某工具调用
  - worker.log 最后一行是 `grep_corpus` → 卡在该工具
- **连锁影响**：`run_replicated_api_eval`（`harness1_api_launch.py:259-260`）用 `for proc in procs: rc = proc.wait()`，**无 per-shard 超时**。rank24 hang → 主进程 `proc.wait(rank24)` 永久阻塞 → 即使其他 63 shard 全完成也**不合并、不退出**，CloudML 永显 running，后于 runTime 16609s（4.6h）被停止（pod 回收）。
- **解决（partial 合并）**：任务 stopped 后，从 63 个 `DONE.ok=true` shard 的 `PER_QUERY.jsonl`（827 条 trace）用**官方** `summarize_api_traces` + `write_eval_outputs` 手动合并：
  - 脚本 `merge_qwen3_4b_zero_eval_partial.py`
  - **注意**：官方 `merge_traces(shard_traces, original_rows)` 会 `raise RuntimeError` 若 original rows（830）没全覆盖（line 88-89）——partial 场景**不能用** `merge_traces`，要手动按 `query_id` 建 dict、按 830 rows 顺序对齐、缺的标 `missing`。
  - 产出均 `partial=True, ok=False`，`DONE.json` 记 `n_missing=3`（qid 1004/289/790）+ `partial_reason`。
- **launcher 修复方向（未实施，供后续 recovery）**：
  - 给单 query agent loop 设**总超时**或**工具调用次数上限**，超限 `write_infra_error_query` skip 该 query 继续下一个；
  - 或给 `proc.wait(timeout=N)` 加 per-shard 上限，超时把该 shard 标 partial 并继续合并；
  - 修后用**新 job name + 新 --out** 重跑全 830（不覆盖本次输出）。

### B2. 无其他报错
- 全程无 Traceback / ModuleNotFound / OOM / EngineDeadError / CUDA out of memory。
- 56-63 个 worker 全程活跃，vLLM KV cache 长期 93-94% 满负荷（Running 27-32 / Waiting 26-33 reqs），单 actor 承 64 shard 稳定。
- preflight 一次通过（torch 2.10.0+cu128 / vllm 0.19.1 / transformers 4.57.6 / pyserini 2.0.0 / structlog 26.1.0 / chromadb 1.5.9 / chz 0.4.0 / tinker 0.28.1 / json_repair 0.63.4 / gpu_count 8 / LUCENE_OK num_docs=100195 / UPSTREAM_MODULES_OK 18）。

---

## C. 时间线
| 时间 | 事件 |
|---|---|
| 22:36:28 | 提交 job t-20260911223628-dhoxl |
| 22:37:28 | 8 卡 gang 调度成功，pod Started |
| 22:38-22:42 | pip 装四包 + import-chain deps + chz/tinker/json_repair（`--no-deps`） |
| 22:42:26 | PREFLIGHT 全过（含 load_upstream_modules 18 模块） |
| 22:44:54 | source wrapper（TRIM_ROOT/PYTHONPATH 已修正） |
| 22:46:56 | vLLM actor（GPU7:8040）就绪 + smoke test 通过 |
| 22:46:58 | run_eval 启动，64 shard spawn |
| 22:48+ | 64 worker 全部开始评测 |
| 23:31 | 402/830 started（48.4%），速率 ~8.9 qid/min |
| 00:01 | 669/830 started（80.6%） |
| ~00:05 | 发现 rank32（qid=822）+ rank24（qid=1004）grep_corpus hang |
| 00:12+ | 任务 stopped（runTime 16609s/4.6h），827/830 query 已落盘 shard PER_QUERY.jsonl |
| 07:53 | 用 merge 脚本从 63 shard 合并 partial 结果（827 query） |

## D. 关键教训
1. **preflight 自愈断言前置**（`load_upstream_modules()` 全 import）能避免 actor 起来一遍 ~7min 后 worker 才崩的浪费——本次相对先例的最大改进。
2. **source /tmp 副本的 wrapper 必须重设 TRIM_ROOT + REL_* + PYTHONPATH**——`BASH_SOURCE[0]` 误算是最易踩的坑。
3. **grep_corpus hang 是独立于 harmony 死循环的新 hang 源**——hermes parser 不解决它；worker 缺单 query/工具超时是根因。
4. **`run_replicated_api_eval` 无 per-shard 超时**——单 shard hang 会拖死整个合并/退出，需在 launcher 层加防护。
5. **partial 合并不能用 `merge_traces`**（强制 830 覆盖），要手动 by_id 合并 + 标 missing + 用 `summarize_api_traces`/`write_eval_outputs` 算指标。
