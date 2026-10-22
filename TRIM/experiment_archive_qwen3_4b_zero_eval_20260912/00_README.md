# TRIM Harness-1 BC+ Full Qwen3-4B "zero" 8-GPU Eval — 实验归档

**归档时间**: 2026-09-12
**CloudML Job ID**: `t-20260911223628-dhoxl`
**Job Name**: `eval-bcplus-full-qwen3-4b-zero-8gpu-20260911`
**最终状态**: `stopped`（partial，827/830 query，partial=True，**非 formal artifact**）

---

## 1. 实验概要

只运行 `scripts/run_bcplus_full_gpu8_eval.sh`（8 卡原生 wrapper）的 **Qwen3-4B zero** 分支：
- harness-1 upstream_api 官方评测路径（`run_eval.py --evaluation-path upstream_api`）
- 单 actor vLLM（GPU 7，端口 8040，served `Qwen3-4B-Instruct-2507`，`--tool-call-parser hermes`）+ TP=64 data-parallel eval worker shard
- retrieval = `local_bm25`（Lucene index + jsonl corpus，**不依赖 Chroma 服务**，`HARNESS1_FORBID_CHROMA=1`）
- setting 对齐 wrapper 原生默认：max_turns=40，max_new_tokens=2048，temperature=1.0，GPU_UTIL=0.70，VLLM_MAX_NUM_SEQS=128，max_model_len=32768，benchmark=bcplus_full（score_split=bcplus_830）
- 锁定四包 runtime：Torch 2.10.0+cu128 / vLLM 0.19.1 / Transformers 4.57.6 / Pyserini 2.0.0

## 2. 文件夹结构

```
experiment_archive_qwen3_4b_zero_eval_20260912/
├── 00_README.md                          ← 本文件（总览+运行指令+结果摘要）
├── 02_ISSUES_AND_RESOLUTIONS.md          ← 中间问题与解决方式（详）
├── 01_run_instructions/                  ← 完整运行指令
│   ├── cloudml_eval_..._not_submitted.yaml      ← custom_train YAML
│   ├── submit_cloudml_..._not_submitted.sh       ← cml 提交脚本（含 JuiceFS CLI JSON）
│   ├── run_bcplus_full_gpu8_eval.sh.source       ← 被 source 的原始 wrapper（复现用）
│   ├── monitor_qwen3_4b_zero_eval_20260911.sh   ← 30min 进度监控脚本
│   └── merge_qwen3_4b_zero_eval_partial.py      ← partial 合并脚本
└── 03_results/                           ← 完整结果（partial）
    ├── FOUR_CELL_OFFICIAL_SUMMARY.json   ← 顶层汇总（partial=True, n=827）
    ├── ADAPTER_RELOAD_AUDIT.json
    ├── LAUNCH.json                       ← 启动配置（22:47）
    ├── PRE_KILL_SNAPSHOT.json            ← 停止前 64 shard 状态快照
    ├── upstream_api/
    │   ├── PER_QUERY.jsonl               ← 827 条/query 轨迹（合并后）
    │   ├── SUMMARY.json                  ← summarize_api_traces 聚合指标
    │   ├── DONE.json                     ← ok=False, partial=True, n_missing=3
    │   └── SHARD_PLAN.json               ← 64 shard 分配计划
    └── shards_sample/                    ← 3 个代表性 shard 的完整产物
        ├── rank0/   (DONE.ok=true, 正常完成 13/13)
        ├── rank24/  (hung, 完成 10/13, 无 DONE.json — partial 来源)
        └── rank32/  (DONE.ok=true, 曾 hung 于 qid=822 turn8 后恢复)
```

> **注**: 完整 64 shard 原始产物（含各 shard 40MB TURNS.jsonl）未全部复制，仍在原始输出目录：
> `/mnt/songzijun/Capability_Evolution/TRIM/outputs/eval_h1_bcplus_full_Qwen3-4B-Instruct-2507_zero_gpu7_20260911_224454_bcplus_full_qwen3_4b_zero_8gpu/`
> 本归档的 `shards_sample/` 含 3 个代表性 shard（rank0 正常 / rank24 hung / rank32 恢复）用于问题分析。

## 3. 运行指令（完整复现步骤）

### 3.1 前置条件
- CloudML context `songzijun-cto`，workspace 10063
- 队列 2334（记忆对话，青海 kscn-hd-cloudml-prod），gm402-80g，申请 8 卡
- 镜像 `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`（缺的 python 包在 entry 内安装）
- JuiceFS 卷 `songzijun`（集群 bj9-cloudml）挂载到 `/mnt/songzijun`（模型/代码/索引/语料/tiktoken bundle 都在此盘）

### 3.2 提交命令
```bash
cml custom_train submit \
  --filename /mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_qwen3_4b_zero_8gpu_20260911_not_submitted.yaml \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --tag "MemQA-General-V5"
```
- YAML 中**不含** JuiceFS 挂载，故意用 CLI `--juicefs_mount_configs` 传 JSON，使 `endpoint=bj9-cloudml` 显式进入 API 请求（endpoint 字段被 Go 后端忽略，实际生效靠 `juiceFsCluster=bj9-cloudml`，双写最稳）
- priority=8，tag=MemQA-General-V5，GUARANTEED_PUBLIC 非抢占，单节点 8 卡 `cloudml.ng1gm402-8-8.20-224`

### 3.3 entry 命令关键阶段（imageCommand）
1. **tiktoken offline**：解压 `/mnt/songzijun/tiktoken-bundle.zip`→`/tmp/tiktoken-bundle`，设 `TIKTOKEN_ENCODINGS_BASE/RS_CACHE_DIR/CACHE_DIR`（含 o200k_base+cl100k_base）
2. **JVM**：`find /usr /opt -name libjvm.so`→镜像自带 `/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so`，`test -f`+`ldd`+`autoclass("java.lang.String")` preflight，`JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`
3. **四包 + import-chain deps**：`pip install --ignore-installed torch==2.10.0 vllm==0.19.1 transformers==4.57.6 pyserini==2.0.0 structlog chromadb ...`；再 `pip install --no-deps json_repair langchain-text-splitters ... chz tinker`（`--no-deps` 防 tinker 的 unconstrained transformers 升级到 5.x 破坏四包锁版）
4. **ANSERINI_CLASSPATH**：指向 `<pyserini包>/resources/jars/anserini-2.0.0-fatjar.jar`（不是包根）
5. **preflight 自愈断言**：四版本 + 8 GPU + PyJNI + Lucene index 打开（num_docs=100195）+ **`load_upstream_modules()` 18 模块全 import**（在 64 worker spawn 前抓 chz/tinker/json_repair 缺失）
6. **source wrapper（cp 到 /tmp 后）**：绕过 JuiceFS heredoc 回读抖动；source 后**重设 TRIM_ROOT + REL_* + PYTHONPATH**（source /tmp 副本会误算 TRIM_ROOT=/）
7. **只调 zero**：`start_zero_stack` + `run_eval(zero)`，不进入 main() 的 all 分支

### 3.4 监控
```bash
bash /mnt/songzijun/Capability_Evolution/TRIM/monitor_qwen3_4b_zero_eval_20260911.sh
```
统计 distinct qids started/done（字段名 `query_id`，非 `qid`）、runTime、worker.log 活跃度、报错扫描。

## 4. 完整结果（partial，827/830 query）

### 4.1 指标（summarize_api_traces，827 query）
| 指标 | 值 |
|---|---|
| n_queries | 827 / 830（99.6%） |
| Curated Recall | 0.0708 |
| Pool Recall | 0.1542 |
| Final-Answer Recall | 0.0855 |
| Precision | 0.0161 |
| F1 | 0.0246 |
| Tool Calls / query | 27.48 |
| partial | **True** |
| n_missing | 3（qid 1004, 289, 790，均来自 hung 的 rank24） |

### 4.2 定级
**stopped partial**——63/64 shard `DONE.ok=true`，但 rank24 在 qid=1004 turn=4 `grep_corpus` 静默 hang（无超时），阻塞主进程 `proc.wait()`，官方合并步骤未执行；后从 63 shard 的 PER_QUERY.jsonl 用官方函数手动合并。**partial=True, ok=False，不得作 formal artifact**。

详见 `02_ISSUES_AND_RESOLUTIONS.md`（grep_corpus hang 根因 + partial 合并方法 + launcher 修复方向）。
