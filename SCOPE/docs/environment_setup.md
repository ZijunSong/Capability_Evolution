# BiSHOP 环境配置说明

本文档说明在本仓库运行 **BrowseComp+ Bare Rollout**、**Harness Rollout（BM25 本地检索）** 以及 **OPD 烟测** 所需的完整环境配置。

适用路径：仓库根目录 `/data/ppnm/BiSHOP`（下文记为 `$REPO`）。

---

## 1. 架构概览

BiSHOP 在 Harness-1 基础上扩展了双向 OPD 实验管线，当前 BrowseComp+ 实验分为两层：

| 模式 | 说明 | 检索 | 典型用途 |
|------|------|------|----------|
| **Bare Rollout** | 单轮生成 \(\tau^{\mathrm{bare}} \sim \pi_\theta(x)\)，无 Harness | 无 | OPD baseline、快速全量采样 |
| **Harness Rollout** | 多轮 search agent + Harness 模块（V8D） | BM25（推荐）或 Chroma | 带检索的 agent 轨迹、模块消融 |

推理与训练拆分（OPHSD/veRL 风格）：

- **Rollout**：`vLLM` 提供 OpenAI 兼容 API（`/v1/completions` 或 chat）
- **Training**：HuggingFace / FSDP（`training/train_opd.py`）

---

## 2. 硬件与系统要求

| 组件 | Bare Rollout | Harness Rollout |
|------|--------------|-----------------|
| GPU | 4× GPU（默认 TP=4），如 GPU 4–7 | 同左 |
| 显存 | 7B 模型约需每卡 15–20GB（`max_model_len=8192`） | 建议 `max_model_len=32768`，显存更高 |
| CPU / 内存 | 常规 | BM25 索引加载需额外磁盘与内存 |
| 磁盘 | 模型 ~15GB + 数据 ~百 MB | + BM25 索引（数 GB，视 HF 包大小） |
| OS | Linux | Linux |
| Java | **不需要** | **需要 JDK 21+**（Pyserini/Lucene） |

---

## 3. Conda 环境

### 3.1 创建 / 更新环境

```bash
cd $REPO
bash scripts/setup_conda_env.sh
```

或手动：

```bash
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
cd /data/ppnm/BiSHOP
export PYTHONPATH=.
```

快捷激活：

```bash
source scripts/activate_bishop.sh
```

| 项目 | 值 |
|------|-----|
| 环境名 | `bishop`（可通过 `BISHOP_CONDA_ENV` 覆盖） |
| Python | 3.11 |
| Conda 根目录 | `/data/ppnm/miniconda3`（可通过 `CONDA_BASE` 覆盖） |

### 3.2 Python 依赖

核心（`setup_conda_env.sh` 已安装）：

- `harness-1` / BiSHOP（editable）
- `tinker-cookbook`（editable）
- `vllm>=0.13.0`
- `chromadb`, `openai-harmony`, `transformers`, `tiktoken`, 等

**Harness BM25 检索额外依赖**（生产路径必需）：

```bash
conda activate bishop
pip install 'pyserini>=1.2.0'
conda install -c conda-forge openjdk=21   # 提供 javac
```

或使用可选 extra（`pyproject.toml`）：

```bash
pip install -e '.[bm25,vllm]'
```

### 3.3 vLLM 运行时环境变量

Bare / Harness rollout 脚本默认设置：

```bash
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

vLLM 启动时还使用 `--enforce-eager` 与 `--disable-custom-all-reduce`（见 `training/opd/vllm_server.py`），以避免部分主机上的兼容性问题。

---

## 4. 凭证与 API Key 说明

### 4.1 配置文件

**推荐：仅三项 LLM 配置（OpenAI 兼容 API，如 MIFY / Kimi）**

在 `BiSHOP/.env` 中填写：

```bash
base_url=https://api.llm.mioffice.cn/v1
api_key=sk-...
model_name=moonshot/kimi-k2.6
```

配置后，Harness 相关组件会自动通过 `get_openai_client()` / `get_llm_client()` 调用该模型，包括：

- Bare / Harness rollout（`--policy auto` 时优先走 API）
- OPD shadow harness 的 verification
- 仍使用 Chroma 检索时需额外配置 `CHROMA_*`（BM25 模式不需要）

BrowseComp+ 数据路径写在 `.env.browsecomp.paths`（由 `scripts/setup_browsecomp_data.sh` 生成，会自动加载）。

也可使用 `.env.local` 覆盖 `.env`：

```bash
cp .env.example .env.local   # 可选
# 或编辑 .env
```

### 4.2 各实验路径需要什么？

| 凭证 / 配置 | Bare Rollout | Harness + **BM25** | Harness + **Chroma** |
|-------------|:------------:|:--------------------:|:--------------------:|
| `base_url` / `api_key` / `model_name` | 可选（替代 vLLM） | 可选（`--policy api`） | 可选 |
| `CHROMA_API_KEY` | 否 | **否** | **是** |
| `CHROMA_DATABASE` | 否 | **否** | **是** |
| `OPENAI_API_KEY`（embedding） | 否 | **否** | **是** |
| `BASETEN_*`（reranker） | 否 | 仅 `RERANKER=baseten` 时 | 同左 |
| 本地 vLLM / 模型权重 | 无 API 时**是** | 无 API 时**是** | 无 API 时**是** |
| BrowseComp+ 数据文件 | **是** | **是** | **是** |
| BM25 Lucene 索引 + Java | 否 | **是** | 否 |
| `HUGGINGFACE_TOKEN` | 下载模型/数据时 | 同左 | 同左 |

**结论**：使用默认的 `RETRIEVAL=bm25` 时，**不需要 Chroma Cloud 或 OpenAI embedding API**；只需本地 BM25 索引、Java、以及用于生成的模型（vLLM）。

### 4.3 其他可选变量

| 变量 | 用途 |
|------|------|
| `base_url` / `api_key` / `model_name` | **主 LLM API**（OpenAI 兼容）；配置后优先于下方 `OPENAI_API_KEY` |
| `OPENAI_API_KEY` | 未配置三项 LLM 时的 fallback；Chroma dense embedding |
| `CHROMA_API_KEY` / `CHROMA_DATABASE` | Chroma Cloud 文档检索 |
| `ANTHROPIC_API_KEY` | 部分 baseline agent |
| `TINKER_API_KEY` | Tinker 托管训练/推理（legacy 路径） |
| `HUGGINGFACE_TOKEN` | 私有 HF 模型/数据集 |
| `MOONSHOT_API_KEY` | Moonshot API agent |
| `BASETEN_API_KEY` / `BASETEN_MODEL_URL` | 可选 reranker |
| `JINA_API_KEY` / `CONTEXTUAL_API_KEY` | 其他检索组件 |
| `BROWSECOMPPLUS_*_PATH` | BrowseComp+ 本地 query/qrel/answer 路径 |

---

## 5. BrowseComp+ 数据准备

### 5.1 一键下载解密

```bash
cd $REPO
bash scripts/setup_browsecomp_data.sh
```

完成后应有（**830 条** query）：

```text
external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
external/BrowseComp-Plus/topics-qrels/queries.tsv
external/BrowseComp-Plus/topics-qrels/qrel_golds.txt
external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt
```

并生成 `$REPO/.env.browsecomp.paths`。

### 5.2 网络与镜像

脚本默认 `HF_ENDPOINT=https://hf-mirror.com`。若直连 HuggingFace 更稳定，可：

```bash
export HF_ENDPOINT=https://huggingface.co
```

代理示例（本机 Clash）：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

### 5.3 手动指定路径

```bash
export BROWSECOMPPLUS_ANSWERS_PATH=/path/to/browsecomp_plus_decrypted.jsonl
export BROWSECOMPPLUS_QUERIES_PATH=/path/to/queries.tsv
export BROWSECOMPPLUS_QRELS_GOLD_PATH=/path/to/qrel_golds.txt
export BROWSECOMPPLUS_QRELS_EVIDENCE_PATH=/path/to/qrel_evidence.txt
```

---

## 6. BM25 本地检索索引（Harness 推荐）

Harness rollout 默认 `RETRIEVAL=bm25`，使用 BrowseComp+ 官方 Pyserini/Lucene 索引，**无需 Chroma**。

### 6.1 安装 Java 与 Pyserini

```bash
conda activate bishop
conda install -c conda-forge openjdk=21
pip install 'pyserini>=1.2.0'
java -version    # 应显示 21+
javac -version
```

### 6.2 下载索引

```bash
cd $REPO
bash scripts/setup_browsecomp_bm25_index.sh
```

默认输出目录：

```text
external/BrowseComp-Plus/indexes/bm25/
```

目录内应包含 Lucene 段文件（如 `segments_1`，**不是** `segments_*.lock`）。

自定义路径：

```bash
export BROWSECOMP_BM25_INDEX_PATH=/your/path/to/lucene/index
```

HF 数据集：`Tevatron/browsecomp-plus-indexes`（`bm25/*`）。

### 6.3 验证索引

```bash
cd $REPO
export PYTHONPATH=.
python -c "
from harness.retrieval.bm25_backend import resolve_bm25_index_path, BrowseCompBm25Backend
p = resolve_bm25_index_path()
print('index:', p)
b = BrowseCompBm25Backend(p)
print('hits:', len(b.search('Einstein Nobel Prize', k=3)))
"
```

---

## 7. 模型配置

### 7.1 当前实验默认模型

| 用途 | 默认路径 |
|------|----------|
| Bare / Harness（本机 vLLM） | `/data/ppnm/models/Qwen2.5-7B-Instruct` |

启动时通过环境变量覆盖：

```bash
export MODEL_PATH=/data/ppnm/models/Qwen2.5-7B-Instruct
```

### 7.2 Harmony 与 Qwen 的重要说明

`SlidingWindowSearchEnv`（Harness 多轮环境）使用 **Harmony token** 渲染与 `/v1/completions` 原始 token id 接口。

| 模型 | Bare Rollout | Harness 多轮 Rollout |
|------|:------------:|:-------------------:|
| Qwen2.5-7B-Instruct（chat） | 可用 | 工具调用格式**不兼容**，仅适合管线调试 |
| `pat-jj/harness-1`（Harmony） | 不适用 | **推荐**，与 Harness-1 评测一致 |

正式 Harness 实验建议使用 Harness-1 或 Harmony 兼容 checkpoint，详见 `docs/run_vllm_browsecompplus.md`。

---

## 8. 离线烟测（无需任何 API Key）

在配置生产环境前，可先验证 BM25 + Harness 工具链：

```bash
cd $REPO
source scripts/activate_bishop.sh
bash scripts/smoke_harness_bm25.sh
```

包含：

1. `pytest tests/test_harness_bm25_smoke.py tests/test_bm25_retrieval.py`
2. `python training/smoke_harness_bm25.py`（内存语料，无 Java/索引）
3. `rollout_harness_browsecomp.py --smoke-retrieval --tools-only`（只测工具链）

单独运行内存烟测：

```bash
python training/smoke_harness_bm25.py
```

---

## 9. 运行实验

### 9.1 Bare Rollout（无 Harness，830 条全量）

```bash
cd $REPO
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/rollout_bare_browsecomp_4gpu.sh
```

后台：

```bash
bash scripts/nohup_rollout_bare_browsecomp.sh
```

| 参数 | 默认值 |
|------|--------|
| `SPLIT` | `all`（830 条） |
| `MAX_NEW_TOKENS` | `2048` |
| `TEMPERATURE` | `1.0` |
| `MAX_MODEL_LEN` | `8192` |
| `VLLM_PORT` | `8770` |
| 输出 | `outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl` |

### 9.2 Harness Rollout（BM25 + 全模块）

**前置**：BrowseComp+ 数据 + BM25 索引 + Java + 模型权重。

```bash
cd $REPO
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/rollout_harness_browsecomp_4gpu.sh
```

后台：

```bash
bash scripts/nohup_rollout_harness_browsecomp.sh
```

| 参数 | 默认值 |
|------|--------|
| `RETRIEVAL` | `bm25` |
| `HARNESS_CONFIG` | `harness/configs/modules_full.yaml` |
| `MAX_TURNS` | `35` |
| `MAX_TOKENS` | `2048`（每轮） |
| `MAX_MODEL_LEN` | `32768` |
| `PARALLEL` | `2`（并发 episode 数） |
| `RERANKER` | `none` |
| `VLLM_PORT` | `8771` |
| 输出 | `outputs/harness_rollout_browsecomp_full/harness_rollouts.jsonl` |

烟测 3 条：

```bash
LIMIT=3 bash scripts/rollout_harness_browsecomp_4gpu.sh
```

仅测检索工具（不启 vLLM）：

```bash
python training/rollout_harness_browsecomp.py \
  --smoke-retrieval --tools-only \
  --queries-json tests/fixtures/browsecomp_sample_queries.json \
  --limit 1 --no-manage-vllm
```

### 9.3 切换回 Chroma Cloud 检索（可选）

```bash
# .env.local 中配置真实 CHROMA_* 与 OPENAI_API_KEY
RETRIEVAL=chroma bash scripts/rollout_harness_browsecomp_4gpu.sh
```

---

## 10. Harness 模块环境变量（V8D）

Harness rollout 脚本默认开启 full operating point：

```bash
V8D_SUBTRACTIVE_CURATION=1
V8D_IMPORTANCE_TAGGING=1
V8D_AUTO_POPULATE_FIRST_SEARCH=1
V8D_EVIDENCE_GRAPH=1
V8D_SENTENCE_COMPRESS=1
V8D_CONTENT_DEDUP=1
V8D_VERIFY_TOOL=1
V8D_TOKEN_BUDGET_MARKER=1
SAVE_TRAJECTORIES=1
```

模块化 YAML 配置见 `harness/configs/`（如 `modules_full.yaml`、`ablate_verification.yaml`）。

---

## 11. 输出目录结构

```text
outputs/
├── bare_rollout_browsecomp_full/
│   ├── bare_rollouts.jsonl          # 轨迹（可断点续跑）
│   ├── bare_rollout_manifest.json
│   ├── vllm_server.log
│   └── nohup_rollout.log
└── harness_rollout_browsecomp_full/
    ├── harness_rollouts.jsonl
    ├── harness_rollout_manifest.json
    ├── harness_resolved_config.yaml
    └── vllm_server.log
```

监控进度：

```bash
wc -l outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl
wc -l outputs/harness_rollout_browsecomp_full/harness_rollouts.jsonl
tail -f outputs/harness_rollout_browsecomp_full/nohup_rollout.log
```

---

## 12. 常见问题

### Q1：`CHROMA_API_KEY is not configured`

你正在走 Chroma 路径。改用 BM25：

```bash
RETRIEVAL=bm25 bash scripts/rollout_harness_browsecomp_4gpu.sh
```

### Q2：`Unable to find javac` / Pyserini 报错

```bash
conda install -c conda-forge openjdk=21
export JAVA_HOME="$CONDA_PREFIX"
```

### Q3：`BM25 Lucene index not found`

```bash
bash scripts/setup_browsecomp_bm25_index.sh
# 确认目录内有 segments_* 文件（非 .lock）
ls external/BrowseComp-Plus/indexes/bm25/
```

### Q4：vLLM 启动失败

- 检查 `CUDA_VISIBLE_DEVICES` 与 `--tensor-parallel-size` 一致
- 查看 `outputs/*/vllm_server.log`
- 尝试降低 `MAX_MODEL_LEN`

### Q5：HuggingFace 下载超时

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

### Q6：Bare 准确率低是否正常？

正常。Bare 无检索，BrowseComp+ 难题上 7B 模型准确率接近 0–5% 是预期 baseline；应用 Harness rollout 对比 `recall` / `final_answer_recall`。

---

## 13. 快速检查清单

**最小 Bare 实验：**

- [ ] `conda activate bishop`，`export PYTHONPATH=.`
- [ ] BrowseComp+ 830 条数据就绪
- [ ] 本地模型路径可访问
- [ ] GPU 4–7 空闲

**最小 Harness（BM25）实验：**

- [ ] 上述全部
- [ ] `openjdk=21` + `pyserini` 已安装
- [ ] BM25 索引已下载且 `resolve_bm25_index_path()` 成功
- [ ] `bash scripts/smoke_harness_bm25.sh` 通过
- [ ] （推荐）Harmony 兼容模型用于正式多轮 rollout

---

## 14. 相关文档与脚本索引

| 文件 | 说明 |
|------|------|
| `scripts/setup_conda_env.sh` | 创建 conda 环境 |
| `scripts/setup_browsecomp_data.sh` | 下载 BrowseComp+ query/answer |
| `scripts/setup_browsecomp_bm25_index.sh` | 下载 BM25 索引 |
| `scripts/smoke_harness_bm25.sh` | 离线烟测（无 API key） |
| `scripts/rollout_bare_browsecomp_4gpu.sh` | Bare 全量 rollout |
| `scripts/rollout_harness_browsecomp_4gpu.sh` | Harness 全量 rollout |
| `docs/run_vllm_browsecompplus.md` | Harness-1 + vLLM 官方指南 |
| `harness1_bidirectional_opd_todo.md` | OPD 共演化设计文档 |
| `training/rollout_bare_browsecomp.py` | Bare Python 入口 |
| `training/rollout_harness_browsecomp.py` | Harness Python 入口 |
| `harness/retrieval/` | BM25 后端与工具实现 |

---

*最后更新：2026-07-24（含 BM25 本地检索与离线烟测路径）*
