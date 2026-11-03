# 运行手册（pilot / formal）

## 当前可运行范围

**已端到端验证（toy，synthetic）**：独立安装、H0 循环、空 bank、代码演化物化
与执行、Teacher fork 隔离、HarnessOPD 联合更新（真实 tiny 模型梯度）、两轮
campaign 闭环+晋升+resume。全部见 `tests/`（31 passed）。

**待真实资源（pilot/formal 阻塞清单）**：

| 阻塞 | 需要 | 就绪命令 |
|---|---|---|
| AppWorld 环境 | `pip install -e '.[appworld]'` + 官方数据（固定 revision） | `rsi.cli doctor` 确认 `appworld deps_ok` |
| SpreadsheetBench | `pip install -e '.[spreadsheetbench]'` + 冻结任务列表 | 同上 |
| SWE-bench | `pip install -e '.[swebench]'` + docker + 实例镜像 digest | 同上 |
| WebShop | `pip install -e '.[webshop]'` + `scripts/prepare_webshop_data.py`（items 数据 + Lucene index + frozen_goals）；Google Drive 与集群出口均不可达，官方文件取自 ModelScope 公共镜像 `brihug/WebShop` + `Yimingastro/WebShop`（digests 入 frozen_manifest） | 同上 + `frozen_manifest.json` 存在 |
| ALFWorld | `pip install -e '.[alfworld]'` + `scripts/prepare_alfworld_data.py --skip-download`（官方 json_2.1.1 树 = train/valid_seen/valid_unseen，**官方发布无 test_unseen/OOD**（alfworld/data/README.md）；本环境数据取自 ModelScope 镜像 `Ecollzhang/alfworld-json-2.1.1`（git+LFS），官方 zip 源保留脚本内） | 同上 + `frozen_manifest.json` 存在 |
| 真实可训练开源模型 | 本地模型路径注入 `policy.model_path` | `rsi.cli doctor` 检查 policy |
| GPU 多卡（torch_fsdp2） | torchrun 进程组 | `rsi.cli train --config configs/pilot_multidomain.yaml` |

真实环境适配器（`src/rsi/environments/{appworld_env,spreadsheetbench_env,swebench_env,webshop_env,alfworld_env}.py`）
目前实现协议外壳 + 依赖门控：缺依赖时构造即抛 `NotImplementedCapability`，
`doctor` 明确列出缺失项，**绝不 fallback 到 Search/toy**。安装依赖后需要按
`environments.yaml` 的 `dataset_revision` 完成 `data prepare` 的冻结接线。

WebShop / ALFWorld 口径（受控测评）：WebShop `final_test` = 官方 500-goal human
测试面板（frozen_goals 前 500 个官方 session 索引），指标 = 官方 `get_reward`
[0,1] + harsh success（reward==1）；gold attributes 只在 scorer 侧。ALFWorld
`final_test` = 官方 `test_unseen`、`transfer_test` = `out_of_distribution`，指标 =
引擎 `won` 的 task success；Actor 只产文本命令轨迹，scorer 独立重放读 `won`。
两者都用 replay-prefix snapshot/fork（冻结数据 + 确定性引擎），group 分别按
asin / task-instance 严格 split 内聚 + 面板反泄漏去重。

⚠️ **WebShop flavor=all 内存门槛**：官方 `load_products` 对 items_shuffle.json
(5.5GB) 的 `json.load` 峰值 >16GB，开发机 user-container memcg ~16GB 会 oom-kill
（dmesg 实测）；全量索引构建/目标冻结/正式 episode 一律在大内存 CloudML 节点跑
（`submit_prepare_webshop_full_20261102.sh`，224GB）。flavor=small（1000 商品，
13 goals）仅够链路自检，不满足官方 500-goal 面板（provider 会 loud refuse）。

## pilot 运行顺序

```bash
cd Capability_Evolution/RSI
python -m pip install -e '.[dev]'
# 1. 注入真实模型路径（不要用研究挂载点硬编码）
$EDITOR configs/pilot_multidomain.yaml   # policy.model_path
python -m rsi.cli doctor --config configs/pilot_multidomain.yaml
# 2. 冻结数据划分（按 scenario/workbook/issue family 聚类）
python -m rsi.cli data prepare --config configs/pilot_multidomain.yaml
python -m rsi.cli data validate --config configs/pilot_multidomain.yaml
# 3. 基线
python -m rsi.cli eval --config configs/pilot_multidomain.yaml --harness minimal --split harness_select
# 4. 全闭环
python -m rsi.cli campaign run --config configs/pilot_multidomain.yaml
python -m rsi.cli campaign report --run-dir runs/pilot_multidomain
```

## formal 前置

1. `configs/formal_multidomain.yaml` 必须填满冻结阈值/规模/统计方案
   （loader 拒绝 `pilot_reused_dev`、空 `frozen_thresholds`、缺 promotion
   policy file 的占位配置）。
2. 预注册：rounds 数、reserve 大小、最大尝试次数、alpha spending。
3. `final_test` 在 campaign freeze 后一次性解封（`campaign/splits.py` 强制）。

## 四格与晋升解读

`A=(θ_k,H0)` `B=(θ_k,H0⊕B_k)` `C=(θ_{k+1},H0)` `D=(θ_{k+1},H0⊕B_k)`；
ΔH=B−A、ΔW=C−A、retention=(C−A)/(B−A)（分母不可靠为正时报告 None，不裁剪）。
部分内化可晋升（`partial_internalization` 标记）；"全部恢复 Teacher 增益" 需
C 对 B 的非劣证据；`surpassed_teacher` 需正差证据。成本降低不预设。

## 实验矩阵（任务书 §16）

Base / Evolve-only / RL-only / HarnessOPD 单轮 / 多轮 / 固定首轮 bank 多轮 /
Aligned Teacher SFT —— 全部经同一 CLI + 配置开关（`training.method`、
`evolution.seed_mode`）实现；结果措辞边界见 README「证据层级」。

## 提交纪律：先 preflight 后大任务（2026-10-29 教训）

历史失败模式：32 卡任务排队 1-2 小时 → 起跑几分钟即挂 → 修 → 再排队。根因几乎都是
**本地 CPU 测试与 GPU 任务配置的分歧**（dtype/device/真实模型结构/分布式）。

**规矩**：
1. 改过训练/推理路径后，先跑本地 `tests/integration/test_gpu_config_parity.py`
   （bf16+LoRA+真实序列，CPU 可跑）与 `test_distributed_sync.py`（2-proc gloo）。
2. 涉及真实模型/多节点改动时，先提 **1 卡 preflight**：
   `bash scripts/submit_preflight_train_1gpu_20261029.sh`（~15 分钟，
   逐阶段 PASS/FAIL 覆盖 generate/collect/joint step/export），过了再提大任务。
3. 大任务失败时先看 `cml custom_train history_logs --pod_name <master> --keyword Traceback`
   定位（NCCL INFO 会淹没日志，用 keyword 过滤）。

已知分歧清单（本地测试必须覆盖的 GPU 形态）：
- 模型 dtype=bfloat16（CPU 测试默认 fp32）— LoRA matmul 曾因此崩
- LoRA 参数须建在 base device 上、forward 里 cast（fp32 A/B + bf16 base）
- 分布式集合通信必须全 rank 同构调用（空 batch 零填充梯度，否则挂死）
- 全局 batch 必须 = world_size × per_rank（否则空转 rank）
