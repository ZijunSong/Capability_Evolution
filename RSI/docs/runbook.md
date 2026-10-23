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
| 真实可训练开源模型 | 本地模型路径注入 `policy.model_path` | `rsi.cli doctor` 检查 policy |
| GPU 多卡（torch_fsdp2） | torchrun 进程组 | `rsi.cli train --config configs/pilot_multidomain.yaml` |

真实环境适配器（`src/rsi/environments/{appworld_env,spreadsheetbench_env,swebench_env}.py`）
目前实现协议外壳 + 依赖门控：缺依赖时构造即抛 `NotImplementedCapability`，
`doctor` 明确列出缺失项，**绝不 fallback 到 Search/toy**。安装依赖后需要按
`environments.yaml` 的 `dataset_revision` 完成 `data prepare` 的冻结接线。

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
