# RSI: Harness 自进化与 HarnessOPD 内化（独立实现）

本项目实现任务书
[`RSI_Generic_Harness_Implementation_Guide.md`](../../RSI_Generic_Harness_Implementation_Guide.md)
所定义的闭环：

```
(theta_k, H0) --自动生成并评估 Harness 代码--> (theta_k, H0⊕B_k)
             --HarnessOPD + outcome RL（Student 仅运行 H0）--> (theta_{k+1}, H0)
             --重新演化--> ...
```

- **独立**：不导入、不调用、不加载 TRIM/SCAPE/SCOPE/Harness-1（见
  `docs/legacy_audit.md` 与 `tests/test_standalone_install.py`）。
- **真正代码演化**：模型生成组件 Python 代码/提示/记忆策略/控制流，
  materializer 原样落盘执行（`tests/unit/test_bank_and_materialization.py`）。
- **通用**：核心不含任何检索/SEC/BrowseComp 语义；环境经
  `EnvironmentAdapter` 协议接入（toy_file / toy_kv / appworld /
  spreadsheetbench / swebench）。
- **原方法语义**：HarnessOPD = Student on-policy + 隔离 Teacher fork +
  显式 recipe 映射 / skip-to-anchor + 三层可实现性检查 + gated
  projected-gap + outcome CISPO 联合更新（`src/rsi/opd/`、
  `tests/unit/test_loss_and_grad_parity.py`）。
- **闭环真实性**：`campaign run` 真实执行演化→确认→probe→训练→四格→晋升；
  toy 两轮闭环见 `tests/e2e/test_two_round_campaign.py`（synthetic 标记，
  仅状态机验收，不是性能结论）。

## 安装

```bash
cd Capability_Evolution/RSI
python -m pip install -e '.[dev]'      # 开发/测试（含 torch+transformers）
python -m pip install -e '.[train]'    # 只要训练后端
python -m pip install -e '.[appworld]' # 可选真实环境（同理 spreadsheetbench / swebench）
```

## CLI（clean install 真实可运行）

```bash
python -m rsi.cli doctor --config configs/toy_smoke.yaml
python -m rsi.cli data prepare --config configs/toy_smoke.yaml
python -m rsi.cli data validate --config configs/toy_smoke.yaml
python -m rsi.cli eval --config configs/toy_smoke.yaml --harness minimal --split harness_select
python -m rsi.cli evolve --config configs/toy_smoke.yaml
python -m rsi.cli probe-opd --config configs/toy_smoke.yaml --bank runs/<id>/evolve/champion.json
python -m rsi.cli train --config configs/toy_smoke.yaml --bank runs/<id>/evolve/champion.json
python -m rsi.cli campaign run --config configs/toy_smoke.yaml
python -m rsi.cli campaign resume --run-dir runs/<id>
python -m rsi.cli campaign report --run-dir runs/<id>
python -m rsi.cli campaign plan --config configs/toy_smoke.yaml   # executed=false
```

`--harness minimal` = 空高级组件库；`--harness bank --bank <ref>` 只加载指定
bank。`campaign plan` 产物永远带 `executed=false`；`campaign run` 在缺能力时
非零退出，不返回"成功"的 plan 报告。

## 配置

| 文件 | 用途 |
|---|---|
| `configs/toy_smoke.yaml` | toy 协议冒烟；不调真实 benchmark，输出全 synthetic |
| `configs/pilot_multidomain.yaml` | 真实模型+真实环境 pilot；pilot_reused_dev 统计 |
| `configs/formal_multidomain.yaml` | 冻结阈值/规模/统计方案；loader 拒绝占位值 |

三套均保持 **一个共享模型 + 一个共享 bank** 用于所有领域
（`suite.shared_model/shared_bank`）。

## 测试

```bash
python -m pytest tests/ -q        # 31 tests：协议/损失对照/隔离/训练/闭环
python -m pytest tests/e2e -q     # toy 两轮闭环（synthetic，仅状态机验收）
```

重要验收测试名（任务书 §15）：
`test_standalone_install`、`test_no_legacy_runtime_dependency`、
`test_empty_bank_identity`、`test_real_code_materialization`、
`test_parent_feedback_continuity`、`test_prune_semantics`、
`test_cross_environment_core`、`test_teacher_isolation`、
`test_future_observation_rejected`、`test_reasoning_not_misclassified_as_leak`、
`test_native_anchor_without_explicit_map`、`test_hidden_grader_inaccessible`、
`test_exact_prompt_ids`、`test_loss_and_grad_parity`、`test_joint_step_count`、
`test_opd_affects_update`、`test_distributed_normalization`、
`test_export_reload_equivalence`、`test_round_handoff`、
`test_resume_idempotence`、`test_split_group_disjointness`、
`test_unimplemented_fails`。

## 证据层级

机器可读报告区分 `implementation_smoke_passed`、`real_training_executed`、
`n_successful_promotions`、`shared_multidomain_evaluated`、
`final_test_unsealed`、`stopped_reason`。toy/mock 输出带 `synthetic=true`，
报告合并器拒绝将其合入正式结果（`formal_eligible=false`）。

**测试通过 ≠ benchmark 收益**。性能结论需要真实模型、真实环境与冻结配置，
见 `docs/runbook.md`。

## 文档

- `docs/legacy_audit.md` — 旧代码核查与迁移映射（含不迁移项）
- `docs/method_migration.md` — HarnessOPD/损失/对齐语义迁移对照
- `docs/runbook.md` — pilot/formal 运行手册、阻塞项清单
