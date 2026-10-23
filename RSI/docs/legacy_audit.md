# 旧代码审计与迁移记录

审计基线：`ZijunSong/Capability_Evolution` 分支 `scope/round14-capability-portfolio`，
HEAD `442da0d4c90039e6dd3857222672925c1d053a85`（与任务书 §19.1 快照一致，实现前已复核）。

本项目（`Capability_Evolution/RSI/`）**不依赖**旧研究运行时。允许读取旧实现以
迁移纯算法；禁止把旧运行时变成隐式依赖（由 `tests/test_standalone_install.py`
与 `tests/test_no_legacy_runtime_dependency.py` 强制）。

## 任务书 §1 偏差核查（复核结论）

| 旧位置 | 核查事实 | 本项目处理 |
|---|---|---|
| `TRIM/README.md` | Harness-1/Harness-G/BrowseComp+ 训练评测线，跨目录 PYTHONPATH | 不继承该运行时 |
| `TRIM/configs/rsi/harness1_sec.yaml` | RSI 配置是旧 Search 场景外层封装 | 新配置 `configs/*`，独立任务注册表（`environments.yaml`） |
| `TRIM/configs/rsi/base_contract.json` | 工具固定为检索族，可见字段含 pool/curated | 新 Action/Observation/Environment 契约（`src/rsi/contracts/`） |
| `TRIM/trim/evolution/self_proposer.py::propose` | 固定编辑 retrieval component、返回 guidance JSON | EvolutionPacket 含真实父代源码/配对结果/失败片段；CandidatePatch 落盘真实代码（`src/rsi/evolution/`） |
| `TRIM/trim/evolution/materialize.py::write_candidate_bank` | 固定复制 `stdlib/query_repair.py`，组件 ID 固定 | materializer 原样落盘 `files[].content`（test_real_code_materialization） |
| `TRIM/scripts/run_harness_evolve.py` | 写死 Harness-1/bcplus_full，止于搜索 | `rsi.cli evolve` 走独立服务；campaign 接通训练/晋升 |
| `TRIM/trim/evolution/campaign.py::distillability_probe` | 构造状态+structural_trigger | probe 只读真实 rollout/fork 统计（`src/rsi/opd/probe.py`） |
| `TRIM/trim/evolution/search.py::select_champion` | 选择池要求全 distillable | runtime_archive 与 distillation_archive 分离；skip-to-anchor 监督计入覆盖 |
| `TRIM/scripts/run_rsi.py` | `AWAITING_MANUAL_LAUNCH`/`RUN_NOT_WIRED` | `campaign run` 真实端到端（tests/e2e） |
| `TRIM/trim/harness_bank/resolution.py` | bank 从 zero mask + legacy_flags | 新 bank 无 legacy_flags（`src/rsi/harness/`） |
| 同文件 `apply_bank_launch` | 拒绝不消费 OPD 字段的后端 | 独立后端真实消费 OPD（joint_update 强制 consumes_opd；test_unimplemented_fails） |

## 任务书 §14 迁移映射（实际执行）

| 旧位置 | 迁移内容 | 新位置 | 重写边界 |
|---|---|---|---|
| `trim/rsi/controller.py` | 状态机/父 policy 检查/原子写 | `campaign/controller.py` + `campaign/registry.py`（CAS 晋升） | 完整服务 wiring、独立 CLI、跨领域 panel |
| `trim/rsi/splits.py` 等 | split 身份/确认消费/配对统计概念 | `campaign/splits.py`、`evaluation/{paired,statistics}.py` | group 语义、领域聚合、pilot_reused_dev 策略 |
| `trim/evolution/search.py` | 档案/预算/平台期概念 | `evolution/search.py` + `selection.py` | 双档案分离、真实新代码父代搜索 |
| `trim/evolution/materialize.py` | 文件物化+hash 思路 | `evolution/materializer.py` | 无 query_repair 模板限定 |
| `trim/harness_bank/loader.py`、`hooks.py` | manifest+确定性 hook 顺序 | `harness/loader.py`、`composition.py` | 无 legacy_flags/检索依赖 |
| `trim/training/opd_projection.py` 等 | 显式映射/合法性检查原则 | `opd/alignment.py`、`realizability.py` | 由 doc_id/curated 改环境对象/先决条件接口 |
| `trim/harness_bank/adaptation.py` | 连续事件与 native-anchor | `opd/anchors.py` | 不止 curate；支持文件/API 对象与信息来源 |
| `trim/harness_bank/shadow.py` | 不修改 Student 的设计意图 | `opd/teacher.py` + 环境 fork | 真实外部状态隔离（状态摘要校验） |
| `trim/training/sr_opd_loss.py` | **逐行核对**的纯 loss/gate/scale | `opd/losses.py` | 无间接 import；本地 kl_from_logits |
| `trim/training/tinker_opd_datum.py` | 双 prefix/目标编码/mask/policy version | `opd/records.py`、`opd/scoring.py` | 不绑定 Tinker datum/检索动作 |
| `trim/training/tinker_rl_opd_trainer.py` | 联合 loss/一次 step/版本同步 | `training/joint.py` | 后端无关接口 |
| `trim/training/sft_engine/*` | 分片 checkpoint/artifact 工程经验 | `training/checkpoint.py` | SFT 引擎不冒充 RL/OPD trainer |

**数值对照**：`sr_opd_loss.py` 的纯函数迁移进 `opd/losses.py`，对照测试
`tests/unit/test_loss_and_grad_parity.py` 将旧实现内嵌为 legacy reference
（保持包独立），loss/梯度在 1e-6 容差内一致；CISPO 语义对照
`trim/training/hf_rl_opd_client.py:270` 附近（`clamp(ratio).detach()`、
`-weight*adv*new_lp*mask`、global-token 归一）。

## 未迁移 / 明确不带入

- Harness-1/Harness-G runtime、BrowseComp+ 检索栈、SEC 配置、vLLM rollout
  worker、mxfp4/bf16 部署经验 —— 与通用 Harness 演化无关，留在旧线。
- `final_answer_recall` 统一奖励 —— 新项目每个环境用官方任务成功指标
  （toy=task_success；真实环境=官方 scorer）。
- `max_extra_llm_calls_per_episode=0` 的旧限制 —— 新默认允许有限额外模型调用
  （pilot ≤2/episode）。
- 旧 `run_rsi.py` 的 plan-only 出口 —— `campaign plan` 显式 `executed=false`，
  `campaign run` 缺能力时非零退出。

## 许可证

纯函数迁移按原仓库许可证保留来源注释（见 `LICENSES/`）。本包其余代码为新实现。
