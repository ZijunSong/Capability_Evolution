# E0-ReAct：接口与测量验收（AppWorld · Qwen3.5-9B · H0-ReAct vs H0-Act 配对）

> **formal artifact**：`t-20261104074933-hf4ax`（rsi-e0-react-appworld-9b-8gpu-20261104-r1），succeed，runTime 53min（07:53-08:43），8×M402。
> **任务书**：`RSI_Per_Environment_ReAct_H0_Refactor_Guide.md` §12 E0。
> **面板**：16 任务（官方 train 8 + dev 8，每 scenario 组 1 题；manifest `e0_react_appworld.json` digest `97193c41`，official split source = `datasets/*.txt`）。
> **双臂**：H0-ReAct（`react_zero_shot_v1`）vs H0-Act（`act_only_v1`），除 profile 外逐字段一致（seed 17、max_actions 24、bank 1beba6d6 空 bank、同 dep 钉版）。

## 四门验收结果

| 门 | 内容 | 结果 |
|---|---|---|
| G1 | 全任务 `measurement_status=VALID`（官方 `evaluate_task` 离线评分） | ✅ react 16/16，act 16/16 |
| G2 | scorer 不改业务 DB（评分前后哈希一致） | ✅ 双臂 0 例 `db_mutated` |
| G3 | 模型真被调用（n_decisions>0、tokens>0）——clock-freeze 回归检测 | ✅ react n_dec 均值 12.3 / 46,587 out tokens；act 17.4 / 78,778；0 个零决策任务 |
| G4 | 官方 split 清单在场（train/dev/test_normal/test_challenge） | ✅ `OFFICIAL_SPLIT_OK`，manifest frozen=True |

**E0 通过。** 测量链路（真实单调钟 + 官方离线评分 + 记录层）在真机端到端可信。

## 配对结果

| 指标 | H0-ReAct | H0-Act |
|---|---|---|
| strict success（官方 TGC） | **0/16** | **0/16** |
| partial 均值 | 26.2% | 26.2% |
| n_decisions 均值 | 12.3 | 17.4 |
| wall/task 均值 | 323s | 505s |
| 终止分布 | 13 finish / 3 max_actions | 10 finish / 6 max_actions |
| out tokens | 46,587 | 78,778 |
| tool_calls | 68 | 97 |

## 关键发现：0/16 任务产生状态变更

对全部 16 任务跑 **no-op 对照**（reset 后零动作直接评分）：**双臂每任务的 partial 与 no-op 初始状态 partial 完全相同（16/16）**，且实验输出 DB 的 per-app 变更日志全空——即 9B 在两臂下都**没有做出任何改变业务状态的 API 调用**。

这不是测量问题（G1-G4 全过、tokens/工具调用真实发生），而是**能力/协议发现**：

1. **初始状态 partial 本身很高**（均值 26.2%，最高 0.5）——大量测试是"未破坏既有状态"型，no-op 就能拿分。旧口径的"23.5% TGC"里相当部分是初始状态噪声，与本发现一致。
2. **9B zero-shot 在 JSON 动作协议下没有完成一次有效的认证→读→改链路**。68/97 次 python_exec 要么报错要么只读。**轨迹未持久化**（已知记录层缺口，eval json 无 events）导致无法从这次运行直接分类失败模式（认证失败 / API 名拼错 / 只读不写），这是 E0 留下的唯一硬件待修项。
3. ReAct 指令让模型**更快收尾**（13 finish、更少决策、-36% tokens）但没有换来任何状态变更——零样本 ReAct 指令本身不解决"会用这个环境"的问题。

## 对 E1 的建议

1. **预算网格（24/48/72）优先级下降**：预算不是当前瓶颈——react 臂 13/16 主动 finish 且 0 状态变更，加预算无意义。先解决"会调 API"，再谈步数。
2. **E1 前必须补轨迹持久化**（per-task events → eval json 或独立 trace 文件）：没有轨迹，进化的 EvidenceStore 又只剩 telemetry——重蹈"budget_wall→压缩提示"覆辙。
3. **E1 进化反馈应围绕接口失败模式**（认证、API 发现、参数构造），env doc 已提供正确的发现命令（`api_docs.show_api_doc` 等），进化器的第一类候选大概率是"先查文档再调用"类的中途决策组件。
4. WebShop/ALFWorld 臂按同模式复制（清单/provider 已就绪），但**先看 AppWorld 轨迹分类**再定三域齐跑还是先迭代。

## 遗留

- trace 持久化（`run_panel` 不存 events）——E0 唯一硬件缺口。
- 27B H0-ReAct/Act 配对未跑（E3 复验阶段做）。
- AppWorld H0-ReAct 底座是否冻结：**先修 trace，看一轮失败分类后再冻结**。
