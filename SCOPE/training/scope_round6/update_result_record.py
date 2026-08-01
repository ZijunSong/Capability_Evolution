#!/usr/bin/env python3
"""Append Round 6 results to result-record.md."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round6"
RECORD = _REPO / "result-record.md"
GATE = OUT / "phase_b/ROOT_CAUSE_GATE.json"
THR = OUT / "calibration/thresholds.json"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_agg(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _holdout_rows() -> list[str]:
    rows = []
    root = OUT / "closed_loop/holdout_50q"
    for path in sorted(root.glob("*/*/aggregated_metrics.json")):
        variant = path.parent.parent.name
        shard = path.parent.name
        d = _load_agg(path)
        db = d.get("direct_behavior", {})
        n_ep = d.get("n_episodes", 0)
        rows.append(
            f"| {variant}/{shard} | {n_ep} | "
            f"{db.get('DupRejectRecall', 0):.3f} | {db.get('FalseSkipRate', 0):.3f} | "
            f"{db.get('BalancedAcc', 0):.3f} | {db.get('predicted_SKIP_prior', 0):.3f} | "
            f"{d.get('mean_reward', 0):.3f} | {d.get('mean_recall', 0):.3f} |"
        )
    return rows


def _calib_rows() -> list[str]:
    rows = []
    root = OUT / "closed_loop/calib_25q"
    for path in sorted(root.glob("*/*/aggregated_metrics.json")):
        tag = path.parent.parent.name
        seed = path.parent.name
        d = _load_agg(path)
        db = d.get("direct_behavior", {})
        rows.append(
            f"| {tag}/{seed} | {db.get('DupRejectRecall', 0):.3f} | "
            f"{db.get('FalseSkipRate', 0):.3f} | {db.get('BalancedAcc', 0):.3f} | "
            f"{d.get('mean_reward', 0):.3f} |"
        )
    return rows


def build_round6_section() -> str:
    gate = _load_agg(GATE) if GATE.exists() else {}
    thr = _load_agg(THR) if THR.exists() else {}
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    holdout = _holdout_rows()
    calib = _calib_rows()

    lines = [
        "",
        "---",
        "",
        "## Round 6 — Closed-loop Calibration & On-Policy Shift Audit（07-31 ~ 08-01）",
        "",
        f"**Git：** `scope/dup-round6-closedloop-calibration` @ `{_git_head()}`",
        "",
        "**文档：** `0731-todo1.md`",
        "**产物根：** `outputs/scope_round6/`",
        "**报告：** `outputs/scope_round6/ROUND6_REPORT.md`",
        f"**记录更新时间：** {now}",
        "",
        "### Gate 结论",
        "",
        "| Flag | 值 |",
        "| --- | --- |",
        f"| `H_RUNTIME` | **{gate.get('H_RUNTIME', 'n/a')}** |",
        f"| `H_CALIB` | **{gate.get('H_CALIB', 'n/a')}** |",
        f"| `H_SHIFT` | **{gate.get('H_SHIFT', 'n/a')}** |",
        f"| `H_FEEDBACK` | **{gate.get('H_FEEDBACK', 'n/a')}** |",
        f"| adapter↔merged parity | {gate.get('adapter_merged_parity', 'n/a')} |",
        f"| HF↔runtime parity | {gate.get('hf_runtime_parity', 'n/a')} |",
        f"| `ROUND6_CLOSED_LOOP_POSITIVE` | **false** |",
        f"| `RECOMMEND_830` | **false** |",
        "",
        "### Setting（冻结）",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| Base model | `Qwen2.5-7B-Instruct` |",
        "| O7 checkpoint | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}` |",
        "| Loss / LoRA | `discriminative_ce` · r=64 · α=128（与 Round5 O7 相同） |",
        "| Runtime | \\(H_{\\min,\\text{v2}}\\) · `modules_minimal_v2.yaml` |",
        "| 100q manifest | `round2_audit_100q/query_manifest.json` |",
        "| Closed-loop | max_turns=35 · max_tokens=2048 · temperature=1.0 · BM25 |",
        "| Calibration slice | shard0（25q）closed-loop states |",
        "| Prospective 25q | shard1（C-CALIB） |",
        "| Holdout 50q | shard2+shard3（Phase D） |",
        f"| τ_seed42 / 43 / 44 | {thr.get('tau_seed42', 'n/a')} / {thr.get('tau_seed43', 'n/a')} / {thr.get('tau_seed44', 'n/a')} |",
        f"| τ_shared | {thr.get('tau_shared', 'n/a')} |",
        "| Decision rule | SKIP iff margin ≥ τ（`score_skip - score_keep`） |",
        "",
        "### Phase B — Cross-score 核心结论",
        "",
        "同一 checkpoint × 多 state source 离线重打分（merged HF scorer）：",
        "",
        "- valid522 与全部 B6 admission states 上 **AUROC=1.0**（三 seeds 一致）",
        "- 同一 states 上 **BalancedAcc@threshold=0 亦为 1.0**（offline 排序完美）",
        "- **H_RUNTIME / H_SHIFT / H_CALIB / H_FEEDBACK 均为 false**",
        "- 推论：Round5 闭环失败**不是** runtime parity 或 on-policy AUROC 崩塌；问题在 **closed-loop 决策边界 / 行为层**（校准后仍高 FSR）",
        "",
        "产物：`phase_b/CROSS_SCORE_MATRIX.csv` · `ROOT_CAUSE_GATE.json` · `STATE_SHIFT_REPORT.md`",
        "",
        "### Phase C-CALIB — shard1 25q（校准后前瞻）",
        "",
        "| Run | DupRejectRecall | FSR | BalancedAcc | mean_reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(calib or ["| (none) | — | — | — | — |"])
    lines.extend([
        "",
        "**解读：** per-seed τ 在 shard0 上可达 FSR≤5%；但 shard1 闭环中 O7 仍 **几乎全部 pred SKIP**（DupRejectRecall≈1 但 FSR≈1），校准 **未** 转化为可接受闭环行为。",
        "",
        "### Phase D — Holdout 50q（shard2+shard3）",
        "",
        "| Run | n_ep | DupRejectRecall | FSR | BalancedAcc | SKIP prior | reward | recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    lines.extend(holdout or ["| (pending) | — | — | — | — | — | — | — |"])
    lines.extend([
        "",
        "**解读：**",
        "",
        "1. **Base：** DupRejectRecall=0（从不 SKIP），FSR=0；与 Round5 一致。",
        "2. **O7 + per-seed τ：** 校准后闭环仍 **SKIP 先验≈1.0**，FSR≈0.97–1.0；DupRejectRecall 高但来自 **误伤 unique**，非成功 duplicate internalization。",
        "3. **任务保持失败：** mean_reward 系统性低于 Base（~0.04–0.26 vs Base ~0.16/0.04 on holdout shards）。",
        "4. **Round6 正信号 gate 未过：** 要求 DupRejectRecall≥0.10 且 FSR≤0.05 且 BalancedAcc>0.50 — O7 满足前者但 **FSR 严重超标**。",
        "",
        "### Round 6 最终判定",
        "",
        "```text",
        "ROUND6_CLOSED_LOOP_POSITIVE = false",
        "RECOMMEND_830 = false",
        "C-SHIFT (Dagg retrain) = 未触发（H_SHIFT=false）",
        "```",
        "",
        "### 工程备注",
        "",
        "1. Phase D 首次运行 `get_tau()` JSON key bug（int vs str）导致 O7 holdout 未启动；已修复并用 `resume_holdout_o7.sh` 补跑。",
        "2. `seed43/shard2` 曾在 query 335 卡住 9/25；kill 后 `--resume` 续跑剩余 16 题。",
        "3. 所有闭环指标从 `episodes.jsonl` + `dup_admission_events.jsonl` 重聚合。",
        "",
        "### 代码与脚本",
        "",
        "```text",
        "training/scope/decision_config.py",
        "training/scope_round6/",
        "scripts/scope_round6/",
        "tests/scope/test_round6_scorer.py",
        "outputs/scope_round6/",
        "```",
        "",
        "### 下一步",
        "",
        "```text",
        "RECOMMEND_830=false → 禁止扩 830 / E1 / weighting / multi-capability",
        "P0 转向：为何 offline margin 完美 + τ 校准后 closed-loop 仍全 SKIP？",
        "  → runtime vLLM scorer vs HF 在 live admission 路径是否仍一致",
        "  → τ 在 offline replay margin 上有效但对 live score scale 无效",
        "  → 考虑 on-policy Dagg 前需先修 live decision 路径或 score telemetry 对齐",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    section = build_round6_section()
    text = RECORD.read_text(encoding="utf-8")

    marker = "## Round 6 — Closed-loop Calibration"
    if marker in text:
        pre = text.split(marker)[0].rstrip()
        text = pre + section
    else:
        # Replace Round 5 "下一步" block tail or append
        old_next = "### 下一步\n\n```text\nROUND5_POSITIVE_SIGNAL=false"
        if old_next in text:
            idx = text.find("### 下一步\n\n```text\nROUND5_POSITIVE_SIGNAL=false")
            text = text[:idx].rstrip() + section
        else:
            text = text.rstrip() + section

    # Update top "当前结论" block
    top_marker = "## 当前结论（"
    if top_marker in text:
        end = text.find("\n---", text.find(top_marker))
        new_top = (
            "## 当前结论（2026-08-01）\n\n"
            "- **已成立：** same-state shadow / info-safe · measurement+scorer · Round5 O7 offline 双侧分离 · "
            "**Round6 offline cross-score AUROC=1.0（valid522 + 全部 B6 states）** · runtime parity=1.0（adapter/merged/HF）。\n"
            "- **尚未成立：** Dup **closed-loop internalization** positive signal · 校准后仍 **FSR≈1.0** · "
            "reward 低于 Base · **`RECOMMEND_830=false`**。\n"
            "- **Round 6 判定：** `H_RUNTIME/H_SHIFT/H_CALIB/H_FEEDBACK` 均为 false；"
            "`ROUND6_CLOSED_LOOP_POSITIVE=false`。\n"
            "- **关键发现：** offline 排序与 AUROC 完美，但 per-seed margin 阈值校准 **不能** 将闭环行为拉到 FSR≤5%；"
            "O7 在 holdout 上表现为 **几乎全部 SKIP**（先验≈1），非可控 duplicate rejection。\n"
            "- **当前 P0：** live closed-loop decision 路径 / score scale 与 offline replay 差异；"
            "禁止扩 830、E1、weighting、multi-capability。\n"
            "- **主线：** 修 live admission 决策一致性 → 再评估是否需 on-policy Dagg。\n"
        )
        text = new_top + text[text.find("\n---", text.find(top_marker)) :]

    RECORD.write_text(text, encoding="utf-8")
    print(f"Updated {RECORD}")


if __name__ == "__main__":
    main()
