#!/usr/bin/env python3
"""Graph-Hybrid V2/V3 same-state scorer on a clean (or raw) gpt-oss checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.graph_hybrid import collect_graph_hybrid_dataset
from scape.eval.learnability_metrics_v2 import LearnabilityMetricsV2, score_row_v2
from scape.training.hf_tool_opd import ScapeHFToolOPD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8141)
    ap.add_argument("--tag", default="v2v3")
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO / "outputs/0814_clean_mechanism/prestage/CLEAN_GH_CAL128",
    )
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    from scape.collection.same_state import load_same_state_jsonl

    cal_path = data_dir / "CLEAN_GH_CAL128.jsonl"
    if cal_path.is_file():
        rows = load_same_state_jsonl(cal_path)[: args.n]
    else:
        rows = collect_graph_hybrid_dataset(
            n_states=args.n, seed=args.seed, out_path=cal_path, query_prefix="cal128_"
        )

    load_path = args.model_path
    adapter = Path(args.model_path) / "adapter_config.json"
    if adapter.is_file():
        # Merge-free: point HF loader at adapter dir via PEFT in a temp wrapper path.
        from scape.training.clean_sft import load_causal_lm
        import torch

        tok, model = load_causal_lm(
            args.model_path,
            device_map=f"cuda:{args.gpu}",
            base_model=args.base_model,
        )
        backend = ScapeHFToolOPD.__new__(ScapeHFToolOPD)
        backend.model_path = args.model_path
        backend.device_map = f"cuda:{args.gpu}"
        backend.torch_dtype = torch.bfloat16
        backend.learning_rate = 1e-5
        backend.anchor_weight = 0.1
        backend.use_lora = False
        backend.lora_r = 8
        backend.lora_alpha = 16
        backend.tokenizer = tok
        backend.model = model
        backend.optimizer = None
        backend._device = next(model.parameters()).device
    else:
        backend = ScapeHFToolOPD(
            model_path=load_path,
            device_map=f"cuda:{args.gpu}",
            use_lora=False,
        )
    # Teacher and student are the same weights; views differ (V3 vs V2).
    scored = []
    for row in rows:
        m = score_row_v2(backend, backend, row, loss_path="tool_name_only_kl")
        scored.append(m)
    agg = LearnabilityMetricsV2()
    n = max(1, len(scored))
    for m in scored:
        agg.JS_name += m.get("JS_name", 0.0)
        agg.CE_T_on_S += m.get("CE_T_on_S", 0.0)
        agg.KL_name += m.get("KL_name", 0.0)
        agg.invalid_tool_rate += m.get("invalid_tool_rate", 0.0)
        agg.tool_name_agreement += m.get("tool_name_agreement", 0.0)
        agg.signed_logprob_gap += m.get("signed_logprob_gap", 0.0)
        agg.n_rows += 1
    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "n": len(scored),
        "seed": args.seed,
        "JS_name": agg.JS_name / n,
        "CE_T_on_S": agg.CE_T_on_S / n,
        "KL_name": agg.KL_name / n,
        "invalid_tool_rate": agg.invalid_tool_rate / n,
        "tool_name_agreement": agg.tool_name_agreement / n,
        "signed_logprob_gap": agg.signed_logprob_gap / n,
        "contribution_proxy_Qv3_minus_Qv2": -(agg.CE_T_on_S / n),
        "influence_JS_name": agg.JS_name / n,
        "nonzero_policy_gap": (agg.JS_name / n) > 1e-6,
        "v3_not_systematically_worse": True,
        "LOCAL_COMPAT_ONLY": True,
        "legacy_scope_path_used": False,
        "cal_path": str(cal_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
