#!/usr/bin/env python3
"""Evaluate saved sr_opd_ce + CISPO adapters on the official 384 pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
if str(_SCAPE) not in sys.path:
    sys.path.insert(0, str(_SCAPE))

from scape.eval.adapter_reload_audit import audit_saved_adapter
from scape.eval.official_query_pool import load_official_384
from scape.eval.sr_opd_four_cell_eval import write_eval_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--component", default="sentence_compress")
    p.add_argument("--adapter-map", type=Path, help="JSON {cell: adapter_dir}")
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--base-model", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows, pool_meta = load_official_384(manifest=args.eval_manifest)
    adapter_map = {}
    if args.adapter_map:
        adapter_map = json.loads(Path(args.adapter_map).read_text(encoding="utf-8"))
    audits = []
    for cell, path in adapter_map.items():
        if path:
            audits.append(audit_saved_adapter(Path(path), cell=str(cell)))
        else:
            audits.append({"cell": cell, "adapter_dir": None, "reload_ready": True})
    if args.audit_only:
        payload = write_eval_outputs(
            args.out,
            component_id=args.component,
            summaries=[
                {
                    "setting": "audit_only",
                    "n_queries": len(rows),
                    "legal_action_rate": None,
                    "test_evidence_recall_at_5": None,
                    "mean_tool_calls_per_query": None,
                    "tool_search_cost": None,
                    "note": "Adapter audit only; closed-loop numbers require --adapter-map plus a live model eval via run_sr_opd_four_cell.py",
                }
            ],
            adapter_audits=audits,
            pool_meta=pool_meta,
        )
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    if not args.base_model:
        raise SystemExit("pass --base-model for live 384 eval, or --audit-only")
    from scape.eval.browsecomp_retrieval import open_retrieval
    from scape.eval.harmony_runtime import load_harmony_enc
    from scape.training.four_cell_runtime import eval_closed_loop
    from scape.training.hf_rl_opd_client import restore_trainable, snapshot_trainable
    from scape.training.hf_tool_opd import ScapeHFToolOPD
    from safetensors.torch import load_file
    from scape.eval.adapter_reload_audit import remap_lora_state

    rows = rows[: args.n_eval] if args.n_eval else rows
    backend = ScapeHFToolOPD(model_path=args.base_model, device_map=f"cuda:{int(args.gpu)}", use_lora=True)
    theta0 = snapshot_trainable(backend.model)
    enc = load_harmony_enc()
    searcher = open_retrieval()
    summaries = []
    for cell, path in (adapter_map or {"before": None}).items():
        restore_trainable(backend.model, theta0)
        if path:
            weights = remap_lora_state(load_file(str(Path(path) / "adapter_model.safetensors")))
            missing, _un = backend.model.load_state_dict(weights, strict=False)
            if [x for x in missing if "lora_" in x]:
                raise RuntimeError(f"reload failed: {cell}")
        ev, traces = eval_closed_loop(
            backend,
            rows,
            component_id=args.component,
            max_new=384,
            max_turns=6,
            seed=42,
            enc=enc,
            searcher=searcher,
        )
        ev["setting"] = cell
        cell_dir = args.out / str(cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        with (cell_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
            for tr in traces:
                handle.write(json.dumps(tr, ensure_ascii=False) + "\n")
        summaries.append(ev)
    payload = write_eval_outputs(
        args.out,
        component_id=args.component,
        summaries=summaries,
        adapter_audits=audits,
        pool_meta=pool_meta,
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
