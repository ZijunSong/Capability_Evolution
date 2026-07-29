#!/usr/bin/env bash
# Quick E0 experiment status (safe to run anytime).
ROOT="/data/ppnm/Capability_Evolution/SCOPE/outputs/scope_e0_distillability"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -8
echo
echo "=== Processes ==="
ps aux | grep -E 'run_e0_distillability_nohup|distillability/runner|vllm serve.*8776' | grep -v grep || echo "(none)"
echo
echo "=== Episode progress (ok / 100) ==="
python3 <<'PY'
import json
from pathlib import Path
root = Path("/data/ppnm/Capability_Evolution/SCOPE/outputs/scope_e0_distillability")
for cap in ["duplicate_evidence","stop_decision","evidence_curation","verification_decision","external_verification","deterministic_truncation"]:
    for mode in ["off","proc","full"]:
        p = root / cap / mode / "episodes.jsonl"
        if not p.exists():
            continue
        ok = err = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("error"):
                err += 1
            else:
                ok += 1
        mark = "✓" if ok >= 100 else ("~" if ok > 0 else "✗")
        print(f"  {mark} {cap}/{mode}: ok={ok} err={err}")
PY
echo
echo "=== Log tail ==="
tail -3 "${ROOT}/nohup_master.log" 2>/dev/null || echo "(no master log)"
