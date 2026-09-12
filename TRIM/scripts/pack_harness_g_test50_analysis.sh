#!/usr/bin/env bash
# Pack four Harness-G bcplus_test_50 runs into one analysis folder.
#
#   RUN_ID=20260911_175948_hg_test50_gpu04567 bash TRIM/scripts/pack_harness_g_test50_analysis.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-20260911_175948_hg_test50_gpu04567}"
SRC="/data/ppnm/trim_bcplus_test50_eval_harness_g/results/${RUN_ID}"
OUT="/data/ppnm/trim_bcplus_test50_eval_harness_g/analysis_${RUN_ID}"
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"

[[ -d "${SRC}" ]] || { echo "missing source ${SRC}" >&2; exit 1; }
mkdir -p "${OUT}"

for model in gpt-oss-20b Qwen3-4B-Instruct-2507; do
  for comp in zero all; do
    dest="${OUT}/${model}/${comp}"
    mkdir -p "${dest}"
    rsync -a "${SRC}/${model}/${comp}/" "${dest}/"
    src_pq="${SRC}/${model}/${comp}/harness/PER_QUERY.jsonl"
    [[ -f "${src_pq}" ]] && cp -a "${src_pq}" "${OUT}/PER_QUERY_${model}_${comp}.jsonl"
  done
done

cp -a "${SRC}/MANIFEST.json" "${OUT}/MANIFEST.json"
[[ -f "/data/ppnm/trim_bcplus_test50_eval_harness_g/logs/${RUN_ID}.nohup.log" ]] && \
  cp -a "/data/ppnm/trim_bcplus_test50_eval_harness_g/logs/${RUN_ID}.nohup.log" "${OUT}/master.nohup.log"

cd "${TRIM_ROOT}"
"${PY}" - <<'PY' "${OUT}"
import json, sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for pq in sorted(out.glob("PER_QUERY_*.jsonl")):
    name = pq.stem.replace("PER_QUERY_", "")
    model, comp = name.rsplit("_", 1) if "_zero" in name or "_all" in name else (name, "")
    if name.endswith("_zero"):
        model = name[: -len("_zero")]
        comp = "zero"
    elif name.endswith("_all"):
        model = name[: -len("_all")]
        comp = "all"
    else:
        continue
    per = []
    with pq.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                per.append(json.loads(line))
    tools = Counter()
    for r in per:
        for t in r.get("tool_names") or []:
            tools[t] += 1
    summary_path = out / model / comp / "FOUR_CELL_OFFICIAL_SUMMARY.json"
    s = {}
    if summary_path.is_file():
        s = json.loads(summary_path.read_text()).get("settings", [{}])[0]
    rows.append(
        {
            "model": model,
            "component": comp,
            "n_queries": len(per),
            "legal_action_rate": s.get("legal_action_rate"),
            "recall": s.get("recall"),
            "test_evidence_recall_at_5": s.get("test_evidence_recall_at_5"),
            "mean_tool_calls_per_query": s.get("mean_tool_calls_per_query"),
            "error_rate": s.get("error_rate"),
            "tool_name_hist": dict(tools),
            "per_query_path": str(pq.relative_to(out)),
            "summary_path": str((model + "/" + comp).replace("\\", "/")),
        }
    )

report = {
    "analysis_dir": str(out),
    "runs": rows,
    "issues": [
        "Qwen outputs {\"tool\":\"init\"} but legacy prompt lacked registered tools; fixed in harness_g_runtime.",
        "gpt-oss lacked Harmony tool-call history on continuation turns; fixed via build_harness_g_context.",
        "parse_harness_g_action now accepts JSON blobs with tool/name keys.",
    ],
}
(out / "ANALYSIS.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo "packed → ${OUT}"
