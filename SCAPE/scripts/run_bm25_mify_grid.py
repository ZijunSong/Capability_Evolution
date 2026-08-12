#!/usr/bin/env python3
"""Launch BM25 + mify API BrowseComp harness condition grid.

This is a SCAPE-local orchestrator that uses the SCOPE BM25 harness runner with
OpenAI-compatible mify credentials. It does not use cloud Chroma and does not
start local vLLM. Outputs are one directory per condition under SCAPE outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCOPE = Path("/mnt/songzijun/Capability_Evolution/SCOPE")
PY = Path("/opt/vllm-qwen3-1.7b-harness/bin/python")
MIFY_ENV = Path("/mnt/songzijun/mify_api.env")
BM25 = SCOPE / "external/BrowseComp-Plus/indexes/bm25"

COMPONENT_ENV = {
    "adaptive_rerank_instruction": "V8D_ADAPTIVE_RERANK_INSTRUCTION",
    "importance_tagging": "V8D_IMPORTANCE_TAGGING",
    "subtractive_curation": "V8D_SUBTRACTIVE_CURATION",
    "auto_populate_first_search": "V8D_AUTO_POPULATE_FIRST_SEARCH",
    "evidence_graph": "V8D_EVIDENCE_GRAPH",
    "sentence_compress": "V8D_SENTENCE_COMPRESS",
    "content_dedup": "V8D_CONTENT_DEDUP",
    "chunk_neighbors": "V8D_CHUNK_NEIGHBORS",
    "verify_tool": "V8D_VERIFY_TOOL",
    "token_budget_marker": "V8D_TOKEN_BUDGET_MARKER",
}

CONDITIONS = [
    ("full", []),
    ("semantic_light", ["adaptive_rerank_instruction", "importance_tagging", "subtractive_curation", "auto_populate_first_search"]),
    ("minus_adaptive_rerank_instruction", ["adaptive_rerank_instruction"]),
    ("minus_importance_tagging", ["importance_tagging"]),
    ("minus_subtractive_curation", ["subtractive_curation"]),
    ("minus_auto_populate_first_search", ["auto_populate_first_search"]),
    ("minus_evidence_graph", ["evidence_graph"]),
    ("minus_sentence_compress", ["sentence_compress"]),
    ("minus_content_dedup", ["content_dedup"]),
    ("minus_chunk_neighbors", ["chunk_neighbors"]),
    ("minus_verify_tool", ["verify_tool"]),
    ("minus_token_budget_marker", ["token_budget_marker"]),
    ("coalition_rerank_budget", ["adaptive_rerank_instruction", "token_budget_marker"]),
    ("coalition_tagging_curation", ["importance_tagging", "subtractive_curation"]),
    ("coalition_curation_graph", ["subtractive_curation", "evidence_graph"]),
    ("coalition_auto_seed_rerank", ["auto_populate_first_search", "adaptive_rerank_instruction"]),
    ("coalition_graph_compress", ["evidence_graph", "sentence_compress"]),
    ("coalition_verify_graph", ["verify_tool", "evidence_graph"]),
]


def load_mify_env() -> dict[str, str]:
    env: dict[str, str] = {}
    text = MIFY_ENV.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        body = line[len("export "):]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value == "$MIFY_API_KEY":
            value = env.get("MIFY_API_KEY", "")
        env[key.strip()] = value
    return env


def condition_env(disabled: list[str]) -> dict[str, str]:
    env = {v: "1" for v in COMPONENT_ENV.values()}
    # Match the SCOPE full_v2 default: chunk_neighbors is off unless explicitly tested.
    env["V8D_CHUNK_NEIGHBORS"] = "0"
    for component in disabled:
        key = COMPONENT_ENV[component]
        env[key] = "0"
    return env


def launch_condition(root: Path, name: str, disabled: list[str], *, limit: int, parallel: int, max_turns: int, max_tokens: int) -> dict[str, str]:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    mify = load_mify_env()
    env.update({
        "base_url": mify["API_BASE"],
        "api_key": mify["MIFY_API_KEY"],
        "model_name": mify.get("MODEL_NAME", "moonshot/kimi-k2.6"),
        "OPENAI_API_KEY": mify["MIFY_API_KEY"],
        "API_BASE": mify["API_BASE"],
        "API_KEY": mify["MIFY_API_KEY"],
        "MIFY_API_KEY": mify["MIFY_API_KEY"],
        "MODEL_NAME": mify.get("MODEL_NAME", "moonshot/kimi-k2.6"),
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        "PYTHONPATH": f"{SCOPE}:{SCOPE / 'tinker-cookbook'}",
        "PYTHONUNBUFFERED": "1",
        "USE_LEGACY_API_AGENT": "0",
        "BROWSECOMP_BM25_INDEX_PATH": str(BM25),
        "BROWSECOMPPLUS_ANSWERS_PATH": str(SCOPE / "external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"),
        "BROWSECOMPPLUS_QUERIES_PATH": str(SCOPE / "external/BrowseComp-Plus/topics-qrels/queries.tsv"),
        "BROWSECOMPPLUS_QRELS_GOLD_PATH": str(SCOPE / "external/BrowseComp-Plus/topics-qrels/qrel_golds.txt"),
        "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH": str(SCOPE / "external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt"),
        "CHAT_MIN_TURNS_BEFORE_END": "4",
        "CHAT_MIN_CURATED_BEFORE_END": "1",
        "CHAT_MAX_WM_CHARS": "18000",
        "CHAT_MAX_RECENT_TURNS": "4",
    })
    env.update(condition_env(disabled))
    cmd = [
        str(PY), "training/rollout_harness_browsecomp.py",
        "--model-path", "/mnt/songzijun/models/harness-1-extracted/harness-1",
        "--harness-config", "harness/configs/modules_full_v2.yaml",
        "--split", "test",
        "--limit", str(limit),
        "--max-turns", str(max_turns),
        "--max-tokens", str(max_tokens),
        "--temperature", "1",
        "--top-p", "0.95",
        "--seed", "42",
        "--parallel", str(parallel),
        "--retrieval", "bm25",
        "--bm25-index-path", str(BM25),
        "--reranker", "none",
        "--output-dir", str(out),
        "--policy", "api",
        "--no-manage-vllm",
        "--no-resume",
    ]
    log = out / "nohup_rollout.log"
    with log.open("wb") as fh:
        proc = subprocess.Popen(cmd, cwd=str(SCOPE), env=env, stdout=fh, stderr=subprocess.STDOUT)
    (out / "nohup_rollout.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    (out / "condition.json").write_text(json.dumps({"name": name, "disabled": disabled, "env": condition_env(disabled), "cmd": cmd}, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "pid": str(proc.pid), "out": str(out), "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=REPO / "outputs" / "bm25_mify_grid")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-concurrent", type=int, default=4)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    queue = list(CONDITIONS)
    running: list[dict[str, str]] = []
    launched: list[dict[str, str]] = []
    while queue or running:
        still: list[dict[str, str]] = []
        for item in running:
            pid = int(item["pid"])
            ret = subprocess.run(["bash", "-lc", f"kill -0 {pid} 2>/dev/null"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ret.returncode == 0:
                still.append(item)
        running = still
        while queue and len(running) < args.max_concurrent:
            name, disabled = queue.pop(0)
            item = launch_condition(args.out_root, name, disabled, limit=args.limit, parallel=args.parallel, max_turns=args.max_turns, max_tokens=args.max_tokens)
            running.append(item)
            launched.append(item)
            print(json.dumps({"launched": item, "running": len(running), "remaining": len(queue)}, ensure_ascii=False), flush=True)
        manifest = {"out_root": str(args.out_root), "limit": args.limit, "parallel": args.parallel, "max_turns": args.max_turns, "max_tokens": args.max_tokens, "max_concurrent": args.max_concurrent, "launched": launched, "running": running, "remaining": [q[0] for q in queue]}
        (args.out_root / "GRID_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if queue or running:
            time.sleep(30)
    print(json.dumps({"done": True, "out_root": str(args.out_root), "n": len(launched)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
