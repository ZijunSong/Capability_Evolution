#!/usr/bin/env python3
"""Actual HF/PEFT LoRA matrix for H100-3 retrieval hygiene projections."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, hashlib, csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = ["fan_out_search","search_corpus","grep_corpus","read_document","review_docs","curate","verify","end_search"]


def action_text(action):
    name = str((action or {}).get("name") or "end_search")
    args = (action or {}).get("arguments") or {}
    return f"to={name}\n{json.dumps(args, ensure_ascii=False, sort_keys=True)}\n</tool_call>"


def prompt(row, full):
    docs = [{"id": d.get("id"), "text": str(d.get("text") or "")[:700]} for d in (row.get("documents") or [])[:10]]
    base = {
        "task": "Choose the next BrowseComp tool call as JSON.",
        "query": row.get("query_text") or "",
        "query_id": str(row.get("query_id")),
        "step": row.get("step", 0),
        "available_tools": TOOLS,
        "runtime": {"snapshot_hash": row.get("snapshot_hash"), "tool_history": (row.get("tool_history") or [])[-4:]},
    }
    if full:
        base["full_runtime_documents"] = docs
        base["curated_ids"] = (row.get("full_view") or {}).get("curated_ids", row.get("curated_ids", []))
        base["projected_target_contract"] = row.get("DEDUP_PROJECTION_TYPE")
    return json.dumps(base, ensure_ascii=False, sort_keys=True)


def convert(src: Path, dst: Path, limit: int = 0, seed: int = 42):
    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if limit: rows = rows[:limit]
    out = []
    for i, r in enumerate(rows):
        projected = r.get("projected_action") or r.get("teacher_native_action") or {"name":"end_search","arguments":{}}
        out.append({
            "row_id": f"{r.get('variant','bundle')}_{seed}_{i:06d}",
            "query_id": str(r.get("query_id")), "snapshot_hash": r.get("snapshot_hash"),
            "variant": r.get("variant"), "student_inference_privilege": False,
            "prompt_reduced": prompt(r, False), "prompt_full": prompt(r, True),
            "response_text": action_text(projected), "projected_action": projected,
            "DEDUP_PROJECTION_TYPE": r.get("DEDUP_PROJECTION_TYPE"),
        })
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(out)


def cell_cmd(args, spec):
    gpu, variant, seed = spec
    out = args.out / "cells" / f"{variant.lower()}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    src = args.bundle / ({"AUTO":"AUTO_PROJECTED_DATA.jsonl","DEDUP":"DEDUP_PROJECTED_DATA.jsonl","AUTO_DEDUP":"AUTO_DEDUP_PROJECTED_DATA.jsonl","SHUFFLED":"SHUFFLED_BUNDLE_PROJECTION_DATA.jsonl"}[variant])
    train = out / "TRAIN_ROWS.jsonl"; valid = out / "VALID_ROWS.jsonl"; test = out / "TEST_ROWS.jsonl"
    if not train.exists(): convert(src, train, args.train_limit, seed)
    if not valid.exists(): convert(src, valid, args.eval_limit, seed+1000)
    if not test.exists(): convert(src, test, args.eval_limit, seed+2000)
    cmd = [args.python, str(REPO / "scripts" / "run_h1003_retrieval_hygiene_cell.py"),
           "--out", str(out), "--model-path", args.model_path,
           "--train-jsonl", str(train), "--valid-jsonl", str(valid), "--test-jsonl", str(test),
           "--component-id", variant, "--seed", str(seed), "--gpu", str(gpu),
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
           "--loss-path", args.loss_path]
    return cmd, out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=REPO/"outputs/0818_retrieval_hygiene_bundle")
    ap.add_argument("--out", type=Path, default=REPO/"outputs/0818_retrieval_hygiene_bundle")
    ap.add_argument("--python", default="/opt/scape-h1003-hf-scorer/bin/python")
    ap.add_argument("--model-path", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--train-limit", type=int, default=256)
    ap.add_argument("--eval-limit", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss-path", default="tool_token_kl")
    ap.add_argument("--poll-seconds", type=int, default=30)
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    specs=[(0,"AUTO",42),(1,"AUTO",43),(2,"DEDUP",42),(3,"DEDUP",43),(4,"AUTO_DEDUP",42),(5,"AUTO_DEDUP",43),(6,"SHUFFLED",42),(7,"SHUFFLED",43)]
    manifest={"status":"running","actual_model_weights":True,"student_inference_privilege":False,"matrix":[],"started_epoch":int(time.time())}
    procs=[]
    for spec in specs:
        cmd,out=cell_cmd(args,spec)
        log=out/"worker.log"
        with log.open("w") as lf:
            p=subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ,"CUDA_VISIBLE_DEVICES":str(spec[0])})
        procs.append((spec,p,out)); manifest["matrix"].append({"gpu":spec[0],"variant":spec[1],"seed":spec[2],"pid":p.pid,"out":str(out)})
        print(json.dumps({"launched":spec,"pid":p.pid}), flush=True)
    (args.out/"LORA_MATRIX_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    while procs:
        alive=[]
        for spec,p,out in procs:
            rc=p.poll()
            if rc is None: alive.append((spec,p,out)); continue
            status={"gpu":spec[0],"variant":spec[1],"seed":spec[2],"pid":p.pid,"returncode":rc,"done":(out/"DONE").exists(),"failed":(out/"FAILED.json").exists()}
            print(json.dumps(status), flush=True)
        procs=alive
        if procs: time.sleep(args.poll_seconds)
    manifest["status"]="completed"; manifest["finished_epoch"]=int(time.time())
    (args.out/"LORA_MATRIX_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__": main()
