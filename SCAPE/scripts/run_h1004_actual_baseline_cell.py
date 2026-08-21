#!/usr/bin/env python3
"""Train one actual HF/PEFT H100-4 baseline cell and save an adapter."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scape.training.hf_tool_opd import ScapeHFToolOPD, run_tool_opd_train

TOOLS = {"fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"}
SRC = REPO / "outputs" / "h100_4_privilege_representation"

def load(path: Path, limit: int | None = None):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit: break
    return rows

def response_from_action(action):
    if isinstance(action, str):
        name = action.strip().split()[0]
        args = {}
    else:
        name = str((action or {}).get("name") or (action or {}).get("tool_name") or "end_search")
        args = (action or {}).get("arguments") or (action or {}).get("parameters") or {}
    if name not in TOOLS: name = "end_search"; args = {}
    return json.dumps({"tool_name": name, "parameters": args}, ensure_ascii=False, sort_keys=True)

def make_rows(rows, method):
    out=[]
    for i, r in enumerate(rows):
        if method == "MATCHED_TEXT_PRIVILEGE":
            full = r.get("prompt_textual") or ""
            # Textual baseline sees exactly the deterministic state-time serialization.
        else:
            full = r.get("prompt_full") or ""
        reduced = r.get("prompt_student") or ""
        action = r.get("teacher_full_greedy_tool_call") or r.get("teacher_action") or r.get("student_action")
        out.append({
            "row_id": f"{method}_{i:06d}", "query_id": str(r.get("query_id")),
            "snapshot_hash": r.get("snapshot_hash"), "prompt_reduced": reduced,
            "prompt_full": full, "response_text": response_from_action(action),
            "teacher_action": action, "method": method,
        })
    return out

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method", choices=["OPSD_ACTION_PI","OPHSD_FAITHFUL","MATCHED_TEXT_PRIVILEGE","SMRC_SD_FAITHFUL","OVCSD_FAITHFUL"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-limit", type=int, default=512)
    ap.add_argument("--valid-limit", type=int, default=128)
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    args=ap.parse_args(); random.seed(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    status=args.out/"STATUS_LIVE.md"
    status.write_text(f"# STATUS_LIVE\n\n- method: {args.method}\n- seed: {args.seed}\n- gpu: {args.gpu}\n- status: loading\n", encoding="utf-8")
    train=make_rows(load(SRC/"train_paired.jsonl", args.train_limit), args.method)
    valid=make_rows(load(SRC/"valid_paired.jsonl", args.valid_limit), args.method)
    write_jsonl(args.out/"TRAINING_CELLS.jsonl", train); write_jsonl(args.out/"VALIDATION_CELLS.jsonl", valid)
    (args.out/"DATA_AUDIT.json").write_text(json.dumps({"method":args.method,"seed":args.seed,"train":len(train),"valid":len(valid),"roundtrip_textual":args.method != "MATCHED_TEXT_PRIVILEGE" or all("Harness textualized" in r["prompt_full"] for r in valid),"student_prompt_has_privilege":False,"target":"actual_teacher_executable_action"}, indent=2)+"\n")
    status.write_text(f"# STATUS_LIVE\n\n- method: {args.method}\n- seed: {args.seed}\n- gpu: {args.gpu}\n- status: loading_model\n", encoding="utf-8")
    import torch
    torch.cuda.set_device(args.gpu)
    backend=ScapeHFToolOPD(model_path=args.model, device_map=f"cuda:{args.gpu}", learning_rate=1e-5, anchor_weight=0.05, use_lora=True, lora_r=8, lora_alpha=16)
    audit=backend.audit_tool_spans([r["response_text"] for r in train[:32]])
    status.write_text(f"# STATUS_LIVE\n\n- method: {args.method}\n- seed: {args.seed}\n- gpu: {args.gpu}\n- status: training\n", encoding="utf-8")
    result=run_tool_opd_train(backend, train, valid, loss_path="action_ce", epochs=1, batch_size=1)
    adapter=args.out/"adapter"
    # Keep the PEFT adapter only; the evaluator loads it onto the shared base.
    backend.save_pretrained(str(adapter))
    summary={"status":"completed_actual_lora_training","method":args.method,"seed":args.seed,"gpu":args.gpu,"train_rows":len(train),"valid_rows":len(valid),"training":result,"span_audit":audit,"adapter_path":str(adapter),"merged_path":None,"actual_model_weights":True,"student_inference_privilege":False,"loss_path":"action_ce","source_sha256":hashlib.sha256((SRC/"train_paired.jsonl").read_bytes()).hexdigest()}
    (args.out/"TRAINING_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    with (args.out/"TRAINING_CELLS.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["method","seed","gpu","train_rows","valid_rows","loss_path","D_pre","D_post","L_m","adapter_path","merged_path"]); w.writeheader(); w.writerow({"method":args.method,"seed":args.seed,"gpu":args.gpu,"train_rows":len(train),"valid_rows":len(valid),"loss_path":"action_ce",**{k:result.get(k) for k in ["D_pre","D_post","L_m"]},"adapter_path":adapter,"merged_path":None})
    status.write_text(f"# STATUS_LIVE\n\n- method: {args.method}\n- seed: {args.seed}\n- gpu: {args.gpu}\n- status: completed\n- adapter: {adapter}\n", encoding="utf-8")
    subprocess.run(["bash","-lc",f"cd {args.out} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"],check=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__ == "__main__": main()
