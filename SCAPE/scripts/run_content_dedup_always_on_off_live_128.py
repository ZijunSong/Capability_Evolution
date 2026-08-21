#!/usr/bin/env python3
"""Strict content_dedup always-on/off live paired fork on the frozen 128 states."""
from __future__ import annotations
import argparse, copy, hashlib, json, re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from run_h100_2_live_fork_replay import (LiveState, HFContinuationScorer, DualViewRenderer,
    build_searcher, _load_queries, _load_qrels, policy_action)
SEEDS=(2214,2215); KS=(4,8)
SOURCE=ROOT/"outputs/0820_content_dedup_real_recall_128"
BCP=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
MODEL=Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
CORPUS=ROOT/"outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl"

def fp(text): return hashlib.sha1(" ".join(re.findall(r"[a-z0-9]+", text.lower())).encode()).hexdigest()[:16]

class StrictState(LiveState):
    def __init__(self, *a, dedup_enabled=False, **kw):
        self.dedup_enabled=dedup_enabled; self._seen_text=set()
        super().__init__(*a, **kw)
    def _filter_dedup(self, docs):
        if not self.dedup_enabled: return docs
        out=[]
        for d in docs:
            key=fp(d.get('text',''))
            if key in self._seen_text: continue
            self._seen_text.add(key); out.append(d)
        return out
    def _search(self, query, k=20):
        hits=self.searcher.search(query,k); docs=[]
        for h in hits:
            docs.append({'id':str(h.docid),'text':self._doc_text(getattr(h,'raw',None) or '')[:1800]})
        docs=self._filter_dedup(docs); self.documents=docs; self.cost+=1
    @staticmethod
    def _doc_text(raw):
        if isinstance(raw,str):
            try:
                o=json.loads(raw); return str(o.get('contents') or o.get('text') or raw) if isinstance(o,dict) else raw
            except Exception:return raw
        return str(raw or '')
    def clone(self,suffix):
        n=object.__new__(StrictState); n.qid=self.qid;n.query=self.query;n.gold=set(self.gold);n.searcher=self.searcher;n.component=self.component;n.branch_seed=f'{self.branch_seed}:{suffix}';n.step=self.step;n.documents=copy.deepcopy(self.documents);n.curated_ids=list(self.curated_ids);n.read_ids=list(self.read_ids);n.verified_supported=list(self.verified_supported);n.verified_unsupported=list(self.verified_unsupported);n.history=copy.deepcopy(self.history);n.observations=copy.deepcopy(self.observations);n.cost=self.cost;n.dedup_enabled=self.dedup_enabled;n._seen_text=set(self._seen_text);return n

def load_rows(source, seed, K, limit=None):
 p=source/f'seed{seed}_K{K}/shards/content_dedup_K{K}.jsonl'; rows=[json.loads(x) for x in p.open() if x.strip()]
 return rows[:limit] if limit else rows

def make_start(r,queries,qrels,searcher,dedup):
 st=StrictState(qid=r['query_id'],query=queries[r['query_id']],gold=qrels[r['query_id']],searcher=searcher,component='content_dedup',branch_seed=f"always:{r['snapshot_hash']}",dedup_enabled=dedup)
 st.documents=copy.deepcopy(r['initial_candidate_evidence_ids']); st.curated_ids=copy.deepcopy(r['branch_S_endpoint']['final_curated_ids'][:2]); st.cost=4; st.step=0
 # Existing frozen initial pool is the state before forced action; seed dedup memory from it.
 if dedup: st._seen_text={fp(d.get('text','')) for d in st.documents}
 st.read_ids=[];st.history=[];st.observations=[];st.verified_supported=[];st.verified_unsupported=[]
 return st

def branch(start, first, K, scorer, renderer, teacher):
 st=start.clone('teacher' if teacher else 'student'); trace=[]; st.execute(first); trace.append({'phase':'forced_first','action':dict(first),'metrics':st.metrics(),'policy':'teacher_always_on' if teacher else 'student_always_off'})
 for i in range(K-1):
  # component mask is retained for every model decision; continuation policy remains Reduced.
  a,d,dual=policy_action(st,scorer,renderer,component='content_dedup',full=teacher)
  st.execute(a); trace.append({'phase':f'continue_{i+1}','action':a,'snapshot_hash':dual['snapshot_hash'],'metrics':st.metrics(),'policy':'teacher_always_on_reduced_continuation' if teacher else 'student_always_off_reduced_continuation'})
 return st,trace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--model',type=Path,default=MODEL);ap.add_argument('--corpus-path',type=Path,default=CORPUS);ap.add_argument('--out-dir',type=Path,default=ROOT/'outputs/0821_content_dedup_always_on_off_live_128');ap.add_argument('--limit',type=int,default=None);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 queries=_load_queries(BCP/'topics-qrels/queries.tsv');qrels=_load_qrels(BCP/'topics-qrels/qrel_evidence.txt');searcher,backend=build_searcher(BCP/'indexes/bm25',args.corpus_path);scorer=HFContinuationScorer(str(args.model),device=args.device);renderer=DualViewRenderer();rows=[]
 for seed in SEEDS:
  for K in KS:
   src=load_rows(SOURCE,seed,K,2 if args.smoke else args.limit)
   for r in src:
    s0=make_start(r,queries,qrels,searcher,False);t0=make_start(r,queries,qrels,searcher,True)
    sf,st=branch(s0,r['a_S'],K,scorer,renderer,False);tf,tt=branch(t0,r['a_T'],K,scorer,renderer,True);sm,tm=sf.metrics(),tf.metrics()
    rows.append({'component':'content_dedup','contract':'strict Teacher-always-on vs Student-always-off; reduced continuation; first action included in K','seed':seed,'K':K,'state_id':r['state_id'],'query_id':r['query_id'],'snapshot_hash':r['snapshot_hash'],'first_action_disagreement_rate':int(r['a_S']!=r['a_T']),'first_action_S':r['a_S'],'first_action_T':r['a_T'],'tool_cost_S':sm['tool_search_cost'],'tool_cost_T':tm['tool_search_cost'],'tool_cost_delta':tm['tool_search_cost']-sm['tool_search_cost'],'utility_S':sm['objective_utility'],'utility_T':tm['objective_utility'],'utility_delta':tm['objective_utility']-sm['objective_utility'],'branch_S_trace':st,'branch_T_trace':tt,'full_harness_takeover':False,'search_backend':backend})
 args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/'CONTENT_DEDUP_ALWAYS_ON_OFF_PER_STATE.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 summary=[]
 for K in KS:
  rs=[r for r in rows if r['K']==K];
  if not rs:continue
  summary.append({'K':K,'n':len(rs),'first_action_disagreement_rate':sum(r['first_action_disagreement_rate'] for r in rs)/len(rs),'tool_cost_delta':sum(r['tool_cost_delta'] for r in rs)/len(rs),'utility_delta':sum(r['utility_delta'] for r in rs)/len(rs)})
 payload={'component':'content_dedup','contract':'strict Teacher-always-on vs Student-always-off; Reduced continuation; forced first action included in K','summary':summary,'audit':{'rows':len(rows),'full_harness_takeover':0,'source_manifest_sha256':hashlib.sha256((SOURCE/'RUN_MANIFEST.json').read_bytes()).hexdigest(),'normalization':'split_at_first_underscore','smoke':args.smoke}}
 (args.out_dir/'CONTENT_DEDUP_ALWAYS_ON_OFF_SUMMARY.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
