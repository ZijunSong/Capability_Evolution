from pathlib import Path
import sys
sys.path.insert(0,'/mnt/songzijun/Capability_Evolution/SCAPE')
from scripts.run_h100_2_live_fork_replay import _load_queries,_load_qrels,build_searcher,HFContinuationScorer,LiveState,policy_action
from scape.rendering.dual_view import DualViewRenderer
root=Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus')
q=_load_queries(root/'topics-qrels/queries.tsv'); qr=_load_qrels(root/'topics-qrels/qrel_evidence.txt')
searcher,b=build_searcher(Path('/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25'),Path('/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl'))
s=HFContinuationScorer('/mnt/songzijun/models/pat-jj_harness-1-full/harness-1',device='cuda:0',dtype='bfloat16',max_prompt_tokens=4096)
r=DualViewRenderer()
for qid in sorted(set(q)&set(qr))[:4]:
 st=LiveState(qid=qid,query=q[qid],gold=qr[qid],searcher=searcher,component='auto_populate_first_search',branch_seed='diag')
 st.documents=[]; st.curated_ids=[]; st.cost=0
 a,d,_=policy_action(st,s,r,component='auto_populate_first_search',full=False)
 print(qid,a,d.get('tool_name_probs'))
