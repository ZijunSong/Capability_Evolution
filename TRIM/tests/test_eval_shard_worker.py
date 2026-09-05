from __future__ import annotations

import inspect

from trim.eval import eval_shard_worker


def test_vllm_eval_inits_lucene_on_main_after_client_start():
    src = inspect.getsource(eval_shard_worker._run_vllm)
    prep = src[src.index("def cpu_prep") : src.index("keepalive = GpuKeepAlive")]
    assert "open_retrieval" not in prep
    assert src.index("client.start()") < src.index("open_retrieval(formal")
