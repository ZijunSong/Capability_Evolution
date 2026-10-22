"""sitecustomize: per-query wall-clock timeout + pyserini anserini fatjar registration.

Loaded automatically by CPython at interpreter startup when this file's directory is
placed FIRST on PYTHONPATH (ahead of sitecustomize_dir/). Python imports only the first
``sitecustomize`` module it finds, so this file must perform BOTH jobs that the codebase
relies on at startup:

  1. Inject ``query_timeout_s`` into every upstream_api eval worker config, so a single
     long/hung query (e.g. Qwen hermes tool-call parser format_retry death-loop on long
     context, seen at t-20260911222900-owa0l rank2 qid=792) is skipped via
     ``write_infra_error_query`` after the wall-clock limit instead of hanging the whole
     shard forever. The worker (trim/eval/harness1_api_worker.py:77,151) reads
     ``cfg["query_timeout_s"]`` and wraps the per-query coroutine in
     ``asyncio.wait_for(..., timeout=query_timeout_s)``; on TimeoutError it records an
     infra_error for that query and CONTINUES to the next query. Without this, one hung
     query dooms the entire 830-query shard.

     ``run_eval.py`` builds ``worker_extra`` without query_timeout_s (no --query-timeout-s
     CLI arg exists), and ``_worker_config`` merges ``extra`` via ``cfg.update(dict(extra))``
     (harness1_api_launch.py:75-76). We wrap ``_worker_config`` so every worker config gets
     the timeout unless the caller already set one.

  2. Register the pyserini anserini fatjar on jnius_config BEFORE the JVM starts (the same
     job sitecustomize_dir/sitecustomize.py does for the rest of the codebase). pyserini's
     pyserini.pyclass wraps configure_classpath in a bare try/except that silently
     swallows failures -> NoClassDefFoundError on io.anserini.*. jnius_config.add_classpath
     only takes effect before the JVM starts, so we run at startup.

     IMPORTANT: only call jnius_config.add_classpath(); do NOT call get_classpath() or
     vm_running() (those import jnius and freeze/start the JVM at sitecustomize time).

This file is idempotent and a no-op for code paths that never call _worker_config (training,
legacy_local eval, non-upstream_api paths). Default timeout is QUERY_TIMEOUT_S env var if
set, else 600s (10 min, per user 2026-09-12). Set QUERY_TIMEOUT_S=0 to disable.
"""

import glob
import os


def _register_anserini_classpath() -> None:
    try:
        import jnius_config  # type: ignore
    except Exception:
        return  # jnius not installed yet
    jar = os.environ.get("ANSERINI_CLASSPATH")
    if not jar:
        try:
            import pyserini  # type: ignore
            jar_dir = os.path.join(os.path.dirname(pyserini.__file__), "resources", "jars")
            hits = glob.glob(os.path.join(jar_dir, "anserini-*-fatjar.jar"))
            if hits:
                jar = hits[0]
                os.environ["ANSERINI_CLASSPATH"] = jar
        except Exception:
            return
    if not jar or not os.path.isfile(jar):
        return
    try:
        jnius_config.add_classpath(jar)
    except Exception:
        return
    try:
        jnius_config.add_options("--add-modules=jdk.incubator.vector")
    except Exception:
        pass


def _install_query_timeout_patch() -> None:
    raw = os.environ.get("QUERY_TIMEOUT_S")
    if raw is not None:
        try:
            timeout = float(raw)
        except ValueError:
            timeout = 600.0
    else:
        timeout = 600.0
    if timeout <= 0:
        return  # explicitly disabled

    try:
        import trim.eval.harness1_api_launch as _h  # type: ignore
    except Exception:
        return  # module not importable in this interpreter (e.g. pre-pip); nothing to patch

    if getattr(_h, "_qwen_all_recovery_timeout_patched", False):
        return
    _orig_worker_config = _h._worker_config

    def _patched_worker_config(*, extra=None, **kwargs):
        merged = dict(extra) if extra else {}
        merged.setdefault("query_timeout_s", timeout)
        return _orig_worker_config(extra=merged, **kwargs)

    _h._worker_config = _patched_worker_config
    _h._qwen_all_recovery_timeout_patched = True


try:
    _register_anserini_classpath()
except Exception:
    pass

try:
    _install_query_timeout_patch()
except Exception:
    pass
