"""Sitecustomize: register pyserini anserini fatjar on jnius_config BEFORE the JVM starts.

Why: pyserini's ``pyserini.pyclass`` wraps ``configure_classpath`` in a bare ``try/except``
that silently swallows failures (e.g. when ``jnius_config.add_options`` raises), so the
anserini fatjar never lands on the JVM classpath. The first ``autoclass('io.anserini.*')``
then raises ``NoClassDefFoundError``. ``jnius_config.add_classpath`` only takes effect
*before* the JVM is started, so we must run this at interpreter startup, before any
training code imports jnius/pyserini.

This module is loaded as ``sitecustomize`` (put its directory on PYTHONPATH). It is
idempotent and a no-op if the fatjar or jnius_config cannot be found.
"""

import glob
import os
import sys


def _register_anserini_classpath() -> None:
    try:
        import jnius_config  # type: ignore
    except Exception:
        return  # jnius not installed yet; nothing to do
    # If the JVM is already up we can no longer add to the classpath.
    if hasattr(jnius_config, "vm_running") and jnius_config.vm_running():
        return
    jar = os.environ.get("ANSERINI_CLASSPATH")
    if not jar:
        # Locate the fatjar bundled inside the pyserini package.
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
        cp = jnius_config.get_classpath()
    except Exception:
        cp = None
    if cp and jar in (cp if isinstance(cp, (list, tuple)) else [cp]):
        return  # already registered
    try:
        jnius_config.add_classpath(jar)
    except Exception:
        return
    try:
        jnius_config.add_options("--add-modules=jdk.incubator.vector")
    except Exception:
        pass


try:
    _register_anserini_classpath()
except Exception:
    # sitecustomize must never break interpreter startup.
    pass
