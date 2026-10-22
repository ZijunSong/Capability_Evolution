"""sitecustomize: register pyserini anserini fatjar on jnius_config BEFORE JVM start.

Loaded automatically by CPython at interpreter startup (this directory is on PYTHONPATH).
pyserini's pyserini.pyclass wraps configure_classpath in a bare try/except that silently
swallows failures, so the anserini fatjar may never reach the JVM classpath ->
NoClassDefFoundError on io.anserini.*. jnius_config.add_classpath only takes effect
before the JVM starts, so we run at startup, before any training import.

IMPORTANT: we ONLY call jnius_config.add_classpath(). We must NOT call
jnius_config.get_classpath() or vm_running(): in recent pyjnius those trigger
``import jnius`` (via importlib.resources.files('jnius')) which starts the JVM at
sitecustomize time and freezes the classpath. add_classpath() is a pure list append
that does not import jnius. Everything is wrapped so sitecustomize never breaks startup.
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
    # add_classpath is a pure list append; it does NOT import jnius or start the JVM.
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
    pass
