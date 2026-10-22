#!/usr/bin/env python3
"""Run INSIDE the cloudml container (after pinned pkgs installed) to discover ALL
missing third-party pkgs that load_upstream_modules actually needs.
Mirrors the worker import path: env_bridge.load_upstream_modules -> harness/training/tinker_cookbook.
Recursively pip-installs each ModuleNotFoundError (top-level pkg) and reports the full list."""
import sys, subprocess, importlib, os
sys.path.insert(0,'/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1')
sys.path.insert(0,'/mnt/songzijun/Capability_Evolution/TRIM')
os.environ['HARNESS1_FORBID_CHROMA']='1'
VENDORED = {'tinker_cookbook','harness','datagen','training','trim','scape','easyopd'}
missing=[]
for _ in range(100):
    try:
        importlib.invalidate_caches()
        from trim.upstream_harness1.env_bridge import load_upstream_modules
        mods=load_upstream_modules()
        print("FULL IMPORT OK", flush=True)
        break
    except ModuleNotFoundError as e:
        name=str(e).replace("No module named '","").rstrip("'").split('.')[0]
        if name in VENDORED or name.startswith('tinker_cookbook'):
            print("SKIP vendored/local:", name, flush=True); break
        if name in missing:
            print("ALREADY TRIED:", name, "-> stop", flush=True); break
        missing.append(name)
        print("MISSING:", name, "-> installing", flush=True)
        r=subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check','--no-cache-dir',name],
                         capture_output=True, text=True)
        if r.returncode!=0:
            print("PIP FAIL", name, r.stderr[-400:], flush=True); break
    except Exception as e:
        print("OTHER:", type(e).__name__, str(e)[:400], flush=True); break
print("===DISCOVERY_RESULT===", " ".join(missing), flush=True)
