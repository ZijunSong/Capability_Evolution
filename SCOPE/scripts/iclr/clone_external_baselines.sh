#!/usr/bin/env bash
# Clone and lock SEED / OPID / SDAR. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p external/baselines

# Prefer local Clash SOCKS if present and no proxy already set.
if [[ -z "${ALL_PROXY:-}${HTTPS_PROXY:-}${https_proxy:-}" ]]; then
  if curl -sS -o /dev/null --connect-timeout 2 -x socks5h://127.0.0.1:7891 https://api.github.com 2>/dev/null; then
    export ALL_PROXY=socks5h://127.0.0.1:7891
    export HTTPS_PROXY=socks5h://127.0.0.1:7891
    export HTTP_PROXY=socks5h://127.0.0.1:7891
    echo "using proxy: socks5h://127.0.0.1:7891"
  fi
fi

GIT_PROXY_ARGS=()
if [[ -n "${ALL_PROXY:-}${HTTPS_PROXY:-}${https_proxy:-}" ]]; then
  proxy="${ALL_PROXY:-${HTTPS_PROXY:-$https_proxy}}"
  GIT_PROXY_ARGS=(-c "http.proxy=$proxy" -c "https.proxy=$proxy")
fi

clone_one() {
  local name="$1" url="$2"
  if [[ -d "external/baselines/$name/.git" ]] && [[ -n "$(ls -A "external/baselines/$name" | grep -v '^\.git$' || true)" ]]; then
    echo "exists: $name"
    return 0
  fi
  rm -rf "external/baselines/$name"
  git "${GIT_PROXY_ARGS[@]}" clone --depth 1 "$url" "external/baselines/$name"
}

clone_one SEED https://github.com/jinyangwu/SEED.git
clone_one OPID https://github.com/jinyangwu/OPID.git
clone_one SDAR https://github.com/ZJU-REAL/SDAR.git

{
  for repo in SEED OPID SDAR; do
    (
      cd "external/baselines/$repo"
      printf "%s\t%s\t%s\n" \
        "$repo" \
        "$(git rev-parse HEAD)" \
        "$(git remote get-url origin)"
    )
  done
} | tee experiments/baselines/BASELINE_LOCK.tsv

echo "locked -> experiments/baselines/BASELINE_LOCK.tsv"
