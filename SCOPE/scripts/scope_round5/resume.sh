#!/usr/bin/env bash
# Resume = 启动/恢复 pipeline supervisor
exec bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/scope_round5/run_nohup.sh" "$@"
