#!/usr/bin/env bash
# 兼容入口 — 转发到 pipeline supervisor
exec bash "$(dirname "$0")/pipeline_supervisor.sh" "$@"
