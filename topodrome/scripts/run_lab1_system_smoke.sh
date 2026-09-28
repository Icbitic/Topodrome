#!/usr/bin/env bash
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/ssd/xiangkun/.cache/topodrome-uv}"
if [ "${1:-}" = "--model-sharded" ]; then
  export TOPODROME_RUN_LAB1_SYSTEM_SHARDED_E2E=RUN_LAB1_SYSTEM_SHARDED_E2E
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: scripts/run_lab1_system_smoke.sh [--model-sharded]" >&2
  exit 2
fi
export TOPODROME_RUN_LAB1_SYSTEM_E2E=RUN_LAB1_SYSTEM_E2E
exec uv run --extra scheduler --extra agent --extra runtime pytest -v \
  tests/integration/test_lab1_system_e2e.py
