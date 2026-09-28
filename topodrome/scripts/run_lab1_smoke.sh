#!/usr/bin/env bash
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/ssd/xiangkun/.cache/topodrome-uv}"
if [ "${1:-}" = "--model-sharded" ]; then
  export TOPODROME_RUN_LAB1_SHARDED_MODEL_TEST=RUN_SHARDED_MODEL_TEST
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: scripts/run_lab1_smoke.sh [--model-sharded]" >&2
  exit 2
fi
export TOPODROME_RUN_LAB1_GPU_TESTS=RUN_LAB1_GPU_TESTS
exec uv run --extra agent --extra runtime pytest -v \
  tests/integration/test_lab1_model_smoke.py
