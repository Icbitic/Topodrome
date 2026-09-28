#!/usr/bin/env bash
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/ssd/xiangkun/.cache/topodrome-uv}"
run_model=0
run_sharded=0
case "${1:-}" in
--with-model)
  run_model=1
  shift
  ;;
--model-sharded)
  run_model=1
  run_sharded=1
  shift
  ;;
esac
if [ "$#" -ne 0 ]; then
  echo "usage: scripts/run_remote_jupyter_smoke.sh [--with-model|--model-sharded]" >&2
  exit 2
fi

export TOPODROME_RUN_REMOTE_JUPYTER_TESTS=RUN_REMOTE_JUPYTER_TESTS
if [ "$run_model" -eq 1 ]; then
  export TOPODROME_RUN_REMOTE_JUPYTER_MODEL_TESTS=RUN_REMOTE_MODEL_TEST
fi
if [ "$run_sharded" -eq 1 ]; then
  export TOPODROME_REMOTE_JUPYTER_MODEL_GPU_COUNT=2
fi
exec uv run --extra agent --extra runtime --extra jupyter pytest -v \
  tests/integration/test_remote_jupyter.py
