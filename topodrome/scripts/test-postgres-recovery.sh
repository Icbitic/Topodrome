#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

exec uv run --group postgres pytest -m postgres tests/integration/test_postgres_recovery.py "$@"
