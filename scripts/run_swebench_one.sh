#!/usr/bin/env bash
# Run exactly one Verified instance through EvoMAS and the local evaluator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
exec conda run -n "${CONDA_ENV:-mas}" --no-capture-output python -u "$SCRIPT_DIR/swebench_one.py" "$@"
