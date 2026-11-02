#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p runs/rl/cache
export TMPDIR="$ROOT/runs/rl/cache"
PYTHON_BIN="${OPENARM_PYTHON:-python}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate a Python 3.12 environment"'
"$PYTHON_BIN" -m pip install -e . -e projects/cartesian_console -e projects/opendrummer_playground -e 'projects/rl[dev,tracking]'
"$PYTHON_BIN" -m pip check
echo 'Ready: bash projects/rl/scripts/run_strike.sh --output runs/rl/strike_teacher --video'
