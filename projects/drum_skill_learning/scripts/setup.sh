#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p runs/rl/cache
export TMPDIR="$ROOT/runs/rl/cache"
PYTHON_BIN="${OPENARM_PYTHON:-$ROOT/.venv/bin/python}"
MODE="${1:---gpu}"
if [[ "$MODE" != --gpu && "$MODE" != --cpu ]]; then
  echo 'Usage: setup.sh [--gpu|--cpu]. Use OPENARM_PYTHON to select an activated Python 3.12 environment.' >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate a Python 3.12 environment"'
if [[ "$MODE" == --cpu ]]; then
  "$PYTHON_BIN" -m pip install 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cpu
else
  "$PYTHON_BIN" -m pip install 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cu130
fi
"$PYTHON_BIN" -m pip install -e . -e projects/cartesian_console -e projects/opendrummer_playground -e 'projects/drum_skill_learning[dev,tracking,imitation]'
"$PYTHON_BIN" -m pip check
echo 'Ready: bash projects/drum_skill_learning/scripts/run_training.sh --help'
