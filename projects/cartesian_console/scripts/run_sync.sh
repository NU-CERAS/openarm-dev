#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="${OPENARM_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Set OPENARM_PYTHON to the Python executable with openarm-sim installed." >&2
    exit 1
fi
cd -- "$REPO_DIR"
mkdir -p runs/cartesian_console/cache runs/cartesian_console/sync
export TMPDIR="$REPO_DIR/runs/cartesian_console/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
exec "$PYTHON_BIN" -u -m openarm_sim.console.app "$@"
