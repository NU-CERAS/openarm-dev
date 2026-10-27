PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
CONSOLE_ENV="${OPENARM_ENV:-$REPO_DIR/runs/cartesian_console/env}"
if [[ "$CONSOLE_ENV" != /* ]]; then
    CONSOLE_ENV="$REPO_DIR/$CONSOLE_ENV"
fi
PYTHON_BIN="${OPENARM_PYTHON:-$CONSOLE_ENV/bin/python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo 'Run: bash projects/cartesian_console/scripts/setup.sh' >&2
    exit 1
fi
cd -- "$REPO_DIR"
mkdir -p runs/cartesian_console/cache runs/cartesian_console/sync
export TMPDIR="$REPO_DIR/runs/cartesian_console/cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
if [[ "${OPENARM_SOFTWARE_RENDERING:-0}" == 1 ]]; then
    export __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1
fi
