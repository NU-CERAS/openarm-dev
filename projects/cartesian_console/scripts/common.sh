PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd -- "$PROJECT_DIR/../.." && pwd)"
if [[ "$(uname -s)" == Darwin && -z "${OPENARM_PYTHON:-}" ]]; then
    if [[ "${PIXI_ENVIRONMENT_NAME:-}" != console ]]; then
        PIXI="${PIXI_EXE:-$HOME/.pixi/bin/pixi}"
        if command -v pixi >/dev/null 2>&1; then PIXI="$(command -v pixi)"; fi
        exec "$PIXI" run --manifest-path "$REPO_DIR/dependencies/mac/pixi.toml" --environment console \
          bash "${BASH_SOURCE[1]}" "$@"
    fi
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    if [[ -z "${OPENARM_PYTHON:-}" && -f /opt/ros/jazzy/setup.bash ]]; then
        set +u
        source /opt/ros/jazzy/setup.bash
        set -u
    fi
    PYTHON_BIN="${OPENARM_PYTHON:-$REPO_DIR/.venv/bin/python}"
fi
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
