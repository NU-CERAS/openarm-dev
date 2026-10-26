set -eu
cd "$(dirname "$0")/.."
mkdir -p ../../runs/cartesian_console/cache
export TMPDIR="$PWD/../../runs/cartesian_console/cache"
export UV_CACHE_DIR="$TMPDIR/uv-cache"
uv pip install --python "$(command -v python)" --torch-backend cu130 --upgrade torch
uv pip install --python "$(command -v python)" -e ../.. -e '.[dev,agent,rl,planning,video,assets]'
python -m openarm_sim.planning.assets
