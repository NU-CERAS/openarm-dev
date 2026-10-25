set -eu
mkdir -p .work
export TMPDIR="$PWD/.work"
export UV_CACHE_DIR="$PWD/.work/uv-cache"
uv pip install --python "$(command -v python)" --torch-backend cu130 --upgrade torch
uv pip install --python "$(command -v python)" -e '.[dev,agent,rl,planning,video]'
python -m openarm_sim.build_planning_assets
