#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash setup/mac.sh [--help] (Apple Silicon; ROS + MuJoCo + Pinocchio via Pixi)'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'Unknown arguments; use --help.' >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo 'This manifest supports Apple Silicon macOS. Intel Macs need a separately resolved manifest.' >&2; exit 1;
}
# Locate an existing installation without changing shell startup files.
if command -v pixi >/dev/null 2>&1; then
  PIXI="$(command -v pixi)"
elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI="$HOME/.pixi/bin/pixi"
else
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 https://pixi.sh/install.sh -o "$installer"
  PIXI_NO_PATH_UPDATE=1 bash "$installer"
  PIXI="$HOME/.pixi/bin/pixi"
fi
cd "$ROOT"
git submodule update --init --recursive
"$PIXI" install --manifest-path "$ROOT/dependencies/mac/pixi.toml" --locked
"$PIXI" run --manifest-path "$ROOT/dependencies/mac/pixi.toml" openarm doctor --profile ros
printf '\nReady. From the repo root:\n  %q shell --manifest-path dependencies/mac/pixi.toml\n  mjpython scripts/sim.py\n' "$PIXI"
echo 'SocketCAN is Linux-only. The optional mujoco_ros2_control package is built/tested separately on Ubuntu.'
