#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROS=0
HARDWARE=0
for arg in "$@"; do
  case "$arg" in
    --build-ros) BUILD_ROS=1 ;;
    --hardware) HARDWARE=1 ;;
    --help) echo 'Usage: bash setup/linux.sh [--build-ros] [--hardware]'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux ]] || { echo 'Requires Ubuntu 24.04 (native or WSL).' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'Requires Ubuntu 24.04.' >&2; exit 1; }
if [[ $EUID -eq 0 ]]; then
  echo 'Run as your normal user; the script uses sudo only for system packages.' >&2; exit 1
fi
sudo apt-get update
sudo apt-get install -y curl ca-certificates software-properties-common python3
sudo add-apt-repository -y universe
# Official ROS apt-source package manages keys and repository configuration.
if ! dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | grep -q 'install ok installed'; then
  release="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
  temp="$(mktemp -d)"
  trap 'rm -rf "$temp"' EXIT
  curl -fsSL "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${release}/ros2-apt-source_${release}.noble_all.deb" -o "$temp/ros2-apt-source.deb"
  sudo dpkg -i "$temp/ros2-apt-source.deb"
fi
sudo apt-get update
packages=()
while IFS= read -r package; do
  [[ -z "$package" || "$package" == \#* ]] || packages+=("$package")
done < "$ROOT/dependencies/ubuntu/apt.txt"
sudo apt-get install -y "${packages[@]}"
cd "$ROOT"
git submodule update --init --recursive
# ROS Python extensions need the Ubuntu system Python and its apt packages.
if [[ -d .venv ]]; then
  [[ -x .venv/bin/python ]] || { echo 'Existing .venv is unusable; move it aside and rerun.' >&2; exit 1; }
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Existing venv must use Ubuntu Python 3.12"'
fi
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r dependencies/ubuntu/requirements.txt
.venv/bin/python -m pip install --no-deps -e .
# ROS setup scripts can reference unset variables.
set +u
source /opt/ros/jazzy/setup.bash
set -u
if [[ "$HARDWARE" == 1 ]]; then
  # Import pinned optional CAN source only; never overwrite the existing submodules.
  python3 - <<'PY' | vcs import src/vendor
import yaml
with open('dependencies/upstream.repos') as f:
    repos = yaml.safe_load(f)['repositories']
print(yaml.safe_dump({'repositories': {'openarm_can': repos['openarm_can']}}))
PY
  expected="$(python3 -c 'import yaml; print(yaml.safe_load(open("dependencies/upstream.repos"))["repositories"]["openarm_can"]["version"])')"
  [[ "$(git -C src/vendor/openarm_can rev-parse HEAD)" == "$expected" ]] || {
    echo 'Existing openarm_can checkout differs from dependency pin; reconcile it before installing.' >&2; exit 1;
  }
  .venv/bin/python -m pip install ./src/vendor/openarm_can/python
fi
if [[ "$BUILD_ROS" == 1 ]]; then
  # Build only the simulation's required local packages, not hardware/MoveIt.
  # CMake/colcon builds do not reliably support whitespace in workspace paths.
  [[ "$ROOT" != *[[:space:]]* ]] || { echo 'Use a path without spaces for --build-ros, e.g. ~/dev/openarm-dev.' >&2; exit 1; }
  colcon build --symlink-install --base-paths src/vendor/openarm_description src/ros/openarm_mujoco_sim
fi
.venv/bin/openarm doctor --profile ros
printf '\nReady. In each terminal:\n  source /opt/ros/jazzy/setup.bash\n  source .venv/bin/activate\n  openarm sim\n'
if [[ "$BUILD_ROS" == 1 ]]; then
  echo 'For ROS control: source install/setup.bash && openarm sim --ros'
fi
echo 'CAN utilities installed; adapter drivers/interface configuration are hardware-specific. See docs/hardware.md.'
