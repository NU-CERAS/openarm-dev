#!/usr/bin/env bash
set -eo pipefail
cd /workspaces/openarm-dev
source /opt/ros/jazzy/setup.bash
mkdir -p src/vendor
vcs import --skip-existing src/vendor < workspace.repos
sudo rosdep init 2>/dev/null || test -f /etc/ros/rosdep/sources.list.d/20-default.list
rosdep update --rosdistro jazzy
sudo apt-get update
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y --skip-keys 'openarm_hardware openarm_can'
colcon build --symlink-install --packages-skip openarm_hardware openarm --parallel-workers 2
bash scripts/verify.sh
