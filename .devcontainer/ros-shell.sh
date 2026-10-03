source /opt/ros/jazzy/setup.bash
if [ -f /workspaces/openarm-dev/install/setup.bash ]; then
  source /workspaces/openarm-dev/install/setup.bash
fi
alias cb='cd /workspaces/openarm-dev && colcon build --symlink-install --packages-skip openarm_hardware openarm --parallel-workers 2'
alias cs='source /workspaces/openarm-dev/install/setup.bash'
