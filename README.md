# openarm_mujoco_sim

Documentation of how we got the two-arm OpenArm v1.0 running in MuJoCo with ROS 2 Jazzy
(`mujoco_ros2_control`), on Windows 11 + WSL2 Ubuntu 24.04.

## Setup

1. **Ubuntu:** in PowerShell, `wsl --install -d Ubuntu-24.04`. Do the rest in Ubuntu.
2. **ROS 2 Jazzy:** install per the [official guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), then:
   ```bash
   sudo apt install -y ros-jazzy-desktop ros-jazzy-xacro ros-jazzy-ros2-control \
     ros-jazzy-ros2-controllers ros-jazzy-mujoco-ros2-control python3-colcon-common-extensions python3-vcstool
   ```
3. **Build** (keep the workspace under `~`, not `/mnt/c`):
   ```bash
   mkdir -p ~/openarm_ws/src && cd ~/openarm_ws/src
   git clone https://github.com/NU-CERAS/openarm-ros2-mujoco-installation.git openarm_mujoco_sim
   vcs import . < openarm_mujoco_sim/openarm_mujoco_sim.repos
   cd ~/openarm_ws && source /opt/ros/jazzy/setup.bash
   colcon build --symlink-install && source install/setup.bash
   ```
4. **GPU (optional):** if `GALLIUM_DRIVER=d3d12 glxinfo -B | grep renderer` shows your GPU, add
   `export GALLIUM_DRIVER=d3d12` to `~/.bashrc`.

## Run

```bash
ros2 launch openarm_mujoco_sim openarm_mujoco.launch.py      # terminal 1
ros2 run openarm_mujoco_sim simple_motion.py                 # terminal 2: looping test motion
```

Manual commands (7 joint angles in radians; gripper 0.0 closed to 0.044 open):

```bash
ros2 topic pub --once /left_forward_position_controller/commands std_msgs/msg/Float64MultiArray \
  "{data: [-0.5, -0.3, 0.0, 1.2, 0.0, 0.0, 0.0]}"
ros2 topic pub --once /left_gripper_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0]}"
```

## Notes

- Joint names must match across the URDF `<ros2_control>` block, `mjcf/openarm_bimanual.xml` and `config/controllers.yaml`.
- `auto_register_cameras=false` and the light `scene.xml` avoid crashes in the WSL D3D12 GPU driver.
  If it still crashes (`D3D12: Removing Device`), `unset GALLIUM_DRIVER` and relaunch.
- MJCF and meshes are from [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) `v1/` (Apache-2.0, see NOTICE).
