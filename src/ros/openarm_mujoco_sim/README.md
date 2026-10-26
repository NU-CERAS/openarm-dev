# OpenArm MuJoCo ROS integration

Preserved Ubuntu/WSL v1.0 integration using `mujoco_ros2_control`. From a checkout without spaces, run `bash setup/linux.sh --build-ros`, then activate ROS and `.venv` and source `install/setup.bash`.

```sh
openarm sim --ros
# In another activated/sourced terminal:
ros2 run openarm_mujoco_sim simple_motion.py --cycles 1
```

Canonical assets live in `robots/v1.0/`; CMake installs them into this package's share directory. Rebuild after asset changes. MuJoCo Python packages on macOS alone do not provide the ROS C++ plugin.
