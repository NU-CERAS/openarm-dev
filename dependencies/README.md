# Dependencies

This folder contains dependency definitions, not installed libraries. Project code imports `openarm_tools`; never copy environments between operating systems.

- `mac/pixi.toml` and `pixi.lock`: ROS Jazzy, MuJoCo, Pinocchio and tools for Apple Silicon. Re-resolve intentionally with `pixi lock --manifest-path dependencies/mac/pixi.toml`; commit both files.
- `ubuntu/apt.txt`: Ubuntu 24.04 + ROS Jazzy packages for native Linux and WSL. Apt patch versions follow the configured repositories; this is not a fully locked OS image.
- `ubuntu/requirements.txt`: Python compatibility ranges (NumPy stays below 2 for Ubuntu ROS binaries). Installed using system Python in `.venv`.
- `upstream.repos`: exact source revisions. Existing description/ROS sources remain Git submodules, pinned by Git. Update their gitlinks and this manifest together. The optional CAN source is fetched into ignored `src/vendor/openarm_can/` by setup.

ROS uses RoboStack on macOS and apt on Ubuntu. Do not mix these two ROS installations in one shell. Pinocchio comes from conda-forge on Mac and ROS apt on Ubuntu. Native macOS CAN is unsupported; Windows hardware support depends on WSL USB/kernel support.
