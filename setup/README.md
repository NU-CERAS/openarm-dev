# Setup

Run setup from any working directory. Scripts install environments but never enable motors, configure a CAN interface, or alter shell startup files. Rerunning reuses the checkout/environment. `--help` on the shell installers is read-only.

## macOS (Apple Silicon)

`bash setup/mac.sh` installs Pixi if needed, resolves the committed lock, initializes upstream submodules, then checks ROS/Python imports and steps MuJoCo. Activate with `pixi shell --manifest-path dependencies/mac/pixi.toml` from the root. Use `mjpython scripts/sim.py` for the GUI. The manifest targets `osx-arm64`; Intel Macs are not yet covered by a tested lock.

For VS Code, select `dependencies/mac/.pixi/envs/default/bin/python` as the interpreter; the Run MuJoCo task uses its matching `mjpython`.

Native MuJoCo plus ROS Python is supported by this environment. The optional `mujoco_ros2_control` C++ integration is not installed/built on macOS by this script. Use the Ubuntu workflow for that integration. SocketCAN cannot be installed natively on macOS.

## Ubuntu 24.04

`bash setup/linux.sh` installs apt dependencies and an editable Python package into `.venv` using Ubuntu Python 3.12 with system site-packages. It sources ROS for its checks; activate it explicitly in new terminals as shown in the root README. Do not reuse a macOS or incompatible Python environment; move that `.venv` aside first.

Optional flags:

- `--build-ros`: build `openarm_description` and `openarm_mujoco_sim`. Requires a checkout path without spaces. Then `source install/setup.bash` and `openarm sim --ros`.
- `--hardware`: fetch the pinned OpenArm CAN source and build its Python bindings. CAN utilities are installed by the base setup, but adapter/kernel configuration is separate.

Both flags can be combined. ROS apt packages track the Jazzy repository; see `dependencies/README.md` for version guarantees.

## Windows / WSL2

Run `setup/windows.ps1` in PowerShell. It installs Ubuntu-24.04 if absent; complete first-run username setup and any requested reboot, then rerun. Existing WSL must be installed and accessible through `wsl.exe`.

The script uses a separate clone at `~/dev/openarm-dev` inside WSL and runs `setup/linux.sh`. It does not copy uncommitted Windows changes, reset existing work, or automatically pull newer commits. Push the reorganization first, or place this checkout in WSL yourself. Options: `-RepoPath`, `-Distro`, `-BuildRos`, `-Hardware`. Default distro must be Ubuntu 24.04. WSLg provides the Linux GUI on Windows 11.

For a checkout already inside WSL, simply run `bash setup/linux.sh` there. USB CAN access in WSL requires USB passthrough and a suitable kernel driver; `--hardware` alone cannot guarantee adapter access. A native Ubuntu robot host is the simplest initial hardware target.

## Sources

- [ROS Jazzy Ubuntu packages](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [RoboStack](https://robostack.github.io/GettingStarted.html)
- [MuJoCo Python and macOS viewer](https://mujoco.readthedocs.io/en/stable/python.html)
- [OpenArm CAN](https://github.com/enactic/openarm_can)
