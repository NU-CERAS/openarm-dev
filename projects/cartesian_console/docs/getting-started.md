# Start Cartesian Console

The primary tested platform is an Ubuntu 24.04 x86_64 graphical desktop.
The console uses CPU OMPL planning, local MuJoCo preview, and ROS Jazzy execution
and feedback. Camera, CUDA, trained policies and downloaded drum assets are
optional. Run these commands from the repository root in Bash.

## Install

Clone the development branch and its submodules:

```bash
git clone --recurse-submodules --branch main https://github.com/jixinyan/openarm-dev.git
cd openarm-dev
bash setup/linux.sh --console
```

Ubuntu and WSL use the root ROS Jazzy `.venv`. The installer requires Internet
and sudo for system packages. Apple Silicon Mac with macOS 15+ uses:

```bash
bash setup/mac.sh --console
```

The Mac launcher selects the locked Pixi console environment. Native Mac runtime
requires validation on that platform. Windows 11 uses the root WSL installer:

```powershell
powershell -ExecutionPolicy Bypass -File setup/windows.ps1 -Console
```

Run subsequent commands inside that WSL checkout. Ubuntu/WSL requires a working
user systemd session; WSL runtime and USB/CAN access require platform validation.
See [root setup](../../../setup/README.md) for platform details.

For an already activated root ROS environment, run `python setup/console.py`.
Set `OPENARM_PYTHON=/absolute/path/to/python` to select an explicitly activated
ROS interpreter. Launch scripts otherwise select the root environment.

## Open and operate

```bash
bash projects/cartesian_console/scripts/run_demo.sh
```

This starts a ROS execution node with native MuJoCo physics, a separate
measured-feedback window, and the console preview. Closing the console closes
its owned processes. The shared compute guard permits one compute job and
preserves at least 8 GiB of available RAM.
The default OpenDrummer model uses each drumstick's striking tip as its TCP.
Use `--model v1.0` for the gripper/tabletop model.

1. Select an arm.
2. Drag its TCP's red X, green Y or blue Z arrow and release to preview.
3. Choose **Execute action**. The preview retains the target during execution.
4. Completion automatically updates the preview, arrows and coordinates from
   measured feedback. The next plan begins from this measured state.
5. **Esc** cancels dragging. **Stop** requests motion cancellation.
6. Save a reached coordinate under **Saved positions**. Loading a saved position
   creates a preview; **Execute action** executes it.

Coordinates are centimeters from the fixed torso-top reference. X is forward,
Y is left and Z is up. Small drags from the synchronized tip provide convenient
initial reach checks. Coordinates saved for one robot model belong to that model.
The demo starts at joint zero; physical startup reads actual joints using the
reviewed motor direction and zero mappings. The interface requires no camera.

Use Mesa rendering explicitly when needed:

```bash
OPENARM_SOFTWARE_RENDERING=1 bash projects/cartesian_console/scripts/run_demo.sh
```

## Check setup

Close other compute jobs, then run:

```bash
bash projects/cartesian_console/scripts/check_setup.sh
bash projects/cartesian_console/scripts/check_setup.sh --gui
```

The checks perform actual ROS/MuJoCo reaches. The GUI check exercises dragging,
preview, explicit execution, automatic synchronization and saved-position reuse.

## Physical robot

```bash
bash setup/linux.sh --console --hardware
```

Fill the generated `runs/cartesian_console/sync/hardware.json` and reviewed
left/right motor configurations as described in the [operator guide](sync-console.md).
Configure both actual SocketCAN interfaces according to the connected devices.
Mechanically support both arms: the current CAN backend releases motor torque
at completion, cancellation and execution errors. Then launch:

```bash
bash projects/cartesian_console/scripts/run_robot.sh --supported
```

Physical CAN motor execution requires commissioning on the actual robot. The
sim model supplies kinematic coordinates and configured collision geometry;
there is no object perception.

## Additional commands

- `run_sync.sh --target remote --namespace /robot` connects existing compatible ROS endpoints.
- `--planner mink` uses IK/interpolation; default `ompl` performs collision-checked planning.
- `scripts/setup_curobo.sh` installs optional CUDA planning; select `--planner curobo`.
- `openarm-reach` opens camera-image surface selection for reach/kit scenes.
- [RL](../../rl/README.md) provides task previews, ACT/PPO, visual distillation,
  learned-policy recording, reference showcases and policy deployment.
