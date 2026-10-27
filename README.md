# OpenArm Development

CERAS @ Northwestern University's shared foundation for OpenArm simulation, controls, and creative robotics.

## Get started

Clone with `git clone --recurse-submodules <repository-url>` and enter the checkout.

| Machine | Setup | Open the simulator afterward |
|---|---|---|
| Apple Silicon Mac | `bash setup/mac.sh` | Activate Pixi below, then `mjpython scripts/sim.py` |
| Ubuntu 24.04 | `bash setup/linux.sh` | Activate ROS and `.venv` below, then `openarm sim` |
| Windows 11 + WSL2 | `powershell -ExecutionPolicy Bypass -File setup/windows.ps1` | Open the WSL checkout, activate ROS and `.venv`, then `openarm sim` |

Mac terminal activation (use `~/.pixi/bin/pixi` if it is not on PATH):

```sh
pixi shell --manifest-path dependencies/mac/pixi.toml
openarm doctor --profile ros
mjpython scripts/sim.py
```

Ubuntu / WSL terminal activation:

```sh
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
openarm doctor --profile ros
openarm sim
```

Setup installs ROS Jazzy, MuJoCo, Pinocchio and shared tools. Windows runs ROS and MuJoCo together **inside WSL**, not as separate Windows/WSL processes. SocketCAN hardware access requires Linux and a compatible adapter. See [setup](setup/README.md) for optional ROS-control builds and CAN bindings.

For the [Cartesian Console](projects/cartesian_console/README.md), add `--console`
to the Linux/Mac installer or `-Console` to Windows setup. Then run
`bash projects/cartesian_console/scripts/run_demo.sh`. Simulation and physical
execution use the shared ROS 2 trajectory-action and joint-state interface.

## Common commands

Simulation is the default. Angles and increments are radians. Commands work from any directory after editable installation.

```sh
openarm status
openarm zero                         # preview the software zero reference
openarm zero --execute               # record current joint positions as zero
openarm home                        # preview return to recorded zero/default home
openarm home --execute --duration 10
openarm jog openarm_left_joint1 0.05 --execute
openarm pose save ready
openarm pose move ready --execute
openarm sim --headless --seconds 1
```

The equivalents in `scripts/` call the same shared implementation. Use `--help` after a command for options. Hardware instructions and release behavior are in [docs/hardware.md](docs/hardware.md). `can-check` is a hardware diagnostic; it never enables motors.

## Structure

- `dependencies/`: environment manifests and pinned upstream revisions.
- `setup/`: platform installers; Windows delegates to the Ubuntu installer.
- `scripts/`: common command entry points.
- `src/openarm_tools/`: shared configuration, backends, motion, calibration, recording.
- `src/ros/`: ROS integration; `src/vendor/`: upstream sources.
- `robots/`: explicitly versioned models, joint configuration, and reviewed poses.
- `projects/`: independent clients of the shared tools. [Cartesian Console](projects/cartesian_console/README.md) contains camera reach, drumming simulation, and ACT/RL skill learning; other project directories contain starting points.
- `examples/`: small executable tutorials; `mechanical/`: CAD/documentation.
- `docs/`: architecture and operation; `tests/`: behavior checks.
- `runs/`: local simulation snapshots, calibration and saved poses; ignored by Git.

Read [architecture](docs/architecture.md) before adding a backend or changing coordinate conventions.

## Validation

```sh
python -m unittest discover -s tests -v
```

Physical arm commissioning and platform setup verification are separate from simulation tests. The shipped hardware mappings/gains are templates, not validated calibrations for your particular arms.
