# Cartesian Console

OpenArm tabletop simulation, feedback-driven hardware synchronization, collision
avoidance and skill learning. The default OMPL planner runs on CPU. cuRobo is an
optional CUDA backend.

Use the [fresh-checkout guide](docs/getting-started.md) for a complete CPU setup,
first launch, feature recording and the physical connection placeholders.

## Install

From the repository root on an Ubuntu 24.04 x86_64 desktop:

```bash
bash projects/cartesian_console/scripts/setup.sh
```

This installs system dependencies and a private Python 3.12 environment with
pinned CPU dependencies, then checks planning, physical execution and rendering.
System package installation requests sudo when required. Repeated setup reuses
the environment. CUDA, PyTorch and ROS are optional.

## Start the tabletop console

```bash
bash projects/cartesian_console/scripts/run_demo.sh
```

The English interface reads both arms at startup and displays their measured
pose in a single native-resolution sim preview. Drag a TCP axis and release to
display the planned target pose. Choose **Execute action** to move the robot.
The preview holds the target during execution, then automatically synchronizes
its posture, TCP handles and coordinates from feedback for the next action.
Coordinates are displayed in centimeters from the fixed torso top center.
Save a reached position for reuse. The demo uses an independent
MuJoCo robot as the controlled plant. Opening the interface starts no motion.
The demo starts both arms at joint zero; physical startup reads the actual arm
feedback. The coordinate fields start at the measured TCP from the torso top.

For an automated check of the console and independent robot windows:

```bash
bash projects/cartesian_console/scripts/check_setup.sh --gui
```

The check exercises both windows, explicit execution and automatic feedback sync,
then closes them. During normal operation, dragging in the console leaves the
independent robot stationary until **Execute action** is selected.

For physical hardware, fill `configs/sync_hardware.example.json` using the
[connection and operating guide](docs/sync-console.md). The current adapter uses
the shared SocketCAN backend and seven motors per arm. Hardware connection
parameters are placeholders until configured for the actual robot.

```bash
bash projects/cartesian_console/scripts/setup.sh --hardware
```

Hardware setup installs the pinned CAN binding and creates configuration
templates under `runs/cartesian_console/sync/`. Fill the actual interfaces and
review both motor configurations before launch. `--supported` confirms both arms
have mechanical support because the current backend releases torque after motion.

After completing those configurations:

```bash
bash projects/cartesian_console/scripts/run_robot.sh --supported
```

## Planning backends

| `--planner` | Execution | Path generation |
| --- | --- | --- |
| `ompl` (default) | CPU | mink IK, RRTConnect obstacle avoidance, MuJoCo collision checking |
| `curobo` | CUDA GPU | cuRobo trajectory optimization and collision checking |
| `mink` | CPU | IK and direct joint interpolation; rejects colliding paths |

```bash
bash projects/cartesian_console/scripts/run_demo.sh --planner ompl
bash projects/cartesian_console/scripts/run_demo.sh --planner mink
```

Use `OPENARM_SOFTWARE_RENDERING=1` before a launcher command to select Mesa.
For cuRobo, activate the environment and run
`bash projects/cartesian_console/scripts/setup_curobo.sh` from the repository root;
installation requires compatible CUDA.

## Other demos and learning

```bash
source runs/cartesian_console/env/bin/activate
openarm-reach --scene pickup
openarm-reach --scene strike
openarm-train-drum --help
openarm-collect-act --help
openarm-train-act --help
openarm-train-pickup-residual --help
openarm-distill-act --help
openarm-train-strike --help
openarm-record-policy --help
```

Install the corresponding extras for these workflows:

```bash
python -m pip install -e './projects/cartesian_console[rl,gpu-sim,tracking,imitation,video,assets]'
```

GPU training requires a compatible CUDA PyTorch installation. Training,
collection and evaluation run serially within the shared compute budget.
Opening a demo starts no training. Checkpoints and datasets are supplied through
CLI paths; generated artifacts are stored under `runs/cartesian_console/`.

| Guide | Purpose |
| --- | --- |
| [Installation and reproduction](docs/getting-started.md) | Setup, launch, checks and policy demonstration reproduction |
| [Synchronization console](docs/sync-console.md) | Hardware placeholders, operator controls, saved targets and recording |
| [Policy deployment](docs/policy-deployment.md) | RL/ACT action decoding, CAN joint/gripper calibration and command replay |

## Verify

```bash
bash projects/cartesian_console/scripts/check_setup.sh
bash projects/cartesian_console/scripts/check_setup.sh --gui
```

The complete project suite additionally requires the learning/CUDA extras.
cuRobo checks require generated planning assets. Asset-specific checks require
the separately imported asset. Physical motor verification is performed on the
actual reviewed hardware configuration.
