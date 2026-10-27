# Cartesian Console

OpenArm tabletop simulation, feedback-driven hardware synchronization, collision
avoidance and skill learning. The default OMPL planner runs on CPU. cuRobo is an
optional CUDA backend.

Use the [fresh-checkout guide](docs/getting-started.md) for a complete CPU setup,
first launch, feature recording and the physical connection placeholders.

## Install

From the repository root, using Python 3.12 or newer:

```bash
mkdir -p runs/cartesian_console/cache
export TMPDIR="$PWD/runs/cartesian_console/cache"
python -m pip install -e . -e './projects/cartesian_console[dev]'
```

This installs MuJoCo, mink IK and OMPL. CUDA, PyTorch and ROS are optional.
`environment.yml` provides the equivalent conda environment.

## Start the tabletop console

```bash
OPENARM_PYTHON="$(command -v python)" bash projects/cartesian_console/scripts/run_sync.sh
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

For testing with a separate window showing the independently simulated robot:

```bash
bash projects/cartesian_console/scripts/run_sync.sh --show-plant
```

That MuJoCo window displays the controlled robot receiving joint commands.
Dragging in the console leaves it stationary until **Execute action** is selected.

For physical hardware, fill `configs/sync_hardware.example.json` using the
[connection and operating guide](docs/sync-console.md). The current adapter uses
the shared SocketCAN backend and seven motors per arm. Hardware connection
parameters are placeholders until configured for the actual robot.

## Planning backends

| `--planner` | Execution | Path generation |
| --- | --- | --- |
| `ompl` (default) | CPU | mink IK, RRTConnect obstacle avoidance, MuJoCo collision checking |
| `curobo` | CUDA GPU | cuRobo trajectory optimization and collision checking |
| `mink` | CPU | IK and direct joint interpolation; rejects colliding paths |

```bash
openarm-sync --planner ompl
openarm-sync --planner curobo
openarm-reach --planner ompl
```

See [motion planning](docs/motion-planning.md) for backend installation,
software rendering and collision behavior.

## Other demos and learning

```bash
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
| [Synchronization console](docs/sync-console.md) | Hardware placeholders, operator controls, saved targets and recording |
| [Motion planning](docs/motion-planning.md) | CPU/GPU backend setup and execution behavior |
| [Simulation demos](docs/simulation-demo.md) | Camera controls, pickup, strike and VLM tool usage |
| [Electronic drum asset](docs/electronic-drum-scene.md) | Import and display an independently downloaded asset |
| [GPU pickup training](docs/gpu-pickup-teacher.md) | MuJoCo Warp environment and teacher commands |
| [Pickup reward](docs/pickup-reward.md) | Reward terms and task success conditions |
| [Compute resources](docs/compute-resources.md) | Resource limits, recovery and supervision |
| [Policy deployment](docs/policy-deployment.md) | RL/ACT action decoding, CAN joint/gripper calibration and command replay |

## Source directories

| Location | Responsibility |
| --- | --- |
| `src/openarm_sim/console/` | Feedback transport, synchronization UI and demo recording |
| `src/openarm_sim/planning/` | OMPL, cuRobo and exported planning assets |
| `src/openarm_sim/training/` | Training, evaluation, pipeline and W&B entry points |
| `src/openarm_sim/deployment/` | Policy action processing, continuous CAN execution and trace replay |
| `src/openarm_sim/` | Shared task environments, observations, ACT models and execution interfaces |
| `configs/` | Operator connection templates and task settings |
| `scripts/` | Unified startup and optional cuRobo installation |
| `tests/` | Actual physics, collision, execution and integration checks |
| Repository `src/openarm_tools/` | Shared robot configuration and hardware backends |
| Repository `robots/` | Canonical models, meshes and actuator settings |
| Repository `runs/cartesian_console/` | Local assets, datasets, checkpoints, recordings and output |

The robot model and joint order come from `openarm_tools.robot.config`.
`OPENARM_ROOT` selects an explicit checkout. Keep generated output and local
research notes under `runs/`; commit source, configuration templates and usage
documentation.

## Verify

```bash
CUDA_VISIBLE_DEVICES="" __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  python -m pytest -q projects/cartesian_console/tests/test_ompl.py \
  projects/cartesian_console/tests/test_sync.py
```

The complete project suite additionally requires the learning/CUDA extras.
cuRobo checks require generated planning assets. Asset-specific checks require
the separately imported asset. Physical motor verification is performed on the
actual reviewed hardware configuration.
