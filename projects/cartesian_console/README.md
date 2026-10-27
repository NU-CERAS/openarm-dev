# Cartesian Console

OpenArm tabletop preview, ROS 2 robot execution, collision avoidance and skill
learning. The console uses the shared `openarm_tools.ros_execution` interface
for simulation and physical robot commands/feedback. Planning and training use
MuJoCo locally. OMPL runs on CPU; cuRobo is optional.

Use the [fresh-checkout guide](docs/getting-started.md) for a complete CPU setup,
first launch, feature recording and the physical connection placeholders.

## Install

Use the root platform setup with the console profile:

```bash
bash setup/linux.sh --console
```

Ubuntu and WSL use the root ROS Jazzy `.venv`. Windows setup accepts `-Console`.
Apple Silicon uses `bash setup/mac.sh --console` and the locked Pixi `console`
environment; its OMPL wheel requires macOS 15 or newer. Mac/WSL runtime checks
require those actual systems. The project `scripts/setup.sh` delegates to the
root platform installer. A graphical desktop is required for the GUI.

## Start the tabletop console

```bash
bash projects/cartesian_console/scripts/run_demo.sh
```

The English interface subscribes to `/joint_states` at startup and displays both arms' measured
pose in a single native-resolution sim preview. Drag a TCP axis and release to
display the planned target pose. Choose **Execute action** to move the robot.
The preview holds the target during execution, then automatically synchronizes
its posture, TCP handles and coordinates from feedback for the next action.
Coordinates are displayed in centimeters from the fixed torso top center.
Save a reached position for reuse. The demo uses an independent
MuJoCo robot in a ROS execution node. A separate window renders its measured
ROS feedback. Opening the interface starts no motion.
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
the shared SocketCAN backend behind the ROS execution node and seven motors per arm. Hardware connection
parameters are placeholders until configured for the actual robot.

```bash
bash setup/linux.sh --console --hardware
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
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
openarm-reach --scene pickup
openarm-reach --scene strike
```

For learning, activate a separate compatible Python 3.12 environment and install
the corresponding extras:

```bash
python -m pip install -e './projects/cartesian_console[rl,gpu-sim,tracking,imitation,video,assets]'
openarm-train-drum --help
openarm-collect-act --help
openarm-train-act --help
openarm-train-pickup-residual --help
openarm-distill-act --help
openarm-train-strike --help
openarm-record-policy --help
```

GPU training requires a compatible CUDA PyTorch installation. Training,
collection and evaluation run serially within the shared compute budget.
Opening a demo starts no training. Checkpoints and datasets are supplied through
CLI paths; generated artifacts are stored under `runs/cartesian_console/`.
Use a separate compatible environment for CUDA training dependencies.

## ROS execution endpoints

Both execution nodes expose `/{left,right}_joint_trajectory_controller/follow_joint_trajectory`
using `control_msgs/action/FollowJointTrajectory`, and publish measured
`sensor_msgs/JointState` on `/joint_states`. `--namespace` scopes these names.
`--target remote` attaches the console to an already-running compatible robot.

```bash
bash projects/cartesian_console/scripts/run_sync.sh --target remote --namespace /robot
```

The root `openarm_mujoco_sim` ROS-control launch supports
`trajectory_control:=true` to expose the same action endpoints. Use the root
`--build-ros` setup option and source the built workspace before launching it.

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
