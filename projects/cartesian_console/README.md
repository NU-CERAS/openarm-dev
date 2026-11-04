# Cartesian Console

OpenArm and OpenDrummer TCP preview, ROS 2 robot execution and collision avoidance.
The console uses the shared `openarm_tools.ros_execution` interface for simulation
and physical robot commands/feedback. Planning uses MuJoCo locally.
OMPL runs on CPU; cuRobo is optional. Learning lives in [Drum Skill Learning](../drum_skill_learning/README.md).

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

## Start the OpenDrummer console

```bash
bash projects/cartesian_console/scripts/run_demo.sh
```

The default model is OpenDrummer with its fixed drumsticks and shared drum scene.
The TCP is the striking end of each drumstick (`left_stick_tip` and
`right_stick_tip`), derived from the source mesh and URDF transforms. Dragging,
IK, collision checking, preview error and feedback synchronization use this same
point. The torso-top reference is derived from the fixed chassis mesh of the
selected model. The v1.0 gripper model remains available with `--model v1.0`;
its TCP remains at the hand.

**Annotate drum layout** opens a separate panel for torso-relative drum-center
coordinates, dragging drum proxies, registering a center at the measured tip
and capturing measured ready joints. Save to `--layout PATH`; the owned MuJoCo
backend refreshes the layout while idle. See the
[annotation and drum skill guide](../drum_skill_learning/docs/camera-free-drumming.md)
for the camera-free registration, training and ROS policy workflow.

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

For OpenDrummer cuRobo assets, use the model-specific export:

```bash
openarm-build-planning-assets --model opendrummer
bash projects/cartesian_console/scripts/run_demo.sh --planner curobo
```

ROS uses the shared seven-joint semantic names `openarm_left_joint1..7` and
`openarm_right_joint1..7`. The OpenDrummer simulation maps these in order to
`lj1..7` and `rj1..7`. Physical ROS drivers must implement a reviewed mapping
of joint order, direction, zero and limits for the selected model. The direct
CAN launcher currently uses the reviewed v1.0 configuration and selects that
model explicitly. OpenDrummer hardware execution has not been validated.

```bash
bash projects/cartesian_console/scripts/run_demo.sh --planner ompl
bash projects/cartesian_console/scripts/run_demo.sh --planner mink
```

Use `OPENARM_SOFTWARE_RENDERING=1` before a launcher command to select Mesa.
For cuRobo, activate the environment and run
`bash projects/cartesian_console/scripts/setup_curobo.sh` from the repository root;
installation requires compatible CUDA.

## Camera reach and learning

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
openarm-reach
```

Task environments, ACT/PPO, visual distillation, policy recording and policy
deployment are installed through [Drum Skill Learning](../drum_skill_learning/README.md). That project
uses these simulation and motion interfaces. The console requires no learning
package or PyTorch installation.

## ROS execution endpoints

Both execution nodes expose `/{left,right}_joint_trajectory_controller/follow_joint_trajectory`
using `control_msgs/action/FollowJointTrajectory`, and publish measured
`sensor_msgs/JointState` on `/joint_states`. `--namespace` scopes these names.
`--target remote` attaches the console to an already-running compatible robot.

```bash
bash projects/cartesian_console/scripts/run_sync.sh --target remote --model opendrummer --namespace /robot
```

The root `openarm_mujoco_sim` ROS-control launch supports
`trajectory_control:=true` to expose the same action endpoints. Use the root
`--build-ros` setup option and source the built workspace before launching it.
Select `--model v1.0` when connecting to that v1.0 simulation. The selected
console model must match the backend's physical kinematics and tool mounting.

| Guide | Purpose |
| --- | --- |
| [Installation and reproduction](docs/getting-started.md) | Setup, launch, checks and policy demonstration reproduction |
| [Synchronization console](docs/sync-console.md) | Hardware placeholders, operator controls, saved targets and recording |
| [Drum skill learning and deployment](../drum_skill_learning/README.md) | Task environments, training, evaluation and action adapters |

## Verify

```bash
bash projects/cartesian_console/scripts/check_setup.sh
bash projects/cartesian_console/scripts/check_setup.sh --gui
```

cuRobo checks require generated planning assets. Asset-specific checks require
the separately imported asset. Physical motor verification is performed on the
actual reviewed hardware configuration.
