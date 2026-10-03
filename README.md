# OpenArm development environment and robotics labs

ROS 2 Jazzy + Pinocchio + the OpenArm description and ROS 2 packages, in a container.
Student code goes in `src/` (one folder per package). Upstream OpenArm code is pulled
into `src/vendor/` automatically and is not committed.

## Learning ROS, URDF, and Pinocchio

**Start with the [numbered learning labs](labs/README.md).** They use a small
two-link arm before introducing the full OpenArm model. No robot hardware is needed.

| Learn | Start here | Example code |
|---|---|---|
| ROS nodes and topics | [Lab 01](labs/01_nodes_and_topics.md) | [ROS Python nodes](src/ros_teaching_examples/ros_teaching_examples/) |
| URDF, frames, and Xacro | [Lab 02](labs/02_urdf_and_rviz.md) | [Two-link robot](src/teaching_arm_description/) |
| Parameters and services | [Lab 04](labs/04_parameters_and_services.md) | [Joint motion node](src/ros_teaching_examples/ros_teaching_examples/joint_motion.py) |
| Pinocchio kinematics and dynamics | [Lab 05](labs/05_pinocchio_model.md) | [Numerical examples](src/pinocchio_teaching_examples/pinocchio_teaching_examples/demos.py) |
| ROS + Pinocchio together | [Lab 10](labs/10_ros_and_pinocchio.md) | [Hand-position node](src/pinocchio_teaching_examples/pinocchio_teaching_examples/hand_position.py) |

If your container was created before these examples were added, run `cb` then `cs`.
After setup, start the teaching viewer from a container terminal:

```bash
ros2 launch teaching_arm_description display.launch.py
```

Open [the browser desktop](http://localhost:6080/vnc.html) with password `openarm`.
Stop other robot viewers before launching this one. Each lab has runnable examples,
an exercise, expected results, and discussion questions. The guide includes
[starter exercises](labs/exercises/) and a shorter recommended learning path.

## Setup (everyone)

1. Install **Docker** (Docker Desktop on Windows/Mac) and **VS Code** with the **Dev Containers** extension.
2. Clone this repo and open the folder in VS Code.
3. Press `F1` -> **Dev Containers: Reopen in Container**. The first build takes 10-20 minutes
   (it downloads ROS and builds the OpenArm packages). Later opens take seconds.
4. In the container terminal: `bash scripts/verify.sh`

## Starting without VS Code

```bash
docker compose -f .devcontainer/compose.yaml up -d --build
docker compose -f .devcontainer/compose.yaml exec dev bash scripts/post-create.sh
docker compose -f .devcontainer/compose.yaml exec dev bash
```

The setup script imports missing upstream repositories, installs dependencies, builds
the workspace, and runs the health check. Interactive Bash terminals source ROS and
the workspace automatically. The desktop uses software OpenGL on both ARM64 and AMD64;
no XQuartz installation is required on macOS. Only localhost can access port 6080.

Stop the desktop with `docker compose -f .devcontainer/compose.yaml stop`.

## Seeing RViz and other GUIs

Open http://localhost:6080/vnc.html in a browser (password: `openarm`), then launch GUI tools
from the VS Code terminal; windows appear in the browser desktop.

```bash
ros2 launch openarm_description display_openarm.launch.py arm_type:=v10
```

If that launch file or the `arm_type` argument has changed upstream, look in
`src/vendor/openarm_description/launch/` and the OpenArm docs (docs.openarm.dev).

## Mock-hardware bringup (no arm needed)

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=v10 use_fake_hardware:=true
```

## Everyday commands

| Alias / command | What it does |
|---|---|
| `cb` | build the workspace (skips real-hardware package and its metapackage) |
| `cs` | source the workspace after a build |
| `bash scripts/verify.sh` | health check |
| `bash scripts/setup_vcan.sh` | create virtual CAN bus `vcan0` |

## Troubleshooting

- **RViz is slow:** expected, it uses software rendering.
- **`vcan0` fails:** the host kernel needs the `vcan` module. Works on native Linux
  (`sudo modprobe vcan` on the host); may not on Docker Desktop or WSL2.
- **Build fails on openarm_*:** upstream `main` moved. Pin versions in `workspace.repos`.
- **Topics from other people show up:** shouldn't happen (discovery is localhost-only);
  tell a lead if it does.
