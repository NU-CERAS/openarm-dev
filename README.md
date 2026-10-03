# OpenArm prerequisite project: dev environment

ROS 2 Jazzy + Pinocchio + the OpenArm description and ROS 2 packages, in a container.
Student code goes in `src/` (one folder per package). Upstream OpenArm code is pulled
into `src/vendor/` automatically and is not committed.

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
