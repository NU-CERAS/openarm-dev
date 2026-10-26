# Motion planning

## CPU: OMPL

OMPL 2.0.1 is installed with the base project. Run:

```bash
openarm-sync --planner ompl
openarm-reach --planner ompl
```

The planner solves a seven-joint goal with mink, then uses OMPL RRTConnect to
search for a collision-free joint path. MuJoCo checks the full robot geometry,
the inactive arm and configured world obstacles. Joint-space validity checks
use at most 0.002 rad Euclidean steps and a 3 mm planning clearance in a separate
collision model. Only exact solutions are accepted.

The simplified path is checked again. Its waypoint segments use monotone cubic
interpolation with zero waypoint velocities, preserving the checked straight
joint segments. Retiming enforces joint speed and acceleration limits. The
console then executes the same timed joint trajectory through either the native
MuJoCo plant or the shared hardware backend.

OMPL supplies the planning algorithms; MuJoCo supplies physics and collision
queries. MoveIt uses OMPL as its primary planning library. This project calls
OMPL directly and requires no ROS service for CPU planning. See the official
[OMPL Python guide](https://ompl.kavrakilab.org/python.html) and
[MoveIt planner documentation](https://moveit.ai/documentation/planners/).

## GPU: cuRobo

Install the optional CUDA dependencies and generate the robot collision model:

```bash
bash projects/cartesian_console/scripts/setup_curobo.sh
openarm-sync --planner curobo
```

The script exports assets from `robots/v1.0/models/scene.xml` into
`runs/cartesian_console/planning/`. The selected arm is movable while the other
arm and grippers remain fixed at their current model positions. cuRobo handles
trajectory optimization; returned joint names are reordered into the shared
robot configuration and the trajectory is retimed for execution.

## CPU: direct IK

```bash
openarm-sync --planner mink
```

This mode solves IK and checks direct joint interpolation for collisions. It
rejects obstructed paths. Select `ompl` when the arm needs to move around an
obstacle. Backend selection is explicit; rejected planning requests do not
switch algorithms automatically.

## Rendering on a CPU-only machine

The desktop launcher uses GLFW. Mesa software rendering can be selected with:

```bash
__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw \
  bash projects/cartesian_console/scripts/run_sync.sh --planner ompl
```

This uses a desktop display and requires Mesa/OpenGL libraries. Headless CPU
rendering can use MuJoCo's OSMesa backend when OSMesa is installed:

```bash
MUJOCO_GL=osmesa bash projects/cartesian_console/scripts/run_sync.sh \
  --planner ompl --record runs/cartesian_console/sync/tabletop_demo
```

Rendering configuration is independent of the motion-planning backend. The
OMPL path imports no Torch, Warp or cuRobo modules.

## Execution scope

Targets use the robot model's root coordinate frame. The tabletop console keeps
the robot mounting plane at sim Z=0. The live twin follows both arms' feedback;
the preview uses the same starting posture. Execution checks that neither arm
changed by more than 0.03 rad after preview.

Collision checks describe the configured sim scene. The operator observes the
physical arm directly and adjusts target coordinates through trials. Planning
with a held object requires an attached-object model; free-object transport is
rejected until that model is provided. Actuator configuration, limits and motor
release behavior are described in the synchronization guide.
