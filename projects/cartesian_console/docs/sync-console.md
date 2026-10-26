# Feedback-driven sim-to-real console

## Start the demo

From the repository root:

```bash
bash projects/cartesian_console/scripts/run_sync.sh
```

The launcher uses `$HOME/miniforge3/envs/mujoco/bin/python`. Set
`OPENARM_PYTHON` to select another environment with this project installed.
It sets the project cache directory, limits CPU threads, obtains the shared
compute slot and opens the English interface. No ROS services are needed for
the direct SocketCAN adapter. The robot transport, live twin, trajectory planner
and preview are started together and closed with the window.
The GUI holds the shared compute slot until it closes.

The default planner is OMPL RRTConnect with MuJoCo collision checks. It requires
no GPU, CUDA packages or ROS services. Select the optional CUDA planner with:

```bash
bash projects/cartesian_console/scripts/run_sync.sh --planner curobo
```

The default scene places OpenArm on a 1.6 x 1.3 m workbench with a 0.75 m
height above the floor. The table surface and robot mounting plane define
sim Z=0; robot kinematics retain their original root coordinates. The overview
opens automatically. The interface uses centimeters relative to the fixed
torso top center, identified by the `torso_top_reference` site on the fixed
chassis. The point is centered on the upper surface of the torso's top housing.
Its location in the v1.0 model is `(-0.00975, 0, 0.773)` meters in the robot root
frame, 77.3 cm above the mounting plane. The viewport marks this origin and its
+X/+Y/+Z axes. The definition is stored in `configs/body_reference.json`.

X points forward, Y points to the robot's left, and Z points up. For example,
`(40, -25, -30) cm` requests a TCP 40 cm forward, 25 cm to the right and
30 cm below the torso top center. This is the default target. The point remains
fixed as the arms move.
Internal planning and actuator interfaces retain their existing meter/radian units.
See [motion planning](motion-planning.md) for CPU software rendering, cuRobo
installation and direct IK mode.

The demo runs an independent native MuJoCo robot that executes joint commands.
Its joint positions and velocities drive a separate live twin. A third model
displays the trajectory preview. Previewing changes the preview model only.
This uses physical simulation and the same trajectory submission path as the
hardware adapter. It does not establish physical robot performance.

## Operator workflow

1. Select **Right arm** or **Left arm**. Enter X/Y/Z in centimeters relative to
   the torso top origin. Click the tabletop to select X/Y and automatically create
   a motion preview. Z remains the entered torso-relative coordinate.
   For typed coordinates, select **Preview reach** after editing.
2. Check **Preview** mode. The orange marker identifies the TCP target. Use
   **Live** to inspect measured robot posture. An unreachable or opposite-side
   point displays guidance and leaves **Move robot** disabled.
3. Select **Move robot**. The viewport switches to **Live** and the robot executes
   the checked trajectory. **Stop** cancels motion. Connecting, selecting a target,
   loading a position and previewing start no physical movement.
4. Observe the physical arm directly. Adjust the sim target and repeat until the
   physical position is satisfactory. The **Adjust** tab provides small XYZ
   increments in centimeters; **Use current TCP** displays the current TCP
   relative to the same torso top origin.
5. After reaching the target, enter a name under **Saved positions** and choose
   **Save this position**. The file stores the sim target, planned final joints,
   measured joints, model-derived TCP, execution state and backend. The file
   also records the reference site, its transform and target coordinates in cm.
6. **Load position** restores a saved target and creates a fresh preview from
   current feedback. It becomes available after a matching position is saved.
   Execution checks both arms against the preview start; changes above 0.03 rad
   reject the trajectory.

The viewport renders at its native pixel dimensions, initially up to 1280 x 760.
Resizing rebuilds the renderers for the available area, up to 1920 x 1080, without
enlarging a low-resolution image. Text and target markers use native Tk drawing.
Joint values and planner details are available in the **Feedback** tab.

Targets are stored in `runs/cartesian_console/sync/targets.json` by default.
`--targets PATH` selects another file. Each GUI session writes preview trajectories,
feedback JSONL and execution reports under `runs/cartesian_console/sync/sessions/`.
The session log also records selected target coordinates and planning rejections.
The console's default joint speed limit is 0.2 rad/s, with 0.5 rad/s² acceleration;
hardware's configured speed can reduce that limit further.

The scene coordinates are operator-selected simulation coordinates. No camera,
object detector or premeasured object position is required. Displayed TCP
coordinates are forward kinematics from the robot model and joint feedback;
they are not independently measured physical TCP positions. Collision checks
describe the current sim scene. Physical object locations are established
through the operator's execution trials.

## Fill the physical connection placeholders

Copy `configs/sync_hardware.example.json` into
`runs/cartesian_console/sync/hardware.json` and fill both arm entries:

| Field | Value to provide |
| --- | --- |
| `arms.left.interface` | Actual left-arm SocketCAN interface name |
| `arms.left.config` | Path to the reviewed left-arm JSON configuration |
| `arms.right.interface` | Actual right-arm SocketCAN interface name |
| `arms.right.config` | Path to the reviewed right-arm JSON configuration |

Relative configuration paths are resolved relative to `hardware.json`. Empty
placeholders produce an error before hardware connection. This adapter requires
separate CAN interfaces for the two arms so both actual poses are available for
the bimanual collision model. There is no assumption that an unobserved arm is
in its home pose.

Use the repository's `setup/linux.sh --hardware` instructions to install the
pinned OpenArm CAN bindings. The adapter and CAN interfaces must be configured
and active according to the actual hardware. Check interface names with
`ip -details link show type can`. Interface bitrate and device setup remain
hardware-specific placeholders; the launcher does not guess those values.

Use the existing `robots/v1.0/config/left.json` and `right.json` as configuration
templates. Verify motor types, IDs, joint signs/zero convention, limits, gains,
feedback timeout and temperature/tracking settings for the actual arms. The
shared hardware backend requires `hardware_reviewed: true` in the reviewed
copies. This concerns actuator configuration; it requires no measurement of
objects in the workspace.

Once those machine-specific settings are complete, this command starts the
transport, feedback synchronization, planner and GUI together:

```bash
bash projects/cartesian_console/scripts/run_sync.sh \
  --backend can \
  --hardware-config runs/cartesian_console/sync/hardware.json \
  --supported
```

The existing shared CAN backend commands seven arm motors. Grippers are unchanged.
It disables the selected arm motors after each trajectory, cancellation or
execution error, so both arms must have mechanical support. `--supported`
records that operating condition. Stop other arm controllers before connection;
the tool uses shared per-interface ownership locks. See the repository's
`docs/hardware.md` for watchdog and release behavior.

Closing the GUI cancels motion, waits for the transport to finish, closes the
CAN resources and exits the scope. An unhealthy bus, stale feedback, motor
fault, excessive tracking error or invalid plan stops execution and reports an
error. The GUI's cancel button is a software cancellation request.

## Reproduce the recorded demonstration

```bash
bash projects/cartesian_console/scripts/run_sync.sh \
  --record runs/cartesian_console/sync/tabletop_demo
```

This records `sync_demo.mp4`, the measured joint/TCP trace in `report.json`, and
a saved target in `targets.json`. Automatic recording is restricted to the
MuJoCo demo backend. `--smoke-test` exercises the actual Tk interface, preview,
execution, target saving/loading and shutdown, then exits automatically.

Physical CAN execution remains unverified until actual interface names,
reviewed configuration files and connected arms are available.

## Local verification

Tests exercise actual MuJoCo execution, measured feedback, preview isolation,
stale/changed starts, cancellation, saved targets and cross-process ownership
locks. The GUI smoke test runs preview, execution, save, load and shutdown.
The placeholder configuration is rejected before hardware connection.
