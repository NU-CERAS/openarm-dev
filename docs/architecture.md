# Architecture

Commands in `scripts/` and the `openarm` entry point share `src/openarm_tools/cli.py`. Reusable behavior lives in configuration, backend, motion, calibration and recording modules. Projects remain independent clients of these modules.

## Coordinates and state

Select a model explicitly with `--robot v0.3` or `--robot v1.0` (default). `--arm left|right` selects seven arm joints; grippers are excluded from these commands. Configurations define exact joint names, motor IDs/types, signs, raw joint limits, gains, default home and command limits. Hardware and simulation cannot silently share calibration.

The CAN adapter applies configured signs to motor radians. There is currently no model-specific motor-to-joint gear conversion; only use direct-drive configurations matching this convention. Hardware configuration must be commissioned for the actual robot.

`zero` records software offsets. It sends no firmware zero command. Status includes both raw and relative positions. Home returns to that recorded zero, or the shipped home if no calibration exists. Raw joint limits remain unchanged. Saved poses contain raw joint positions and a configuration fingerprint, so recording a new zero does not reinterpret old poses. Configuration edits invalidate existing calibration/poses rather than silently using stale data.

## Simulation

Each invocation loads the model and selected arm's session snapshot from `runs/sim/<session>/<version>/<arm>/`. Commands manipulate that local simulation and save it after execution. Status/previews do not advance or write it. Left and right sessions are separate snapshots of the full model, not a synchronized two-arm service. A GUI session owns its snapshot while open; close it before using another command on the same session. `sim --reset` resets physics, not calibration or saved poses; choose a new session for fresh calibration as well.

MuJoCo steps physics. Torque-actuated v0.3 models use a basic bounded PD plus bias-force controller; v1.0 uses the model's position servos. These are starter controllers, not tuned physical-arm controllers. Motion commands report tracking error; reaching a target exactly is not guaranteed. The standalone actuator example intentionally applies a sinusoidal torque.

`sim --ros` is a separate process-based launcher for the existing v1.0 ROS-control package. The common session commands do **not** control that running ROS graph. Use the preserved `simple_motion.py` ROS example or ROS topics for that workflow. The ROS launch publishes simulation time.

## Source ownership

Keep upstream submodules intact. Canonical team-owned robot assets live under `robots/`; the ROS package installs the v1.0 assets into its package share during a colcon build. Rebuild after model changes. Version v0.3 and v1.0 have different joint names and must not be interchanged.

Promote reusable logic from projects into the shared package. Store local calibration/results under ignored `runs/`; commit only reviewed configuration and small reproducible fixtures. Do not commit generated `build/`, `install/`, `log/`, `.venv/` or `.pixi/` folders.
