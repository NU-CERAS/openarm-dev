# RL and ACT policy deployment

The deployment package converts policy output into the existing ArmCommand
interface and executes it through native MuJoCo or the shared SocketCAN backend.
It supports continuous seven-joint position commands and calibrated gripper
commands at the policies' 50 Hz control rate.

## Action formats

| Mode | Eight input values | Processing |
| --- | --- | --- |
| absolute | Seven joint targets in rad, gripper model coordinate in m | Joint-limit and 0.025 rad per-step clipping |
| joint_delta | Eight normalized values in [-1, 1] | Add 0.025 rad times the arm action to the previous target; map the gripper to [0, 0.044] m |
| pickup_residual | Eight normalized residuals plus eight ACT nominal values | Apply the configured residual smoothing and scales, then training-time action limits |
| strike_residual | Eight normalized residuals plus seven nominal joint values | Add the arm residual at 0.08 rad scale; map the strike gripper to [0, 0.0066] m |

Pickup residual defaults match the current robust teacher: smoothing 0.8,
joint scale 0.06 rad and gripper scale 0.004 m. Supply matching PolicySettings
for a different trained teacher. Absolute ACT commands are the output of
ACTController.command after model normalization and relative-command conversion.
The adapter takes that decoded output.

For online PPO inference, restore the checkpoint's VecNormalize statistics,
disable training and reward normalization, normalize the actor's observation
and pass its predicted action to the appropriate adapter mode. The caller
provides the actual camera/state/reference inputs required by the actor.
The adapter supplies measured joint/gripper feedback. It performs no object
perception or oracle-state estimation.

PolicyAdapter is a context manager. On entry it reads feedback and initializes
the previous command from the measured robot pose. Each step checks feedback
age, action age, cadence, tracking error and permitted speed before writing.
Use the interface's clock: time.monotonic for CAN and sim.data.time for native
MuJoCo. The current models require a 20 ms step; commands arriving less than
19 ms or more than 30 ms after the previous step terminate execution.

## Replay an actual policy recording in MuJoCo

Install the project from the repository root:

~~~bash
bash projects/cartesian_console/scripts/setup.sh
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
openarm-policy-replay runs/cartesian_console/policy_demo/policy_trace.jsonl \
  --mode absolute
openarm-policy-replay runs/cartesian_console/policy_demo/policy_trace.jsonl \
  --mode absolute --backend sim --initialize-sim --execute \
  --output runs/cartesian_console/deployment/pickup_replay.json
openarm-policy-replay runs/cartesian_console/policy_demo/strike/policy_trace.jsonl \
  --mode strike_residual --backend sim --initialize-sim --execute \
  --output runs/cartesian_console/deployment/strike_replay.json
~~~

Generate the traces with `openarm-record-policy --output runs/cartesian_console/policy_demo`
using the [demonstration reproduction guide](getting-started.md).
The default replay command only
inspects the selected episode and prints its required starting joints and
gripper position. --episode selects a recorded episode, starting at zero.
The initial state is loaded into MuJoCo only with --initialize-sim. Simulation
replay checks command transport and robot tracking in the tabletop scene.
Task success is assessed in the task environments and the policy demonstration.
After the recorded commands, replay holds the last target for 0.5 seconds before
closing the interface. --hold-seconds changes this settling duration.

## Configure physical execution

Use the reviewed two-arm configuration from the [console guide](sync-console.md).
Joint order and limits are validated against the shared v1.0 model. Optional
motor_zero_rad and sign values define the reversible model/motor mapping for
both the console and policy adapter.

Install the pinned CAN dependency and activate the environment:

~~~bash
bash projects/cartesian_console/scripts/setup.sh --hardware
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
~~~

Copy configs/policy_gripper.example.json under runs/ and fill the actual motor
type, CAN IDs, gains, feedback/command timeouts, temperature and speed/tracking
limits. The gripper IDs must be distinct from the arm IDs on its bus. Set
hardware_reviewed to true after commissioning these values.

The model_positions_m table covers the simulated finger coordinate [0, 0.044].
motor_positions_rad contains the corresponding measured motor angles. Include
intermediate calibration points when needed. The motor-angle table can increase
or decrease but must be strictly monotonic. The adapter uses interpolation for
command conversion and its inverse for feedback. All placeholders must be filled
before any CAN connection is opened.

The robot's actual starting joints and gripper must match the selected trace
within 0.03 rad and 0.003 m. Use the existing planned-motion workflow to establish
that starting pose. Physical initialization from a recorded trace is rejected.

~~~bash
openarm-policy-replay /path/to/policy_trace.jsonl \
  --mode absolute --backend can \
  --hardware-config runs/cartesian_console/sync/hardware.json \
  --gripper-config runs/cartesian_console/deployment/gripper.json \
  --execute --supported \
  --output runs/cartesian_console/deployment/real_replay.json
~~~

CAN writes use the shared Hardware class and MIT position control with reviewed
kp/kd, zero target velocity and zero feedforward torque. Arm and gripper motor
feedback are refreshed for each command. The other arm's pose is monitored.
The selected arm's motors and gripper stay enabled during the command stream;
completion, cancellation, exceptions and command timeout release them. Support
both arms mechanically and retain the motor/controller watchdog configuration
required by the repository's hardware guide. The host-side watchdog terminates
a stream when new commands stop arriving.

The CAN speed limit comes from the reviewed arm configuration. A speed violation
terminates execution. Use matching constraints during policy training and
evaluation. Current policies were trained with a 0.025 rad per-step arm limit,
which permits 1.25 rad/s; the shipped arm configuration permits 0.3 rad/s.
The gripper additionally checks calibrated motor target speed and tracking.

## Embed a live actor

Construct CanPolicyInterface(hardware_config, gripper_config, arm), then enter
PolicyAdapter(interface, interface.limits, settings) as a context manager.
Call adapter.read for measured ArmState and adapter.step(action, sampled_at,
reference) for each fresh policy result. Supply a monotonic timestamp recorded
when the actor's inputs were sampled. The adapter closes the CAN resources
when its context exits. Physical writes begin with the first valid step.

The visual pickup student needs a real RGB input for visual feedback control.
Oracle strike needs an explicit source for its object/contact/reference state.
Trace replay executes recorded commands and can be used when those inputs are
provided by a separately operated simulation. Robot tracking uses real encoder
feedback throughout replay.

Connected physical arm/gripper execution remains unverified until the actual hardware and reviewed
configuration are available.
