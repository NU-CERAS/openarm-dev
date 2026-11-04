# Camera-free OpenDrummer skills

## Projects and installation

`projects/cartesian_console` owns TCP preview, drum annotation and ROS execution.
`projects/drum_skill_learning` owns proprioceptive policies, training, evaluation
and the tool runtime. `projects/opendrummer_playground` supplies the shared robot,
drums and collision geometry. Python imports remain `openarm_sim`, `openarm_rl`
and `openarm_drummer`. Generated artifacts belong under ignored `runs/`.

On Ubuntu, install the root ROS/console environment and learning dependencies:

```bash
bash setup/linux.sh --console
bash projects/drum_skill_learning/scripts/setup.sh --gpu
export OPENARM_PYTHON="$PWD/.venv/bin/python"
```

Use `--cpu` to install the CPU PyTorch build for deployment or small training
checks. CUDA training requires a compatible GPU/driver. The setup uses PyTorch
2.14.1, SB3 2.9.0 and MuJoCo 3.17.0. An existing Python 3.12 environment can be
selected explicitly with `OPENARM_PYTHON=/absolute/path/to/python`. ROS commands
require ROS Jazzy and `rclpy` in that same environment. Installation does not
modify kernel/driver packages.

Run `wandb login` in the selected environment before online training. W&B
project: `openarm-drum-skills`. API credentials belong in the terminal.

## Annotate a kit using the robot reference

```bash
bash projects/cartesian_console/scripts/run_demo.sh --model opendrummer \
  --layout runs/cartesian_console/annotations/drum_layout.json
```

The console reads encoder feedback on startup. Its TCP is each drumstick's
striking end. Coordinates are centimeters from the fixed torso-top center:
X forward, Y left, Z up. Opening the console commands no movement.

1. Open **Annotate drum layout** and select a drum.
2. Enter its center coordinates or enable **Drag selected drum in preview** and
   drag an X/Y/Z arrow. These edits move the simulated drum head and associated
   collision geometry. Robot movement still requires **Execute action**.
3. To register a physical drum without external measurements, use TCP preview
   and explicit execution to approach an operator-observed center. When the tip
   is at the chosen center, select **Place center at measured stick tip**.
4. Move the tip to a clear ready pose above the drum using the normal TCP controls.
   Disable drum dragging while moving the TCP. Execute and wait for feedback
   synchronization, then choose **Capture measured ready joints**.
5. Choose **Save layout and ready poses**. The file records both arms' measured
   angles and whether the feedback came from simulation or physical hardware.

The editable preview uses drum-head proxies. Head normals/radii and stand
placement retain the shared asset's values. Translation annotation does not
estimate physical surface tilt or reconstruct stands. Changing a drum center
clears that drum's recorded ready pose. Saving updates the owned MuJoCo test
robot while it is idle. A physical robot's environment is represented by this
operator-provided map.

Layout files bind robot/import/contact/scene versions and the torso reference.
Training snapshots bind the saved layout hash. Changed layouts require a fresh
preflight. Captured ready poses are checked for an actual clear center stroke;
an incompatible capture is reported as unavailable. With no captured pose,
preflight generates a fixed simulation reference using IK. No IK is run on each
randomized training reset.

## Train independent drum policies

```bash
bash projects/drum_skill_learning/scripts/run_training.sh \
  --layout runs/cartesian_console/annotations/drum_layout.json \
  --output runs/rl/my_drum_skills --device cuda --num-envs 4 \
  --steps-per-policy 131072 --position-range 0.005 --video
```

Omit `--layout` to evaluate the shared source kit. All seven stick targets are
preflighted. Left targets use the left arm; right targets use the right arm.
Kick is a pedal task. Unavailable centers are recorded with their reason.
Trainable targets run sequentially, at most four native CPU physics environments
per CUDA actor. One compute scope reserves 8 GiB system RAM and 4 GiB GPU memory,
limits CPU work to eight cores and bounds process memory. Close other guarded
console/training jobs before launching this command.

For headless MP4 recording, install `xvfb` and `xauth` through the system package
manager and add `--xvfb /usr/bin/Xvfb`. The recorder creates an authenticated
private display, selects software GLFW rendering, and closes it on completion.
Without `--xvfb`, `--video` requires a working graphical display.

The policy controls a J7 residual around a fixed timed joint reference. Its
maximum residual is 0.08 rad with a 0.015 rad change limit per 10 ms control
step. Residual authority fades during retraction and is zero during final
recovery. References, residual limits, command rate and normalization travel
with each policy. With generated references, the other arm uses a collision-
checked zero parking pose. Captured references retain their recorded posture.

| Actor input per history frame | Values |
| --- | ---: |
| Both arms' encoder positions and velocities | 28 |
| Nominal reference joint positions and velocities | 28 |
| Previous scalar action, elapsed time, requested strike speed | 3 |
| Eight frames at 100 Hz | 472 total |

The actor and critic each use two 128-unit Tanh layers. The actor's Gaussian
action is Tanh-squashed, with initial standard deviation `exp(-2)`. The critic
receives a separate 75-value observation, including the current actor frame,
relative tip/target geometry, surface normal, domain scales, layout shift and
contact force. The exported actor contains only its own network and observation
normalization. Tests alter privileged inputs and verify identical actor output.

Randomization includes the kit's XY translation, active-arm ready-joint error,
initial velocities, mass/inertia, friction, PD gains, encoder noise, 0–30 ms
command delay and 50/100 Hz encoder updates. The nominal gravity model remains
fixed and uses encoder positions. It does not read randomized physical parameters.
The reference advances by time and encoder checks, including when no contact
occurs. Position/velocity motor commands are generated by shared training/runtime
code. The ROS controller must expose both command interfaces and matching gains.

Reward gives an impulse-confirmed impact bonus weighted by center/timing/speed
quality, a discrete qualified-strike-and-recovery bonus, and penalties for
missing/extra hits, extended contact, unsafe contacts, failed recovery, effort
and action changes. A qualified strike requires one contact event, center error
at most 1 cm, impact timing within 60 ms, speed error at most 0.3 m/s, accumulated
contact duration at most 120 ms and encoder-confirmed recovery. Force and
penetration thresholds remain in the profile. A plane crossing gives no hit reward.

PPO uses normalized actor/critic observations and rewards, gamma 0.99,
learning rate 1e-4, entropy coefficient 0.001, 256 rollout steps per environment,
batch size 256, five epochs and target KL 0.01. Validation seeds start at 700000;
held-out seeds start at 800000. Reference and policy use the same seed sets.
Checkpoint selection excludes a policy with no optimization updates.

Each target writes W&B curves, CSV metrics, validation/held-out comparisons,
checkpoint/best/final bundles, MP4 and actual command traces. `tools.json`
contains relative artifact paths. A bundle includes `actor.pt`, normalization,
timed reference, command limits/gains, nominal gravity model, source identities
and SHA-256 checksums. Export verification compares actor actions during native
physics rollouts. Copy the complete output directory to preserve the scene and
profiles alongside all bundles.

`simulation_qualified` requires the configured minimum held-out success
(default 80%), success at least as high as reference, zero observed unsafe
episodes and at least 95% recovery. `trained_unqualified` is retained for explicit
experiments. Simulation qualification describes the tested domain and seeds.
Exact center strikes after an unobserved arbitrary 5 cm drum translation require
additional location information or another operator annotation.

## Preview and execute tools

```bash
bash projects/drum_skill_learning/scripts/run_tools.sh \
  --manifest runs/rl/my_drum_skills/tools.json --list
bash projects/drum_skill_learning/scripts/run_tools.sh \
  --manifest runs/rl/my_drum_skills/tools.json --tool strike_snare
bash projects/drum_skill_learning/scripts/run_tools.sh \
  --manifest runs/rl/my_drum_skills/tools.json --tool strike_snare --execute
```

`--execute` is explicit. Add `--allow-experimental` to inspect/run a trained
unqualified bundle. Missing or unavailable policies fail at the request; no
reference-controller substitution occurs. Native simulation tools keep one robot
state and clock across calls. Preparation starts at measured joints and uses
collision-checked transfers; the actor begins only after the fixed ready pose
has settled. Robot motion is serialized across tool calls.

The one-command ROS simulation demo starts an owned execution backend, optionally
opens the console, runs the requested tool and closes its owned services:

```bash
bash projects/drum_skill_learning/scripts/run_demo.sh \
  --manifest runs/rl/my_drum_skills/tools.json --tool strike_snare --gui --execute
```

For an agent process, `run_tools.sh --serve --execute` exposes the same function
schemas and reads JSON Lines requests. Each request names a fixed-arm tool and
includes `arguments.target_speed_m_s` and zero `arguments.timing_offset_s`.
Requests must include `execute: true` to request execution; otherwise they preview.

## Physical ROS backend

Install learning with `--cpu` in the root ROS environment when the host is only
running actors. Configure the existing hardware driver with the actual arm
mapping and a position/velocity trajectory controller. The example
`configs/deployment/ros_position_velocity.yaml` describes the controller
interfaces supported by [ROS 2 Jazzy](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html).
The shared vendor driver exports position, velocity and effort commands; its
activation behavior must be reviewed before powering the actual robot.

Copy `configs/deployment/drum_joint_mapping.example.json` under `runs/`, fill
directions, zero offsets, fourteen position/velocity gains, speed/acceleration
limits and confirm the hardware mapping and robot-side stream watchdog. These
values must match the controller used during training. Fill
`robot_source_sha256`, `import_config_sha256` and `layout_sha256` from the reviewed
bundle's `skill.json` scene identity (`robot`, `import`, and optional `layout`).
Use an empty layout hash only for an unannotated scene. The example remains
unreviewed and cannot enable physical writes.

```bash
bash projects/drum_skill_learning/scripts/run_tools.sh \
  --manifest runs/rl/my_drum_skills/tools.json --backend ros --namespace /robot \
  --mapping runs/rl/reviewed_mapping.json --physical --tool strike_snare
```

Inspect this preview before adding `--execute`. Execution reads fresh ROS
encoders, prepares the ready pose through the standard trajectory action, streams
the actor's position/velocity commands at 100 Hz through the controller's
`joint_trajectory` topic, checks encoder age/limits/tracking and reads final
feedback. Console and policy commands share a host-side ownership lock. The
ROS action service runs trajectory execution asynchronously while its event
loop continues handling feedback, cancellation and command subscriptions. The
OpenDrummer simulation advances in an independent process with its own clock.
Both-arm command packets are applied together and expire after 100 ms. ROS
feedback is received on a background executor; the policy control loop reads the
latest state without waiting for ROS callbacks. The bundle's encoder-age limit
and the 50 ms control-period limit remain enforced. Execution reports include
the command count, maximum control period and maximum observed feedback age.
The simulation backend has a 100 ms stream watchdog; the physical controller needs
its separately verified watchdog and hardware emergency stop. The current CAN
console adapter retains its v1.0 configuration; use the reviewed ROS controller
for OpenDrummer policy streaming.

The actor receives no camera or simulator object state on this path. Encoders
verify robot motion and recovery. Physical hit/acoustic quality remains an
operator observation until an appropriate feedback source is available. Physical
task success and controller timing require actual hardware commissioning.

## Record a combined demonstration

For a presentation focused on the OpenDrummer stick-tip TCP and coordinate point
registration, use:

```bash
python -m openarm_rl.recording.stick_tcp_demo \
  --manifest runs/rl/my_drum_skills/tools.json \
  --output runs/rl/stick_tcp_demo \
  --xvfb /usr/bin/Xvfb
```

This records actual ROS simulation execution from zero through a collision-checked
working-pose transfer, two stick-tip TCP drags with explicit execution, measured
point registration and saved-layout reload. A yellow tip marker, measured-motion
trail and centimeter readout identify the controlled point. The final section
shows the four policy evaluation results and records two exported actors at the
fixed evaluation seed 800000 with a closer camera. Playback is labeled at 0.5x
speed, and each recorded episode's outcome is reported.
The registered point is an operator-workflow demonstration in simulation.
The output is `drumstick_tcp_calibration_demo.mp4` with a verification report.

With the ROS Python environment active, Xvfb installed, and an evaluated tools
manifest available, run:

```bash
python -m openarm_rl.recording.compilation \
  --manifest runs/rl/my_drum_skills/tools.json \
  --output runs/rl/combined_demo \
  --xvfb /usr/bin/Xvfb
```

The output directory must be new. The recorder starts isolated ROS simulation
backends, records actual TCP preview/execution and drum annotation interactions,
and appends the four existing held-out policy videos. It writes
`openarm_changes_demo.mp4`, chapter timestamps and operation checks in
`report.json`. The annotation segment captures the current simulated encoder
pose to demonstrate saving; strike preparation poses require their separate
feasibility checks. No physical robot is commanded. Status cards identify
experimental policies and pending hardware/continuous-stream verification.

## Run verification checks

To inspect the measured causes of failed held-out episodes:

```bash
python -m openarm_rl.evaluation.strike_failures \
  --manifest runs/rl/my_drum_skills/tools.json \
  --output runs/rl/my_drum_skills/failure_analysis.json
```

The report checks each stored success label against the recorded measurements
and requires matching reference/policy evaluation seeds. It counts center,
timing, speed, contact-duration, recovery, unsafe-contact and encoder failures.
Counts can overlap within an episode. Requested-speed groups are
`[0.35, 0.5)`, `[0.5, 0.7)` and `[0.7, 0.85]` meters per second. Analysis reads
evaluation artifacts and does not start training or robot execution.

```bash
TMPDIR="$PWD/runs/rl/cache" "$OPENARM_PYTHON" -m pytest \
  projects/cartesian_console/tests/test_drum_annotation.py \
  projects/drum_skill_learning/tests/envs/test_camera_free_strike.py \
  projects/drum_skill_learning/tests/deployment/test_drum_actor.py
```

These checks use actual MuJoCo geometry, contact, PPO updates and exported actors.
The deployment tests train a small native policy to exercise bundle/runtime
behavior; they do not establish policy accuracy. ROS checks additionally require
the activated ROS environment.
