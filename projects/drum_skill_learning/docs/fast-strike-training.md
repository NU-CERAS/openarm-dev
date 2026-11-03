# Fast drum-center ACT and PPO training

Run commands from the repository root. This experimental workflow trains one
fixed-arm policy per drum. Its results are recorded independently for all seven
targets; completing training does not imply simulation qualification or physical
validation.

## Install and launch

Install the root Ubuntu console environment, then the learning dependencies:

```bash
bash setup/linux.sh --console
bash projects/drum_skill_learning/scripts/setup.sh --gpu
export OPENARM_PYTHON="$PWD/.venv/bin/python"
"$OPENARM_PYTHON" -m wandb login
```

Select an existing Python 3.12 environment by setting `OPENARM_PYTHON` before
setup. `--cpu` installs CPU PyTorch for inference or small checks. Rendering
requires a graphical display, or installed `xvfb` and `xauth` packages.

```bash
bash projects/drum_skill_learning/scripts/run_fast_training.sh \
  --output runs/rl/fast_drum_skills \
  --device cuda --num-envs 4 \
  --demonstrations 96 --act-steps 2500 \
  --ppo-steps 262144 --ppo-rounds 3 \
  --position-range 0.005 --domain-range 0.1 \
  --video --xvfb /usr/bin/Xvfb
```

Use a new output directory. Omit `--xvfb` when recording on an existing display;
omit both recording options for training without video. W&B defaults to online
logging. `--wandb-mode offline` and `--wandb-mode disabled` are explicit options.
`--targets snare` limits training to that target. Physics uses native MuJoCo on
CPU; CUDA runs the networks. Targets and training stages run sequentially under
the shared compute guard, with at most four environment workers, eight CPU cores,
an 8 GiB system-memory reserve and a 4 GiB GPU-memory reserve.

## Preparation and demonstrations

Preparation generates a separate training layout. It keeps the snare, both rack
toms and floor tom centers and searches for reachable positions for the hi-hat
and both cymbals. Each target retains its assigned arm. The source kit remains
available. `preparation/layout_changes.json` records translations in meters.
Geometry checks validate each center's wrist arc against the other drums.
Native rollouts then calibrate a bank of stroke references at different speeds.
The fast references have approximately one-second complete strike/recovery
cycles; the generated files record their actual duration.

The demonstration teacher uses OpenDrummer's `TaughtStrike` controller from
`projects/opendrummer_playground/scripts/taught.py`. Its ready and bottom poses
are generated for the current model and each drum center. It commands wrist
position and velocity, detects contact, and retracts. The teacher can read
simulator contact and dynamics information.

Each episode runs through native physics with randomized initial joint state,
kit translation, actuator gains, mass/friction, encoder sampling and control
delay. `--position-range` specifies the XY translation bound in meters.
Successful episodes supply ACT training samples. Collection retains outcomes
and seeds for rejected attempts and fails if it cannot collect the requested
number within eight times that number of attempts.

Each target's `ACT/train.h5` contains encoder/reference actor observations,
privileged critic observations, requested commands, nominal commands, actually
applied commands, episode boundaries and accepted/rejected seeds. Attributes
record the profile/controller source hashes and command units. Requested
commands precede shared command limits and delay; applied commands record what
the simulation received.

## Policy input and output

The actor receives eight frames of encoder and reference history at 100 Hz:

| Per-frame input | Values |
| --- | ---: |
| Both arms' joint positions and velocities | 28 |
| Reference positions and velocities | 28 |
| Previous two-component action | 2 |
| Elapsed time and requested speed | 2 |
| Eight-frame actor input | 480 |

The actor receives no camera, object pose or contact measurements. The critic
has a separate 76-value observation with privileged simulation measurements.
The nominal gravity model uses encoder positions and fixed model parameters.

ACT predicts chunks of wrist motor-position and velocity corrections to the
known reference. PPO initializes its actor from ACT outputs on collected states,
then optimizes a two-layer 128-unit Tanh network. Its two normalized outputs map
to wrist motor position in radians and velocity in radians per second. Shared
position, velocity, acceleration and tracking checks constrain execution. The
remaining joints follow the reference. An exported bundle includes the reference,
normalization, gains, limits, nominal gravity model and file checksums.

ACT-initialized PPO uses learning rate `3e-5`, entropy coefficient `1e-4`, initial
position/velocity log standard deviations `[-5, -4]`, observation/reward
normalization, gamma `0.99`, five epochs and target KL `0.01`. Checkpoints are
saved every 65,536 environment steps. Each round runs the requested PPO steps,
rounded to complete rollouts. Later rounds resume the saved best bundle.

Reward and qualification use measured contact events. A qualified episode
requires one hit within 1 cm of the center, timing error at most 60 ms, speed
error at most 0.3 m/s, contact duration at most 120 ms, safe contact and recovery.
Missing/extra hits, extended contact, unsafe contact, effort, failed recovery and
action changes receive penalties. Validation seeds start at 700000 and held-out
seeds at 800000; reference and learned policies use matching seeds.

## Results and execution

Inspect `logs/`, each target's `ACT/` and `PPO_*/` directories, and `tools.json`.
The latter records ACT results, PPO comparisons and artifact paths. A target is
`simulation_qualified` only with at least 80% held-out success, zero observed
unsafe episodes, at least 95% recovery and success at least equal to reference.
Other trained targets are `trained_unqualified`; the command reports failure
after processing all requested targets if any remain unqualified.

```bash
bash projects/drum_skill_learning/scripts/run_tools.sh \
  --manifest runs/rl/fast_drum_skills/tools.json --list
bash projects/drum_skill_learning/scripts/run_demo.sh \
  --manifest runs/rl/fast_drum_skills/tools.json \
  --tool strike_snare --gui --execute
```

Unqualified bundles require explicit `--allow-experimental`. Copy the complete
run directory to retain the scene and profiles. For operator annotation and
physical ROS command mapping, use the
[camera-free drumming guide](camera-free-drumming.md). Physical operation requires
matching drum registration, reviewed joint mapping and controller settings, and
hardware validation. An unobserved change to a physical drum's position requires
an updated annotation.
