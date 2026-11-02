# OpenArm RL

Task environments, rewards, ACT imitation, residual PPO teachers, visual student
distillation, evaluation and policy deployment live in `openarm_rl`.
The project uses `openarm_sim` for existing v1.0 simulation/motion tools and
`openarm_drummer` for the OpenDrummer model, kit geometry and reference controller.
The Cartesian Console can run independently of the learning package.

## Directory layout

Paths below are relative to `projects/rl/`.

| Directory | Responsibility |
| --- | --- |
| `src/openarm_rl/envs/` | Gymnasium tasks, native/Warp physics wrappers, observations and rewards |
| `src/openarm_rl/models/` | ACT networks and squashed Gaussian PPO policy |
| `src/openarm_rl/imitation/` | Demonstration datasets, teacher queries, ACT residual composition and distillation |
| `src/openarm_rl/training/` | Training entry points and checkpoint callbacks |
| `src/openarm_rl/evaluation/` | Held-out pickup comparisons and marker ablation |
| `src/openarm_rl/pipelines/` | Serial teacher/student workflow orchestration |
| `src/openarm_rl/tracking/` | W&B synchronization |
| `src/openarm_rl/recording/` | Policy/reference video and feature recording |
| `src/openarm_rl/deployment/` | Action decoding and robot command/feedback adapters |
| `src/openarm_rl/demos/` | Interactive task previews |
| `configs/tasks/`, `configs/deployment/` | Task and actuator configuration templates |
| `scripts/` | Setup and strike launch only |
| `tests/envs/`, `tests/models/`, `tests/imitation/`, `tests/deployment/` | Checks grouped by component |

Generated runs, videos, datasets and checkpoints live under the ignored
repository directory `runs/rl/`.

## Setup

From the repository root, activate a Python 3.12 environment with the intended
PyTorch CPU/CUDA build, then run:

```bash
bash projects/rl/scripts/setup.sh
```

The script installs the local projects and the training/tracking dependencies.
CUDA is required for the default training device. Use `--device cpu` explicitly
for CPU inference/training. ROS is required only for ROS execution/console use.
Rendering needs a working MuJoCo OpenGL backend. W&B defaults to online logging;
authenticate with `wandb login`, or explicitly select `--wandb-mode offline` or
`--wandb-mode disabled`.

## OpenDrummer oracle strike teacher

```bash
bash projects/rl/scripts/run_strike.sh \
  --output runs/rl/strike_teacher --steps 262144 --num-envs 4 --video
```

The scene uses the fixed-stick OpenDrummer robot and the actual drum kit contact
proxies. The task is one left-arm snare strike followed by withdrawal and stable
recovery. Physics runs in native MuJoCo on CPU; the PPO network uses the requested
device. Separate worker processes have separate model/data instances. Up to eight
environments are permitted. A shared compute guard serializes compute jobs and
enforces the existing CPU, RAM and GPU reserves.

The nominal controller uses 7 cm ready and 4 cm approach clearances and plans
the arm approach and wrist stroke. The actor emits
one normalized J7 residual, scaled to 0.08 rad with a 0.015 rad/control-step slew
limit, at 100 Hz. Joints 1–6 and the other arm follow the reference. Effort-limited
PD plus bias compensation executes the joint commands. The actor sees 67 oracle
values: both-arm joint states, relative tip position/velocity, drum normal, desired
strike time/speed, nominal joints, prior action, phase and domain parameters.
This actor requires simulator state and is a teacher for subsequent distillation.

| Oracle input | Dimensions |
| --- | ---: |
| Both arms' joint positions and velocities | 28 |
| Stick-tip position relative to the snare and world velocity | 6 |
| Drum surface normal | 3 |
| Time until requested impact and requested impact speed | 2 |
| Nominal reference joint positions | 14 |
| Previous scalar action | 1 |
| Seven controller phases, one-hot encoded | 7 |
| Mass, friction and gain scales | 3 |
| Kit translation | 3 |
| Total | 67 |

The actor and critic each have two 128-unit Tanh hidden layers. The actor learns
a Gaussian mean and trainable standard deviation, followed by Tanh squashing.
The critic estimates the state value for PPO/GAE. Initial Gaussian standard
deviation is `exp(-2)`, approximately 0.135. This strike actor uses no RGB and
has no ACT initialization. ACT and visual-student workflows remain available
for the existing pickup task.

Each reset translates the whole kit by up to 5 cm in X/Y, selects an IK pose with
a feasible collision-checked J7 raise/impact arc, perturbs initial joints/velocities, and randomizes mass,
friction and PD gains by 10%. Requested timing and impact speed are randomized.
Positions use meters and joints use radians. Model source hashes and the compiled
scene are saved with each run.

Reward includes one contact-event bonus weighted by timing/speed quality,
a discrete qualified-hit-and-recovery bonus, and penalties for extra hits,
unsafe contacts, missing strikes, timeouts, effort and action changes. Tracking
shaping uses a discounted potential difference with zero terminal potential.
An impact requires measured tip/head contact force and impulse; crossing a plane
alone provides no impact reward. Force, penetration, tracking, joint limits and
tip speed retain the reference controller's failure checks.

PPO uses observation/reward normalization, `[128, 128]` actor/critic networks,
256 rollout steps per environment, batch size 256, five epochs, learning rate
`1e-4`, entropy coefficient `0.001`, gamma `0.99`, and target KL `0.01`.
Checkpoints default to every 32,768 environment steps. `--steps` rounds up to
complete rollouts. Use a new output directory for each experiment. To continue
from a saved policy and its matching normalization statistics, provide
`--resume-checkpoint /path/to/checkpoint`; `--steps` then counts additional steps.
The initial/resumed policy is evaluated and included in best-checkpoint selection.

Each run writes `run_config.json`, `progress.csv`, reference evaluations,
checkpoints, `best/`, `final/`, and held-out comparisons. Each checkpoint contains
`policy.zip` and matching `vecnormalize.pkl`. `--video` records actual reference,
final and best held-out physics in MP4 files, with joint/action traces. Validation
seeds start at 700000; held-out seeds start at 800000. The same seed sets evaluate
the zero-residual reference. `--eval-episodes` defaults to 16 and
`--heldout-episodes` to 64. `comparison.json` reports whether held-out success
improved; completion alone does not qualify a teacher for deployment.

## Existing pickup, ACT and legacy strike workflows

Install optional learning backends in the activated environment:

```bash
python -m pip install -e 'projects/rl[gpu-sim,imitation,tracking]'
openarm-collect-act --help
openarm-train-act --help
openarm-train-pickup-residual --help
openarm-distill-act --help
openarm-train-strike --help
openarm-record-policy --help
openarm-task-demo --scene pickup
openarm-task-demo --scene strike
```

`openarm-train-strike` retains the v1.0 free-stick environment; the new fixed-stick
task uses `openarm-train-opendrummer`. Pickup requires the v1.0 gripper model.
The serial `openarm-train-skills` workflow uses the OpenDrummer strike trainer
after its existing pickup-teacher and visual-student stages.
Existing datasets/checkpoints remain at their supplied paths under
`runs/cartesian_console/` or release downloads; new defaults use `runs/rl/`.
Public command names remain available after installing this package. Python
imports use `openarm_rl`; training-module invocations use
`python -m openarm_rl.training.<module>`.

The [policy deployment guide](docs/policy-deployment.md) describes the existing
50 Hz eight-value pickup/strike adapters. The OpenDrummer teacher's 100 Hz scalar
J7 action requires its own nominal reference and oracle inputs and has not been
commissioned on physical hardware. Its evaluation trace exports the actual 14
joint commands for inspection; those model joint names require reviewed mapping
before hardware execution.

## Checks

```bash
python -m pytest projects/rl/tests/envs/test_opendrummer.py
python -m pytest projects/rl/tests/envs/test_pickup.py projects/rl/tests/envs/test_residual_drumming.py
```

GPU pickup/ACT checks require the `gpu-sim` and `imitation` extras. Temporary
test files should use an ignored repository cache, for example
`TMPDIR="$PWD/runs/rl/cache"`. Test and training jobs share the compute budget.
