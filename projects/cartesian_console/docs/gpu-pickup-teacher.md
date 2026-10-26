# GPU pickup teacher

The GPU pickup trainer uses MuJoCo Warp 3.17.0 for batched physics and Stable-Baselines3 PPO on CUDA. The existing interactive demos continue to use native MuJoCo.

Install the optional dependencies in the existing environment:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e . -e './projects/cartesian_console[rl,gpu-sim,video]'
```

Run from the repository root:

```bash
TMPDIR="$PWD/runs/cartesian_console/cache" MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
openarm-train-pickup-gpu --num-envs 1024 --steps 50000000 \
  --save-every 5000000 --position-range 0.05 --normalize --ent-coef 0.01
```

The default policy uses 57 privileged state inputs, eight actions, two 128-unit Tanh layers, and the [pickup milestone reward](pickup-reward.md). Each action spans 20 ms of physics. The arm actions increment position targets by at most 0.025 radians; the final action controls gripper position. Gravity feedforward, actuator limits, finger coupling, free-stick dynamics, and physical contact remain active.

The default rollout contains 32 steps from each of 1024 worlds, with minibatches of 1024, ten PPO epochs, a learning rate of 0.0003, and entropy coefficient 0.01. A transition means one world taking one control step. VecNormalize standardizes observations and scales rewards by running discounted-return statistics. Both clipping thresholds are 10. Evaluation uses copied, frozen observation statistics and reports original rewards. Every model checkpoint includes `vecnormalize.pkl`; deployment must load it with the policy and disable statistic updates.

## Initial states

The trainer generates a finite bank of 1024 physically checked starting configurations. It samples stick X/Y within 0.05 m of the nominal position, TCP X/Y/Z within 0.05 m of a preparation point above the sampled stick, and drum X/Y/Z within 0.05 m of its nominal position. Drum samples intersecting the tabletop are rejected. IK failures, initial robot collisions, and preparation tracking errors above 10 mm are rejected. The bank stores settled joint configurations, position targets, and drum positions. Episode reset samples this bank on the GPU. TCP orientation and the initial stick orientation are fixed.

A separately generated bank with seed 1000 supports held-out evaluation. This is evaluation of sampled starting configurations within the same task distribution. The pickup observation contains stick state, robot state, reward progress, and relative drum position. The drum is an obstacle.

## Checkpoints and videos

The output defaults to repository `runs/cartesian_console/rl/pickup-teacher-milestones50m/`. It contains the exact run configuration, both reset banks, a CSV training log, and checkpoint directories. Each checkpoint contains `policy.zip`, `vecnormalize.pkl`, `evaluation.json`, `execution.mp4`, and `execution.json`. The MP4 records the first deterministic evaluation episode from the actual GPU state. The JSON trace contains the corresponding joint state and policy actions. Evaluation reports all 16 default evaluation episodes, including failures.

The callback saves at the first batch step crossing each 5000000-transition threshold. With 1024 worlds, the first checkpoint is at 5000192 transitions. Directory names record actual counts. PPO finishes complete rollouts, so a requested 50 million transitions ends at 50003968 with the default rollout size. Checkpoints are sampled during rollout collection, using the latest completed policy update.

Reset-bank construction and video rendering use the CPU MuJoCo model. Training physics and action stepping use GPU MuJoCo Warp. This backend currently implements pickup only. The [project README](../README.md) describes the separate ACT and student-supervision pipeline. Real-robot perception and hardware execution remain unvalidated.

## Weights & Biases

Install `.[tracking]` and authenticate with `wandb login`. Start live synchronization in another terminal:

```bash
TMPDIR="$PWD/runs/cartesian_console/cache" openarm-track-training \
  runs/cartesian_console/rl/pickup-teacher-milestones50m --project openarm-teachers
```

The tracker imports existing CSV history and continues polling every five seconds. Charts use `env_steps` as their horizontal axis. It records PPO losses, KL, entropy, training episode return and success rate, plus deterministic evaluation return, success rate, unsafe-contact rate, execution videos, and model artifacts. The run URL is stored in `wandb_url.txt`. A persistent run ID and uploaded-event list support continuing synchronization. The tracker requires online authentication and stops when final evaluation is available. Use `--once` to import an already completed run without watching it.
